"""Extract latents from the canonical (full-train-data) nested-CV models.

The canonical model is the stage_f retrain on the full training population with
the modal winning HPs, with the seed chosen by stage_g (step1_specificity for
IDRNN).  This script loads that model and forward-passes the FULL training
participants + the held-out test cohort, saving per-subject latents in one
coherent latent space — the substrate for downstream simulation / env-decoding /
three-regression analyses.

For each (dataset, arch) it writes:
  final_plots/{dgp}{tag}/dataset{ID}/canonical/{arch}/latents_train.npy   (N_train, z_or_h)
  final_plots/{dgp}{tag}/dataset{ID}/canonical/{arch}/latents_test.npy    (N_test,  z_or_h)
where latents are per-subject time-means (matching the analyzer convention).

Usage:
  python extract_canonical_latents.py --dgp synthetic --hypothesis_z 1 \
      --dataset_ids 0 2 5 10 17 --arch both
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, ".")
from modelsandtraining import IDRNN, Decoder, LatentRNN_secondstep, AblatedRNN
from nested_cv.config import path_tag

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _canonical_root(dgp, dataset_id, arch, hypothesis_z, metric):
    """Directory holding the {arch}_choice.json / {arch}_spec.json files.

    IDRNN z-anchored sweeps live under synthetic{z_tag}/...; vanilla has no z
    so its canonical lives under the plain free-z path (synthetic/...).
    """
    tag = path_tag(hypothesis_z, metric) if arch == "idrnn" else ""
    base = f"final_plots/{dgp}{tag}"
    if dataset_id is not None:
        base = f"{base}/dataset{dataset_id}"
    return f"{base}/canonical"


def _load_choice_or_spec(root, arch):
    """Prefer {arch}_choice.json (has selected_seed); fall back to spec.
    JSONs sit directly under canonical/, while per-seed runs are in canonical/{arch}/runs/."""
    choice = os.path.join(root, f"{arch}_choice.json")
    spec   = os.path.join(root, f"{arch}_spec.json")
    if os.path.exists(choice):
        obj = json.load(open(choice))
        return obj, obj.get("selected_seed")
    if os.path.exists(spec):
        obj = json.load(open(spec))
        seeds = obj.get("seeds", [])
        return obj, (seeds[0] if seeds else None)
    return None, None


def _build_idrnn(cfg):
    mc = cfg["model_config"]
    enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"], hid=mc["enc_hidden"],
                n_tasks=mc.get("n_tasks"), task_emb_dim=mc.get("task_emb_dim", 0))
    dec = Decoder(in_dim=mc["dec_in_dim"], z_dim=mc["z_dim"], hid=mc["hidden"], A=mc["A"])
    for p in dec.parameters():
        p.requires_grad = False
    m = LatentRNN_secondstep(encoder=enc, hid=mc["hidden"], z_dim=mc["z_dim"],
                              in_dim=mc["dec_in_dim"], A=mc["A"], decoder=dec,
                              n_tasks=mc.get("n_tasks"), task_emb_dim=mc.get("task_emb_dim", 0),
                              reinit_decoder_per_block=mc.get("reinit_decoder_per_block", False))
    return m.to(DEVICE)


def _build_vanilla(cfg):
    mc = cfg["model_config"]
    block = False
    return AblatedRNN(hid=mc["hidden"], in_dim=mc["dec_in_dim"], A=mc["A"],
                       block_structure=block,
                       n_tasks=mc.get("n_tasks"), task_emb_dim=mc.get("task_emb_dim", 0)).to(DEVICE)


def extract_one(dgp, dataset_id, arch, hypothesis_z, metric):
    root = _canonical_root(dgp, dataset_id, arch, hypothesis_z, metric)
    cdir = os.path.join(root, arch)   # where latents_train/test.npy go
    obj, seed = _load_choice_or_spec(root, arch)
    if obj is None or seed is None:
        print(f"  [{arch} ds{dataset_id}] no canonical choice/spec — skip")
        return False
    run_dir = obj.get("selected_run_dir") or os.path.join(obj["canonical_runs_dir"], f"seed_{seed}")
    cfg_p = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_p):
        print(f"  [{arch} ds{dataset_id}] config not found at {run_dir} — skip")
        return False
    cfg = json.load(open(cfg_p))
    epoch = cfg.get("cv_selected_epoch")
    # Canonical (no-fold) synthetic doesn't run inner CV → no cv_selected_epoch.
    # Fall back to the last checkpoint.
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if epoch is not None and os.path.exists(os.path.join(ckpt_dir, f"epoch{epoch:04d}.pt")):
        ckpt = os.path.join(ckpt_dir, f"epoch{epoch:04d}.pt")
    else:
        avail = sorted(int(f.replace("epoch","").replace(".pt","")) for f in os.listdir(ckpt_dir)
                       if f.startswith("epoch") and f.endswith(".pt"))
        if not avail:
            print(f"  [{arch} ds{dataset_id}] no checkpoints — skip"); return False
        ckpt = os.path.join(ckpt_dir, f"epoch{avail[-1]:04d}.pt")

    # Load full train + test data
    data_dir = f"data_dataset{dataset_id}"
    xin_train = torch.from_numpy(np.load(f"{data_dir}/xin_train.npy")).float().to(DEVICE)
    xin_test  = torch.from_numpy(np.load(f"{data_dir}/xin_test.npy")).float().to(DEVICE)

    model = _build_idrnn(cfg) if arch == "idrnn" else _build_vanilla(cfg)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
    model.eval()

    with torch.no_grad():
        if arch == "idrnn":
            xtr = xin_train.unsqueeze(1) if xin_train.dim() == 3 else xin_train
            xte = xin_test.unsqueeze(1)  if xin_test.dim()  == 3 else xin_test
            z_tr, _ = model.encoder(xtr, return_per_timestep=False)
            z_te, _ = model.encoder(xte, return_per_timestep=False)
            lat_tr = z_tr.cpu().numpy()       # (N, z_dim)
            lat_te = z_te.cpu().numpy()
        else:
            lat_tr = _vanilla_hidden(model, xin_train)
            lat_te = _vanilla_hidden(model, xin_test)

    os.makedirs(cdir, exist_ok=True)
    np.save(os.path.join(cdir, "latents_train.npy"), lat_tr)
    np.save(os.path.join(cdir, "latents_test.npy"),  lat_te)

    # Also emit the .pt files that analyze_synthetic_three_regressions.py expects
    # (model_state + per-session latents) so the rollout R1/R2/R3 analysis runs
    # unchanged on the canonical model.  It globs
    #   plots_dataset{ID}/step1_vs_vanilla/latents_{idrnn_step1,vanilla}_bestseed*.pt
    mc = cfg["model_config"]
    tr_dir = f"plots_dataset{dataset_id}/step1_vs_vanilla"
    os.makedirs(tr_dir, exist_ok=True)
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    if arch == "idrnn":
        torch.save({
            "z_dim": mc["z_dim"], "hidden": mc["hidden"],
            "base_in_dim": mc["dec_in_dim"], "A": mc["A"],
            "z": lat_tr,                       # canonical encoder mu, train sessions
            "model_state": state,
        }, os.path.join(tr_dir, "latents_idrnn_step1_bestseed_canonical.pt"))
    else:
        torch.save({
            "hidden": mc["hidden"], "A": mc["A"],
            "h": lat_tr,                       # canonical hidden, train sessions
            "model_state": state,
        }, os.path.join(tr_dir, "latents_vanilla_bestseed_canonical.pt"))

    print(f"  [{arch} ds{dataset_id}] seed={seed} epoch={os.path.basename(ckpt)} "
          f"train{lat_tr.shape} test{lat_te.shape} -> {cdir}/ + {tr_dir}/")
    return True


def _vanilla_hidden(model, xin):
    """Per-subject time-mean hidden state from AblatedRNN.

    AblatedRNN(block_structure=False).forward(xin) returns
    (logits, final_hid, hidden_traj); hidden_traj is (B, T, hid) — match the
    extraction used in decode_environment_per_seed.py.
    """
    with torch.no_grad():
        logits, _, hidden_tr = model(xin)
    if hidden_tr.dim() == 4:
        hidden_tr = hidden_tr.squeeze(1)
    return hidden_tr.mean(axis=1).cpu().numpy()   # (B, hid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgp", default="synthetic")
    ap.add_argument("--dataset_ids", type=int, nargs="+", required=True)
    ap.add_argument("--arch", choices=["idrnn", "vanilla", "both"], default="both")
    ap.add_argument("--hypothesis_z", type=int, default=None)
    ap.add_argument("--metric", default="cv_val_loss")
    args = ap.parse_args()
    archs = ["idrnn", "vanilla"] if args.arch == "both" else [args.arch]
    for did in args.dataset_ids:
        for arch in archs:
            extract_one(args.dgp, did, arch, args.hypothesis_z, args.metric)


if __name__ == "__main__":
    main()
