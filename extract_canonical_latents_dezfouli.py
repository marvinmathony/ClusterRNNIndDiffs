"""Extract canonical-model latents for Dezfouli.

Mirrors extract_canonical_latents.py but adapted for Dezfouli (no dataset_id,
2-dim [prev_choice, prev_reward] inputs, 12 blocks per session, optional step
2.5 fine-tune). Reads stage_f / stage_g outputs at:

    final_plots/dezfouli{path_tag}/canonical/{arch}_choice.json
    final_plots/dezfouli{path_tag}/canonical/{arch}_spec.json   (fallback)

Writes per-subject latents to the same canonical dir, plus a `.pt` bundle
(model state + latents) that analyze_dezfouli_three_regressions.py consumes.

Usage:
  python extract_canonical_latents_dezfouli.py --hypothesis_z 2 \
      --metric step1_specificity --arch both
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, ".")
from modelsandtraining import IDRNN, Decoder, LatentRNN_secondstep, AblatedRNN
from nested_cv.config import path_tag

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DGP    = "dezfouli"
DATA_DIR = "data_dezfouli"


def _canonical_root(hypothesis_z, metric, arch):
    """Canonical dir for Dezfouli — BOTH archs live under the {z,metric}-tagged
    path because stage_e writes both specs there (verified on the live run:
    final_plots/dezfouli_z2_spec/canonical/{idrnn,vanilla}/runs/)."""
    tag  = path_tag(hypothesis_z, metric)
    return f"final_plots/{DGP}{tag}/canonical"


def _load_choice_or_spec(root, arch):
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
                              n_tasks=mc.get("n_tasks"),
                              task_emb_dim=mc.get("task_emb_dim", 0),
                              reinit_decoder_per_block=mc.get("reinit_decoder_per_block",
                                                              True))
    return m.to(DEVICE)


def _build_vanilla(cfg):
    mc = cfg["model_config"]
    return AblatedRNN(hid=mc["hidden"], in_dim=mc["dec_in_dim"], A=mc["A"],
                       block_structure=True,
                       n_tasks=mc.get("n_tasks"),
                       task_emb_dim=mc.get("task_emb_dim", 0)).to(DEVICE)


def _vanilla_hidden_per_block(model, xin):
    """Return (h_avg, h_last) per subject across the session.

    xin: (N, 12, max_T, 2) — Dezfouli flat blocked input
    Returns:
      h_avg:  (N, hid) — time-averaged hidden state across all valid trials
      h_last: (N, hid) — last valid trial's hidden state, taken from the
                         final block (block index 11 in Dezfouli convention)
    """
    with torch.no_grad():
        logits, _, hidden_tr = model(xin)
    # hidden_tr shape: (N, Bk, T, hid)  — block_structure=True
    h = hidden_tr.cpu().numpy()
    N, Bk, T, H = h.shape

    # Mask: trials with valid input (xin != -100 on the choice dim)
    valid = (xin[..., 0].cpu().numpy() != -100.0)   # (N, Bk, T)
    valid_flat = valid.reshape(N, -1)
    h_flat     = h.reshape(N, Bk * T, H)

    h_avg = np.zeros((N, H), dtype=np.float32)
    h_last = np.zeros((N, H), dtype=np.float32)
    for i in range(N):
        m = valid_flat[i]
        if m.sum() == 0:
            continue
        h_avg[i]  = h_flat[i, m].mean(axis=0)
        # last valid trial across the session (typically end of block 11)
        last_idx  = np.where(m)[0][-1]
        h_last[i] = h_flat[i, last_idx]
    return h_avg, h_last


def extract_one(arch, hypothesis_z, metric):
    root = _canonical_root(hypothesis_z, metric, arch)
    cdir = os.path.join(root, arch)
    obj, seed = _load_choice_or_spec(root, arch)
    if obj is None or seed is None:
        print(f"  [{arch}] no canonical choice/spec at {root} — skip")
        return False
    run_dir = obj.get("selected_run_dir") or os.path.join(obj["canonical_runs_dir"],
                                                            f"seed_{seed}")
    cfg_p = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_p):
        print(f"  [{arch}] config not found at {run_dir} — skip")
        return False
    cfg = json.load(open(cfg_p))

    epoch = cfg.get("cv_selected_epoch")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if epoch is not None and os.path.exists(os.path.join(ckpt_dir, f"epoch{epoch:04d}.pt")):
        ckpt = os.path.join(ckpt_dir, f"epoch{epoch:04d}.pt")
    else:
        avail = sorted(int(f.replace("epoch","").replace(".pt",""))
                       for f in os.listdir(ckpt_dir)
                       if f.startswith("epoch") and f.endswith(".pt")
                       and "pre_step25" not in f)
        if not avail:
            print(f"  [{arch}] no usable checkpoints — skip"); return False
        ckpt = os.path.join(ckpt_dir, f"epoch{avail[-1]:04d}.pt")

    xin_train = torch.from_numpy(np.load(f"{DATA_DIR}/xin_train.npy")).float().to(DEVICE)
    xin_test  = torch.from_numpy(np.load(f"{DATA_DIR}/xin_test.npy")).float().to(DEVICE)

    model = _build_idrnn(cfg) if arch == "idrnn" else _build_vanilla(cfg)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
    model.eval()

    os.makedirs(cdir, exist_ok=True)
    mc = cfg["model_config"]

    if arch == "idrnn":
        with torch.no_grad():
            z_tr, _ = model.encoder(xin_train, return_per_timestep=False)
            z_te, _ = model.encoder(xin_test,  return_per_timestep=False)
        lat_tr = z_tr.cpu().numpy()   # (N_train, z_dim)
        lat_te = z_te.cpu().numpy()
        np.save(os.path.join(cdir, "latents_train.npy"), lat_tr)
        np.save(os.path.join(cdir, "latents_test.npy"),  lat_te)

        # .pt bundle for downstream rollout script
        state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save({
            "z_dim": mc["z_dim"], "hidden": mc["hidden"],
            "base_in_dim": mc["dec_in_dim"], "A": mc["A"],
            "z_train": lat_tr, "z_test": lat_te,
            "model_state": state,
        }, os.path.join(cdir, "latents_idrnn_canonical.pt"))
        print(f"  [idrnn] seed={seed} ckpt={os.path.basename(ckpt)}  "
              f"z_train{lat_tr.shape} z_test{lat_te.shape} -> {cdir}")
    else:
        h_avg_tr, h_last_tr = _vanilla_hidden_per_block(model, xin_train)
        h_avg_te, h_last_te = _vanilla_hidden_per_block(model, xin_test)
        np.save(os.path.join(cdir, "latents_train.npy"),       h_avg_tr)
        np.save(os.path.join(cdir, "latents_train_h_last.npy"), h_last_tr)
        np.save(os.path.join(cdir, "latents_test.npy"),        h_avg_te)
        np.save(os.path.join(cdir, "latents_test_h_last.npy"),  h_last_te)

        state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save({
            "hidden": mc["hidden"], "A": mc["A"],
            "base_in_dim": mc["dec_in_dim"],
            "h_avg_train":  h_avg_tr,  "h_avg_test":  h_avg_te,
            "h_last_train": h_last_tr, "h_last_test": h_last_te,
            "model_state": state,
        }, os.path.join(cdir, "latents_vanilla_canonical.pt"))
        print(f"  [vanilla] seed={seed} ckpt={os.path.basename(ckpt)}  "
              f"h_avg{h_avg_tr.shape} h_last{h_last_tr.shape} -> {cdir}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["idrnn", "vanilla", "both"], default="both")
    ap.add_argument("--hypothesis_z", type=int, default=None)
    ap.add_argument("--metric", default="cv_val_loss")
    args = ap.parse_args()
    archs = ["idrnn", "vanilla"] if args.arch == "both" else [args.arch]
    for a in archs:
        extract_one(a, args.hypothesis_z, args.metric)


if __name__ == "__main__":
    main()
