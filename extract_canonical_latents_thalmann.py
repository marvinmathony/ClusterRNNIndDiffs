#!/usr/bin/env python3
"""Extract canonical-model latents for Thalmann z=3 (panels b/c/e).

Reads the full-cohort (236-subject) retrain at
  final_plots/thalmann_z3_full/canonical/{idrnn,vanilla}/runs/seed_*/
picks the best seed (IDRNN: max step1_specificity; Vanilla: min cv_val_loss),
and writes per-arch bundles consumed by decode_thalmann_canonical.py (b/c) and
the cross-task-regret port (panel e).

Latent source = STEP-1 lookup z (REPLICATION §2): the per-subject embedding the
frozen decoder was co-trained against, saved by run_Q_model as step1_z_lookup.npy.
No step 2.5 for thalmann, so the final epoch{N}.pt decoder IS that frozen decoder.

Writes (final_plots/thalmann_z3_full/canonical/):
  idrnn/latents_idrnn_canonical.pt   {z_dim,hidden,base_in_dim,task_emb_dim,A,
                                       z_train(step-1),subids,model_state,seed}
  idrnn/latents_train_step1.npy      (236, z_dim)
  idrnn/idrnn_choice.json
  vanilla/latents_vanilla_canonical.pt {hidden,A,base_in_dim,task_emb_dim,
                                       h_avg_train,h_last_train,subids,model_state,seed}
  vanilla/latents_train.npy          (236, hidden)  [h_avg]
  vanilla/vanilla_choice.json
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
from modelsandtraining import IDRNN, Decoder, LatentRNN_secondstep, AblatedRNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# THAL_FULL / THAL_DATA let the same script serve the S1 (data_thalmann_full) and
# the pooled S1+S2 (data_thalmann_s2_full -> final_plots/thalmann_z3_s2_full) runs.
ROOT = os.environ.get("THAL_FULL", "final_plots/thalmann_z3_full") + "/canonical"
DATA_DIR = os.environ.get("THAL_DATA", "data_thalmann_full")


def _ckpt(run_dir, cfg):
    cdir = os.path.join(run_dir, "checkpoints")
    ep = cfg.get("cv_selected_epoch")
    if ep is not None and os.path.exists(os.path.join(cdir, f"epoch{ep:04d}.pt")):
        return os.path.join(cdir, f"epoch{ep:04d}.pt")
    avail = sorted(int(f[5:9]) for f in os.listdir(cdir)
                   if f.startswith("epoch") and f.endswith(".pt") and "pre_step25" not in f)
    return os.path.join(cdir, f"epoch{avail[-1]:04d}.pt")


IDRNN_SELECT = os.environ.get("IDRNN_SELECT", "nll")   # "nll" (cv_val_loss) | "specificity"


def _pick_seed(arch):
    """Vanilla -> min cv_val_loss.  IDRNN -> min cv_val_loss (NLL, default;
    leakage-free) or max step1_specificity (if IDRNN_SELECT=specificity)."""
    runs = f"{ROOT}/{arch}/runs"
    if arch == "idrnn" and IDRNN_SELECT == "specificity":
        key, better = "step1_specificity", max
    else:
        key, better = "cv_val_loss", min
    cands = []
    for d in sorted(glob.glob(f"{runs}/seed_*")):
        cp = os.path.join(d, "config.json")
        if not os.path.exists(cp):
            continue
        cfg = json.load(open(cp))
        if cfg.get(key) is None:
            continue
        cands.append((int(d.split("seed_")[1]), float(cfg[key]), d, cfg))
    if not cands:
        raise RuntimeError(f"no scored seeds under {runs}")
    best = better(cands, key=lambda t: t[1])
    return best  # (seed, score, run_dir, cfg)


def _load_state(ckpt):
    sd = torch.load(ckpt, map_location=DEVICE)
    if isinstance(sd, dict) and "model_state" in sd:
        sd = sd["model_state"]
    return sd


def extract_idrnn():
    seed, score, run_dir, cfg = _pick_seed("idrnn")
    mc = cfg["model_config"]
    state = _load_state(_ckpt(run_dir, cfg))
    z1 = np.load(os.path.join(run_dir, "step1_z_lookup.npy"))         # (236, z_dim)
    subids = np.load(os.path.join(DATA_DIR, "subids_full.npy")).astype(int)
    assert z1.shape[0] == subids.shape[0], (z1.shape, subids.shape)

    cdir = f"{ROOT}/idrnn"; os.makedirs(cdir, exist_ok=True)
    np.save(os.path.join(cdir, "latents_train_step1.npy"), z1)
    bundle = {
        "z_dim": mc["z_dim"], "hidden": mc["hidden"],
        "base_in_dim": mc["in_dim"], "task_emb_dim": mc["task_emb_dim"], "A": mc["A"],
        "z_train": z1, "subids": subids, "seed": seed,
        "model_state": {k: v.cpu() for k, v in state.items()},
    }
    torch.save(bundle, os.path.join(cdir, "latents_idrnn_canonical.pt"))
    json.dump({"selected_seed": seed, "score_key": "step1_specificity",
               "selected_seed_score": score, "selected_run_dir": run_dir,
               "hp": cfg.get("model_config")},
              open(os.path.join(cdir, "idrnn_choice.json"), "w"), indent=2)
    print(f"[idrnn] seed={seed} step1_specificity={score:.4f}  "
          f"z{z1.shape} dec_in={mc['in_dim']}+{mc['task_emb_dim']} hidden={mc['hidden']}")
    # sanity: decoder + task embedding present
    assert any(k.startswith("decoder.") for k in state) and "task_embedding.weight" in state


@torch.no_grad()
def _vanilla_hidden(model, xin):
    """h_avg, h_last per subject over valid trials. xin: (N,Bk,T,in_dim)."""
    logits, _, hidden_tr = model(xin)                   # block_structure=True
    h = hidden_tr.cpu().numpy()                          # (N,Bk,T,H)
    N, Bk, T, H = h.shape
    valid = (xin[..., 0].cpu().numpy() != -100.0).reshape(N, -1)   # (N, Bk*T)
    hf = h.reshape(N, Bk * T, H)
    h_avg = np.zeros((N, H), np.float32); h_last = np.zeros((N, H), np.float32)
    for i in range(N):
        m = valid[i]
        if m.sum() == 0:
            continue
        h_avg[i] = hf[i, m].mean(0)
        h_last[i] = hf[i, np.where(m)[0][-1]]
    return h_avg, h_last


def extract_vanilla():
    seed, score, run_dir, cfg = _pick_seed("vanilla")
    mc = cfg["model_config"]
    state = _load_state(_ckpt(run_dir, cfg))
    subids = np.load(os.path.join(DATA_DIR, "subids_full.npy")).astype(int)
    xin = torch.from_numpy(np.load(f"{DATA_DIR}/xin_train.npy")).float().to(DEVICE)
    # task_ids from the ACTIVE data dir (pooled=62 blocks, 3task=111, S1=31).
    task_ids = torch.from_numpy(np.load(os.path.join(DATA_DIR, "task_ids_per_block.npy"))).long().to(DEVICE)

    model = AblatedRNN(hid=mc["hidden"], in_dim=mc["dec_in_dim"], A=mc["A"],
                       block_structure=True, n_tasks=mc["n_tasks"],
                       task_emb_dim=mc["task_emb_dim"]).to(DEVICE)
    model.set_task_ids(task_ids)
    model.load_state_dict(state)
    model.eval()
    h_avg, h_last = _vanilla_hidden(model, xin)

    cdir = f"{ROOT}/vanilla"; os.makedirs(cdir, exist_ok=True)
    np.save(os.path.join(cdir, "latents_train.npy"), h_avg)
    bundle = {
        "hidden": mc["hidden"], "A": mc["A"],
        "base_in_dim": mc["dec_in_dim"] - mc["task_emb_dim"],   # raw feature dim (5)
        "task_emb_dim": mc["task_emb_dim"],
        "h_avg_train": h_avg, "h_last_train": h_last,
        "subids": subids, "seed": seed,
        "model_state": {k: v.cpu() for k, v in state.items()},
    }
    torch.save(bundle, os.path.join(cdir, "latents_vanilla_canonical.pt"))
    json.dump({"selected_seed": seed, "score_key": "cv_val_loss",
               "selected_seed_score": score, "selected_run_dir": run_dir},
              open(os.path.join(cdir, "vanilla_choice.json"), "w"), indent=2)
    print(f"[vanilla] seed={seed} cv_val_loss={score:.4f}  h_avg{h_avg.shape} hidden={mc['hidden']}")


if __name__ == "__main__":
    extract_idrnn()
    extract_vanilla()
    print("Done.")
