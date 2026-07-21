#!/usr/bin/env python3
"""Data-scaling held-out NLL: does adding training tasks improve predictive
accuracy?  For every task subset (runs_pooled/{tag}/{arch}/fold{F}/seed_{S}) we
forward each retrained model on its fold's held-out test, compute per-trial NLL
per ORIGINAL task, seed-average, and convert to distance-to-random
(chance_nll[task] - nll[task]; higher = better, comparable across tasks/subsets).

Each task appears in subsets of size 1/2/3 → plot distance-to-random vs n-tasks,
IDRNN vs dim-matched-irrelevant vanilla.  Expectation: IDRNN climbs (latent
integrates across tasks), vanilla flat (no cross-task individual-difference
mechanism).

Outputs final_plots/thalmann_z3_datascaling/datascaling_nll.csv (+ plot via
plot_datascaling_nll.py).  Validate with --only s2_2task (reproduces panel-a
IDRNN task0≈0.473 task1≈0.707).
"""
import argparse, glob, json, math, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
from modelsandtraining import IDRNN, Decoder, LatentRNN_secondstep, AblatedRNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SEEDS = int(os.environ.get("THAL_NLL_NSEEDS", "15"))
OUT = "final_plots/thalmann_z3_datascaling"
CHANCE = {0: math.log(2), 1: math.log(4), 2: math.log(2)}   # 2/4/2-armed

# tag -> (data_root, orig_tasks for contiguous ids 0..k-1)
SUBSETS = {"t0": [0], "t1": [1], "t2": [2], "t01": [0, 1],
           "t02": [0, 2], "t12": [1, 2], "t012": [0, 1, 2]}


def _build(cfg, arch):
    mc = cfg["model_config"]
    if arch == "vanilla":
        return AblatedRNN(hid=mc["hidden"], in_dim=mc["dec_in_dim"], A=mc["A"],
                          block_structure=True, n_tasks=mc["n_tasks"],
                          task_emb_dim=mc["task_emb_dim"]).to(DEVICE)
    enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"], hid=mc["enc_hidden"],
                n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"],
                continuous_encoder=mc.get("continuous_encoder", True))
    dec = Decoder(in_dim=mc["dec_in_dim"], z_dim=mc["z_dim"], hid=mc["hidden"], A=mc["A"])
    return LatentRNN_secondstep(encoder=enc, hid=mc["hidden"], z_dim=mc["z_dim"],
                                in_dim=mc["dec_in_dim"], A=mc["A"], decoder=dec,
                                n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"],
                                reinit_decoder_per_block=mc.get("reinit_decoder_per_block", True)).to(DEVICE)


def _ckpt(rd, cfg):
    cdir = os.path.join(rd, "checkpoints")
    ep = cfg.get("cv_selected_epoch")
    if ep is not None and os.path.exists(os.path.join(cdir, f"epoch{ep:04d}.pt")):
        return os.path.join(cdir, f"epoch{ep:04d}.pt")
    av = sorted(int(f[5:9]) for f in os.listdir(cdir)
                if f.startswith("epoch") and f.endswith(".pt") and "pre_step25" not in f)
    return os.path.join(cdir, f"epoch{av[-1]:04d}.pt") if av else None


def _load_state(p):
    sd = torch.load(p, map_location=DEVICE)
    return sd["model_state"] if isinstance(sd, dict) and "model_state" in sd else sd


@torch.no_grad()
def _nll_per_subject_per_cid(logits, c_t, task_ids):
    """Returns {contiguous_task_id -> (N,) per-trial NLL}."""
    logp = F.log_softmax(logits, dim=-1)
    valid = (c_t >= 0)
    c_safe = c_t.clone(); c_safe[~valid] = 0
    nll = -logp.gather(-1, c_safe.long().unsqueeze(-1)).squeeze(-1)
    nll = torch.where(valid, nll, torch.zeros_like(nll))
    vmask = valid.float()
    tid = torch.as_tensor(task_ids, device=logits.device)
    out = {}
    for cid in sorted(set(int(t) for t in task_ids)):
        bsel = (tid == cid).view(1, -1, 1).float()
        num = (nll * vmask * bsel).sum(dim=(1, 2))
        den = (vmask * bsel).sum(dim=(1, 2)).clamp(min=1)
        out[cid] = (num / den).cpu().numpy()
    return out


def _winner_run(tag, arch, fold):
    """Path to the fold's winning HP-search run dir (1 seed) for preview mode."""
    wj = f"winners_pooled/{tag}/{arch}/fold{fold}.json"
    if not os.path.exists(wj):
        return None
    cfg_name = json.load(open(wj)).get("cfg")
    rd = f"hp_pooled/{tag}/{arch}/fold{fold}/{cfg_name}"
    return rd if cfg_name and os.path.exists(os.path.join(rd, "config.json")) else None


def analyze_subset(tag, arch):
    if tag == "s2_2task":                         # validation alias
        orig = [0, 1]; data_root = "data_thalmann_s2"
    else:
        orig = SUBSETS[tag]; data_root = f"data_sub_{tag}"
    task_ids = np.load(f"{data_root}/fold0/task_ids_per_block.npy")
    # per-seed subject-averaged nll per contiguous id (pooled over folds).
    # HP_PREVIEW=1 -> use each fold's winner HP-search run (1 seed) as a fast
    # preview while the 15-seed retrains are still running.
    per_seed = {cid: [] for cid in sorted(set(task_ids.tolist()))}
    if os.environ.get("HP_PREVIEW") == "1":
        seed_runs = [{f: _winner_run(tag, arch, f) for f in range(3)}]
    else:
        seeds = sorted({int(d.split("seed_")[1]) for f in range(3)
                        for d in glob.glob(f"runs_pooled/{tag}/{arch}/fold{f}/seed_*")})[:N_SEEDS]
        seed_runs = [{f: f"runs_pooled/{tag}/{arch}/fold{f}/seed_{s}" for f in range(3)} for s in seeds]
    for srun in seed_runs:
        acc = {cid: [] for cid in per_seed}      # (n_subj_in_fold,) arrays across folds
        n_ok = 0
        for f in range(3):
            rd = srun[f]
            cp = os.path.join(rd, "config.json") if rd else "/nonexistent"
            if not os.path.exists(cp):
                continue
            cfg = json.load(open(cp)); ck = _ckpt(rd, cfg)
            if ck is None:
                continue
            xin = torch.from_numpy(np.load(f"{data_root}/fold{f}/xin_test.npy")).float().to(DEVICE)
            c_t = torch.from_numpy(np.load(f"{data_root}/fold{f}/c_test.npy")).float().to(DEVICE)
            tid = torch.as_tensor(task_ids, dtype=torch.long, device=DEVICE)
            m = _build(cfg, arch); m.load_state_dict(_load_state(ck), strict=False)
            m.set_task_ids(tid); m.eval()
            logits = m(xin, xin)[0] if arch == "idrnn" else m(xin)[0]
            d = _nll_per_subject_per_cid(logits, c_t, task_ids)
            for cid in acc:
                acc[cid].append(d[cid])
            n_ok += 1
        if n_ok == 0:
            continue
        for cid in per_seed:
            per_seed[cid].append(float(np.concatenate(acc[cid]).mean()))
    rows = []
    for cid, vals in per_seed.items():
        if not vals:
            continue
        ot = orig[cid]; v = np.array(vals)
        rows.append(dict(tag=tag, n_tasks=len(orig), arch=arch, orig_task=ot,
                         nll_mean=float(v.mean()), nll_sd=float(v.std()),
                         dist_mean=float(CHANCE[ot] - v.mean()), dist_sd=float(v.std()),
                         n_seeds=len(v)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="single tag (e.g. s2_2task to validate)")
    args = ap.parse_args()
    tags = [args.only] if args.only else list(SUBSETS.keys())
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for tag in tags:
        for arch in ("idrnn", "vanilla"):
            r = analyze_subset(tag, arch)
            rows.extend(r)
            for x in r:
                print(f"  {tag:8s} n={x['n_tasks']} {arch:7s} task{x['orig_task']}: "
                      f"nll={x['nll_mean']:.4f}±{x['nll_sd']:.3f} dist={x['dist_mean']:+.4f}")
    if not args.only:
        fn = "datascaling_nll_preview.csv" if os.environ.get("HP_PREVIEW") == "1" else "datascaling_nll.csv"
        pd.DataFrame(rows).to_csv(f"{OUT}/{fn}", index=False)
        print(f"\nSaved {OUT}/{fn} ({len(rows)} rows)")
