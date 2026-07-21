#!/usr/bin/env python3
"""Selection-free representational analysis: average the decoding result across
10 seeds of the final full-cohort model, for BOTH IDRNN and dim-matched Vanilla.

Motivation: if the representation is brittle across seeds (different selection
metrics pick different seeds with different readouts), the honest summary is the
across-seed mean ± SD of the decodability — not any single selected seed.

Per seed: IDRNN representation = step-1 lookup z (N, z_dim); Vanilla = h_avg
PCA-projected to z_dim (dim-matched).  Per target we compute LOO R^2 (out-of-
sample), then average over seeds.

Env: THAL_FULL (canonical dir), THAL_DATA (for subids + xin), N_SEEDS (default 10).
Outputs: {THAL_FULL}/decoding/seed_averaged_decoding.csv  + console table.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut

sys.path.insert(0, ".")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
from modelsandtraining import AblatedRNN
from decode_thalmann_canonical import load_targets, label_for, ALL_KEYS, pc_match, insample_ols

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FULL = os.environ.get("THAL_FULL", "final_plots/thalmann_z3_full")
DATA = os.environ.get("THAL_DATA", "data_thalmann_full")
N_SEEDS = int(os.environ.get("N_SEEDS", "10"))
Z_DIM = 3


def loo_r2_and_r(Z, y, alphas=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
    loo = LeaveOneOut(); preds = np.zeros(len(y))
    for tr, te in loo.split(Z):
        mz, sz = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        my, sy = y[tr].mean(), y[tr].std() + 1e-8
        preds[te] = RidgeCV(alphas=list(alphas)).fit((Z[tr]-mz)/sz, (y[tr]-my)/sy
                              ).predict((Z[te]-mz)/sz) * sy + my
    r2 = 1.0 - np.sum((y-preds)**2)/np.sum((y-y.mean())**2)
    return float(r2), float(pearsonr(y, preds)[0])


def best_abs_pearson(Z, y):
    return max(abs(pearsonr(Z[:, d], y)[0]) for d in range(Z.shape[1]))


def _seed_dirs(arch):
    return sorted(glob.glob(f"{FULL}/canonical/{arch}/runs/seed_*"),
                  key=lambda d: int(d.split("seed_")[1]))[:N_SEEDS]


def _idrnn_z(run_dir):
    return np.load(os.path.join(run_dir, "step1_z_lookup.npy"))


@torch.no_grad()
def _vanilla_h(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))["model_config"]
    ck = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "epoch*.pt")),
                key=lambda p: int(p[-7:-3]))[-1]
    sd = torch.load(ck, map_location=DEVICE)
    sd = sd["model_state"] if isinstance(sd, dict) and "model_state" in sd else sd
    m = AblatedRNN(hid=cfg["hidden"], in_dim=cfg["dec_in_dim"], A=cfg["A"],
                   block_structure=True, n_tasks=cfg["n_tasks"], task_emb_dim=cfg["task_emb_dim"]).to(DEVICE)
    tid = torch.from_numpy(np.load(os.path.join(DATA, "task_ids_per_block.npy"))).long().to(DEVICE)
    m.set_task_ids(tid); m.load_state_dict(sd); m.eval()
    xin = torch.from_numpy(np.load(f"{DATA}/xin_train.npy")).float().to(DEVICE)
    _, _, h = m(xin)
    h = h.cpu().numpy(); N, Bk, T, H = h.shape
    valid = (xin[..., 0].cpu().numpy() != -100.0).reshape(N, -1); hf = h.reshape(N, Bk*T, H)
    hav = np.zeros((N, H), np.float32)
    for i in range(N):
        mm = valid[i]
        if mm.sum(): hav[i] = hf[i, mm].mean(0)
    return hav


METRICS = ["pearson", "spearman", "loo_r", "loo_r2", "best_pearson"]


def run_arch(arch, subids, targets):
    dirs = _seed_dirs(arch)
    print(f"[{arch}] {len(dirs)} seeds: {[int(d.split('seed_')[1]) for d in dirs]}")
    per = {k: {mm: [] for mm in METRICS} for k in ALL_KEYS}
    for run_dir in dirs:
        rep = _idrnn_z(run_dir) if arch == "idrnn" else pc_match(_vanilla_h(run_dir), Z_DIM)
        for key in ALL_KEYS:
            y = targets[key].values.astype(float)
            m = np.isfinite(y) & np.all(np.isfinite(rep), axis=1)
            if m.sum() < 30:
                continue
            R, yy = rep[m], y[m]
            r_ins, _, pred_ins = insample_ols(R, yy)          # in-sample multiple-R (sqrt R2)
            loo_r2, loo_r = loo_r2_and_r(R, yy)
            per[key]["pearson"].append(r_ins)
            per[key]["spearman"].append(float(spearmanr(pred_ins, yy)[0]))
            per[key]["loo_r"].append(loo_r)
            per[key]["loo_r2"].append(loo_r2)
            per[key]["best_pearson"].append(best_abs_pearson(R, yy))
    return per


def main():
    subids = np.load(os.path.join(DATA, "subids_full.npy")).astype(int)
    targets = load_targets(subids)
    idr = run_arch("idrnn", subids, targets)
    van = run_arch("vanilla", subids, targets)
    rows = []
    print(f"\n=== Seed-averaged metrics (mean ± SD across {N_SEEDS} seeds) — selection-free ===")
    print(f"  {'target':<14}{'IDRNN R2 mean(sd)':>22}{'frac>0':>8}{'Vanilla R2 mean(sd)':>22}{'frac>0':>8}")
    for key in ALL_KEYS:
        ir = np.array(idr[key]["loo_r2"]); vr = np.array(van[key]["loo_r2"])
        if len(ir) == 0:
            continue
        row = {"target": key}
        for arch, src in [("idrnn", idr), ("vanilla", van)]:
            for mm in METRICS:
                a = np.array(src[key][mm])
                row[f"{arch}_{mm}_mean"] = float(a.mean())
                row[f"{arch}_{mm}_sd"] = float(a.std())
            row[f"{arch}_frac_pos"] = float((np.array(src[key]["loo_r2"]) > 0).mean())
        rows.append(row)
        print(f"  {key:<14}{ir.mean():>+12.3f}({ir.std():.3f}){ (ir>0).mean():>8.0%}"
              f"{vr.mean():>+12.3f}({vr.std():.3f}){ (vr>0).mean():>8.0%}")
    out = f"{FULL}/decoding/seed_averaged_decoding.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
