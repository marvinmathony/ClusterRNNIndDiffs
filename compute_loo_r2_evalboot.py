#!/usr/bin/env python3
"""Panel-b LOO R² decoding (14 traits, 3-task model thalmann_z3_3task_full), recomputed with the
STABLE evaluation bootstrap on FIXED out-of-fold predictions (same logic now used for the
data-scaling climbs). Compute honest OOF LOO predictions ONCE per (arch, seed, trait) with
per-fold RidgeCV (alpha fitted per fold), then resample participants and recompute R² on the
fixed (y, ŷ). No refit inside the bootstrap → no fixed-alpha artifact.

Writes a SEPARATE csv (loo_r2_evalboot.csv) and prints a comparison vs the current
train-resample loo_r2_traintest_bootstrap.csv. Does NOT overwrite anything.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import glob, json
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from modelsandtraining import AblatedRNN
from decode_thalmann_canonical import load_targets, ALL_KEYS, pc_match

FULL = "final_plots/thalmann_z3_3task_full"
DATA = "data_thalmann_3task_full"
Z_DIM = 3
B = int(os.environ.get("B_BOOT", "500"))
ALPHAS = (0.001, 0.01, 0.1, 1, 10, 100, 1000)
EPS = 1e-8


def seed_dirs(arch):
    return sorted(glob.glob(f"{FULL}/canonical/{arch}/runs/seed_*"),
                  key=lambda d: int(d.split("seed_")[1]))[:10]


@torch.no_grad()
def vanilla_h(run_dir):
    cfg = json.load(open(f"{run_dir}/config.json"))["model_config"]
    ck = sorted(glob.glob(f"{run_dir}/checkpoints/epoch*.pt"), key=lambda p: int(p[-7:-3]))[-1]
    sd = torch.load(ck, map_location="cpu")
    sd = sd["model_state"] if isinstance(sd, dict) and "model_state" in sd else sd
    m = AblatedRNN(hid=cfg["hidden"], in_dim=cfg["dec_in_dim"], A=cfg["A"],
                   block_structure=True, n_tasks=cfg["n_tasks"], task_emb_dim=cfg["task_emb_dim"])
    m.set_task_ids(torch.from_numpy(np.load(f"{DATA}/task_ids_per_block.npy")).long())
    m.load_state_dict(sd); m.eval()
    xin = torch.from_numpy(np.load(f"{DATA}/xin_train.npy")).float()
    _, _, h = m(xin); h = h.numpy(); N, Bk, T, H = h.shape
    valid = (xin[..., 0].numpy() != -100.0).reshape(N, -1); hf = h.reshape(N, Bk * T, H)
    hav = np.zeros((N, H), np.float32)
    for i in range(N):
        mm = valid[i]
        if mm.sum():
            hav[i] = hf[i, mm].mean(0)
    return hav


def load_reps():
    reps = {}
    for arch in ("idrnn", "vanilla"):
        cache = f"{FULL}/decoding/_evalboot_reps_{arch}.npy"
        if os.path.exists(cache):
            reps[arch] = np.load(cache); continue
        dirs = seed_dirs(arch)
        if arch == "idrnn":
            arr = np.stack([np.load(f"{d}/step1_z_lookup.npy") for d in dirs])
        else:
            arr = np.stack([pc_match(vanilla_h(d), Z_DIM) for d in dirs])
        np.save(cache, arr); reps[arch] = arr
        print(f"  loaded {arch}: {arr.shape}")
    return reps


def loo_preds(Z, y):
    loo = LeaveOneOut(); preds = np.zeros(len(y))
    for tr, te in loo.split(Z):
        mz, sz = Z[tr].mean(0), Z[tr].std(0) + EPS
        my, sy = y[tr].mean(), y[tr].std() + EPS
        preds[te] = RidgeCV(alphas=list(ALPHAS)).fit((Z[tr]-mz)/sz, (y[tr]-my)/sy
                            ).predict((Z[te]-mz)/sz) * sy + my
    return preds


def main():
    print(f"B={B}  panel-b eval bootstrap (3task_full)")
    reps = load_reps()
    subids = np.load(f"{DATA}/subids_full.npy").astype(int)
    tg = load_targets(subids)
    rows = []
    for arch in ("idrnn", "vanilla"):
        for t in ALL_KEYS:
            y = tg[t].values.astype(float)
            m = np.isfinite(y)
            for s in range(10):
                m &= np.all(np.isfinite(reps[arch][s]), axis=1)
            n = int(m.sum())
            yv = y[m]
            preds = [loo_preds(reps[arch][s][m], yv) for s in range(10)]
            rng = np.random.default_rng(0); pool = np.empty(B * 10)
            for b in range(B):
                idx = rng.integers(0, n, n)
                yr = yv[idx]; sstot = np.sum((yr - yr.mean()) ** 2) + EPS
                for s in range(10):
                    pool[b * 10 + s] = 1.0 - np.sum((yr - preds[s][idx]) ** 2) / sstot
            rows.append(dict(target=t, arch=arch, n=n,
                             evalboot_mean=float(pool.mean()), evalboot_sd=float(pool.std())))
    df = pd.DataFrame(rows)
    df.to_csv(f"{FULL}/decoding/loo_r2_evalboot.csv", index=False)

    tt = pd.read_csv(f"{FULL}/decoding/loo_r2_traintest_bootstrap.csv").set_index("target")
    ev = df.pivot(index="target", columns="arch", values=["evalboot_mean", "evalboot_sd"])
    print("\n=== COMPARISON  (train-resample  vs  eval-bootstrap) ===")
    print(f"{'trait':14s}{'IDRNN train-resamp':>22s}{'IDRNN eval-boot':>20s}   |{'VAN train-resamp':>20s}{'VAN eval-boot':>18s}")
    for t in ALL_KEYS:
        im_tt, is_tt = tt.loc[t, "idrnn_mean"], tt.loc[t, "idrnn_sd"]
        vm_tt, vs_tt = tt.loc[t, "vanilla_mean"], tt.loc[t, "vanilla_sd"]
        im_ev = ev.loc[t, ("evalboot_mean", "idrnn")]; is_ev = ev.loc[t, ("evalboot_sd", "idrnn")]
        vm_ev = ev.loc[t, ("evalboot_mean", "vanilla")]; vs_ev = ev.loc[t, ("evalboot_sd", "vanilla")]
        print(f"{t:14s}{im_tt:>+9.3f}±{is_tt:.3f}{im_ev:>+11.3f}±{is_ev:.3f}   |{vm_tt:>+9.3f}±{vs_tt:.3f}{vm_ev:>+9.3f}±{vs_ev:.3f}")
    print(f"\nSaved {FULL}/decoding/loo_r2_evalboot.csv  (NOT overwriting loo_r2_traintest_bootstrap.csv)")


if __name__ == "__main__":
    main()
