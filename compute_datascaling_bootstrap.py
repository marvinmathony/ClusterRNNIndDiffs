#!/usr/bin/env python3
"""Participant train-resample bootstrap for the data-scaling readout climbs
(WM + Openness), matching panels b/d's train-resample scheme.

Design (shared-participant, average-within-iteration):
  For each bootstrap iteration b: draw ONE participant train-resample T (shared
  across every subset, seed and group-trait, since all subsets share the same 238
  participants). For each #tasks K, average the LOO R² over the K-task combinations
  and the 10 seeds (and, for WM, over the 3 WM traits) → one R²_b(K). Repeat B times.
  Bar stays the existing combo+seed mean; error bar = SD of R²_b(K) across bootstraps.
  Increase test: Δ_b = R²_b(3) − R²_b(1) and the linear slope over K, summarised as
  mean, SD, 95% percentile CI and P(≤0).

Train-resample LOO (leakage-free): the test participant is held out, its training
fold is bootstrap-resampled, the ridge readout is refit (α fixed per trait/subset/
seed/arch from the full-data GCV choice), predict the held-out participant.

Outputs: final_plots/thalmann_z3_datascaling/datascaling_bootstrap.csv (+ _delta.json)
Caches the vanilla forward passes to <ds_dir>/decoding/_bootreps_vanilla.npy.
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
from modelsandtraining import AblatedRNN
from decode_thalmann_canonical import load_targets, pc_match

SUBSETS = ["t0", "t1", "t2", "t01", "t02", "t12", "t012"]
NTASK = {"t0": 1, "t1": 1, "t2": 1, "t01": 2, "t02": 2, "t12": 2, "t012": 3}
GROUPS = {"openness": ["BIG5_open"], "wm": ["WM_composite", "WM_WMU", "WM_SS"]}
ARCHES = ["idrnn", "vanilla"]
Z_DIM = 3
B = int(os.environ.get("B_BOOT", "500"))
ALPHAS = (0.001, 0.01, 0.1, 1, 10, 100, 1000)
EPS = 1e-8
OUTDIR = "final_plots/thalmann_z3_datascaling"


def fulldir(sub):  return f"final_plots/thalmann_z3_ds_{sub}"
def datadir(sub):  return f"data_sub_{sub}_full"
def seed_dirs(sub, arch):
    return sorted(glob.glob(f"{fulldir(sub)}/canonical/{arch}/runs/seed_*"),
                  key=lambda d: int(d.split("seed_")[1]))[:10]


@torch.no_grad()
def vanilla_h(run_dir, dd):
    cfg = json.load(open(f"{run_dir}/config.json"))["model_config"]
    ck = sorted(glob.glob(f"{run_dir}/checkpoints/epoch*.pt"), key=lambda p: int(p[-7:-3]))[-1]
    sd = torch.load(ck, map_location="cpu")
    sd = sd["model_state"] if isinstance(sd, dict) and "model_state" in sd else sd
    m = AblatedRNN(hid=cfg["hidden"], in_dim=cfg["dec_in_dim"], A=cfg["A"],
                   block_structure=True, n_tasks=cfg["n_tasks"], task_emb_dim=cfg["task_emb_dim"])
    m.set_task_ids(torch.from_numpy(np.load(f"{dd}/task_ids_per_block.npy")).long())
    m.load_state_dict(sd); m.eval()
    xin = torch.from_numpy(np.load(f"{dd}/xin_train.npy")).float()
    _, _, h = m(xin); h = h.numpy(); N, Bk, T, H = h.shape
    valid = (xin[..., 0].numpy() != -100.0).reshape(N, -1); hf = h.reshape(N, Bk * T, H)
    hav = np.zeros((N, H), np.float32)
    for i in range(N):
        mm = valid[i]
        if mm.sum():
            hav[i] = hf[i, mm].mean(0)
    return hav


def load_reps():
    """(sub, arch) -> (10, 238, 3) latents; vanilla forward passes cached to disk."""
    reps = {}
    for sub in SUBSETS:
        for arch in ARCHES:
            cache = f"{fulldir(sub)}/decoding/_bootreps_{arch}.npy"
            if os.path.exists(cache):
                reps[(sub, arch)] = np.load(cache)
                continue
            dirs = seed_dirs(sub, arch)
            if arch == "idrnn":
                arr = np.stack([np.load(f"{d}/step1_z_lookup.npy") for d in dirs])
            else:
                dd = datadir(sub)
                arr = np.stack([pc_match(vanilla_h(d, dd), Z_DIM) for d in dirs])
            np.save(cache, arr); reps[(sub, arch)] = arr
            print(f"  loaded {sub}/{arch}: {arr.shape}")
    return reps


def full_alpha(Z, y):
    Zs = (Z - Z.mean(0)) / (Z.std(0) + EPS); ys = (y - y.mean()) / (y.std() + EPS)
    return float(RidgeCV(alphas=list(ALPHAS)).fit(Zs, ys).alpha_)


def draw_T(n, rng):
    raw = rng.integers(0, n - 1, size=(n, n - 1)); ii = np.arange(n)[:, None]
    return raw + (raw >= ii)


def boot_loo_r2(Z, y, T, alpha):
    Xtr = Z[T]; ytr = y[T]
    mz = Xtr.mean(1, keepdims=True); sz = Xtr.std(1, keepdims=True) + EPS
    my = ytr.mean(1, keepdims=True); sy = ytr.std(1, keepdims=True) + EPS
    Xs = (Xtr - mz) / sz; ys = (ytr - my) / sy
    XtX = np.einsum("fnj,fnk->fjk", Xs, Xs); Xty = np.einsum("fnj,fn->fj", Xs, ys)
    beta = np.linalg.solve(XtX + alpha * np.eye(Z.shape[1])[None], Xty)
    Xte = (Z - mz[:, 0, :]) / sz[:, 0, :]
    yhat = np.einsum("fj,fj->f", Xte, beta) * sy[:, 0] + my[:, 0]
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def std_loo_r2(Z, y, alpha):  # standard LOO (no train resample) → real point estimate
    n = len(y); preds = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        Xt = Z[tr]; yt = y[tr]
        mz = Xt.mean(0); sz = Xt.std(0) + EPS; my = yt.mean(); sy = yt.std() + EPS
        Xs = (Xt - mz) / sz; ys = (yt - my) / sy
        beta = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Z.shape[1]), Xs.T @ ys)
        preds[i] = (((Z[i] - mz) / sz) @ beta) * sy + my
    return 1.0 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)


def pctl(a, q): return float(np.nanpercentile(a, q))


def main():
    print(f"B={B}")
    reps = load_reps()
    ref = np.load(f"{datadir('t0')}/subids_full.npy").astype(int)
    tg = load_targets(ref)
    rows, delta = [], {}
    for gname, gtraits in GROUPS.items():
        m = np.ones(len(ref), bool)
        for t in gtraits:
            m &= np.isfinite(tg[t].values.astype(float))
        for (sub, arch), arr in reps.items():
            for s in range(arr.shape[0]):
                m &= np.all(np.isfinite(arr[s]), axis=1)
        n = int(m.sum())
        ys = {t: tg[t].values.astype(float)[m] for t in gtraits}
        # α per (arch, sub, seed, trait); real point estimate per (arch, K)
        alpha = {}
        for arch in ARCHES:
            for sub in SUBSETS:
                for s in range(10):
                    Z = reps[(sub, arch)][s][m]
                    for t in gtraits:
                        alpha[(arch, sub, s, t)] = full_alpha(Z, ys[t])
        real = {arch: {} for arch in ARCHES}
        for arch in ARCHES:
            for K in (1, 2, 3):
                combos = [c for c in SUBSETS if NTASK[c] == K]
                cv = []
                for sub in combos:
                    sv = [np.mean([std_loo_r2(reps[(sub, arch)][s][m], ys[t], alpha[(arch, sub, s, t)])
                                   for t in gtraits]) for s in range(10)]
                    cv.append(np.mean(sv))
                real[arch][K] = float(np.mean(cv))
        # bootstrap (shared T across everything within group)
        rng = np.random.default_rng(0)
        boot = {arch: {K: np.empty(B) for K in (1, 2, 3)} for arch in ARCHES}
        for b in range(B):
            T = draw_T(n, rng)
            for arch in ARCHES:
                for K in (1, 2, 3):
                    combos = [c for c in SUBSETS if NTASK[c] == K]
                    cv = []
                    for sub in combos:
                        sv = [np.mean([boot_loo_r2(reps[(sub, arch)][s][m], ys[t], T, alpha[(arch, sub, s, t)])
                                       for t in gtraits]) for s in range(10)]
                        cv.append(np.mean(sv))
                    boot[arch][K][b] = np.mean(cv)
        delta[gname] = {}
        for arch in ARCHES:
            for K in (1, 2, 3):
                rows.append(dict(group=gname, arch=arch, ntasks=K, n=n,
                                 real_mean=real[arch][K], boot_mean=float(boot[arch][K].mean()),
                                 boot_sd=float(boot[arch][K].std())))
            d = boot[arch][3] - boot[arch][1]
            Km = np.array([1, 2, 3])
            sl = np.array([np.polyfit(Km, [boot[arch][k][b] for k in (1, 2, 3)], 1)[0] for b in range(B)])
            delta[gname][arch] = dict(
                delta_mean=float(d.mean()), delta_sd=float(d.std()),
                delta_lo=pctl(d, 2.5), delta_hi=pctl(d, 97.5), p_delta_le0=float((d <= 0).mean()),
                slope_mean=float(sl.mean()), slope_sd=float(sl.std()),
                slope_lo=pctl(sl, 2.5), slope_hi=pctl(sl, 97.5), p_slope_le0=float((sl <= 0).mean()))
        print(f"[{gname}] n={n}")
        for arch in ARCHES:
            r = [f"{real[arch][K]:+.4f}±{boot[arch][K].std():.4f}" for K in (1, 2, 3)]
            dd = delta[gname][arch]
            print(f"  {arch:8s} K1/2/3 = {r}  |  Δ(3-1)={dd['delta_mean']:+.4f} "
                  f"[{dd['delta_lo']:+.4f},{dd['delta_hi']:+.4f}] P(<=0)={dd['p_delta_le0']:.3f}")
    os.makedirs(OUTDIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{OUTDIR}/datascaling_bootstrap.csv", index=False)
    json.dump(delta, open(f"{OUTDIR}/datascaling_bootstrap_delta.json", "w"), indent=2)
    print(f"\nSaved {OUTDIR}/datascaling_bootstrap.csv + _delta.json")


if __name__ == "__main__":
    main()
