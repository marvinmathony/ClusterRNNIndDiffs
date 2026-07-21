#!/usr/bin/env python3
"""Bayes factor for the data-scaling readout increase (3 tasks vs 1 task), IDRNN + vanilla,
for WM and Openness. Model-based paired BF tied exactly to ΔR²:

  Per participant i and group-trait t: e^K_{t,i} = (y_{t,i} − ŷ^K_{t,i})²  (standard LOO
  out-of-fold squared error at K tasks, seed- and combo-averaged predictions).
  d_i = mean_t[(e^1_{t,i} − e^3_{t,i}) / SS_tot_t].   Then Σ_i d_i = ΔR²_mean, so a
  JZS one-sample t-test BF on d_i tests whether the readout reliably improves 1→3 tasks.

Reuses the cached per-seed latents (no forward passes). Appends bf_increase to the
existing datascaling_bootstrap_delta.json.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import glob, json
import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from pingouin import bayesfactor_ttest
from decode_thalmann_canonical import load_targets

SUBSETS = ["t0", "t1", "t2", "t01", "t02", "t12", "t012"]
NTASK = {"t0": 1, "t1": 1, "t2": 1, "t01": 2, "t02": 2, "t12": 2, "t012": 3}
GROUPS = {"openness": ["BIG5_open"], "wm": ["WM_composite", "WM_WMU", "WM_SS"]}
ARCHES = ["idrnn", "vanilla"]
ALPHAS = (0.001, 0.01, 0.1, 1, 10, 100, 1000)
EPS = 1e-8
OUTDIR = "final_plots/thalmann_z3_datascaling"


def reps(sub, arch):
    return np.load(f"final_plots/thalmann_z3_ds_{sub}/decoding/_bootreps_{arch}.npy")  # (10,238,3)


def full_alpha(Z, y):
    Zs = (Z - Z.mean(0)) / (Z.std(0) + EPS); ys = (y - y.mean()) / (y.std() + EPS)
    return float(RidgeCV(alphas=list(ALPHAS)).fit(Zs, ys).alpha_)


def loo_preds(Z, y, alpha):
    n = len(y); pr = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        Xt = Z[tr]; yt = y[tr]
        mz = Xt.mean(0); sz = Xt.std(0) + EPS; my = yt.mean(); sy = yt.std() + EPS
        Xs = (Xt - mz) / sz; ys = (yt - my) / sy
        beta = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Z.shape[1]), Xs.T @ ys)
        pr[i] = (((Z[i] - mz) / sz) @ beta) * sy + my
    return pr


def main():
    ref = np.load("data_sub_t0_full/subids_full.npy").astype(int)
    tg = load_targets(ref)
    R = {}
    for arch in ARCHES:
        REP = {sub: reps(sub, arch) for sub in SUBSETS}
        for gname, gtraits in GROUPS.items():
            m = np.ones(len(ref), bool)
            for t in gtraits:
                m &= np.isfinite(tg[t].values.astype(float))
            for sub in SUBSETS:
                for s in range(10):
                    m &= np.all(np.isfinite(REP[sub][s]), axis=1)
            n = int(m.sum())
            contrib = np.zeros(n)                          # d_i accumulates over traits
            for t in gtraits:
                y = tg[t].values.astype(float)[m]
                sstot = np.sum((y - y.mean()) ** 2)
                # mean of per-model SQUARED ERRORS (matches the plotted mean-of-R² readout;
                # NOT the squared error of the mean prediction — that would be R²-of-ensemble)
                e = {}
                for K in (1, 3):
                    combos = [c for c in SUBSETS if NTASK[c] == K]
                    acc = np.zeros(n); cnt = 0
                    for sub in combos:
                        for s in range(10):
                            Z = REP[sub][s][m]
                            acc += (y - loo_preds(Z, y, full_alpha(Z, y))) ** 2; cnt += 1
                    e[K] = acc / cnt
                contrib += (e[1] - e[3]) / sstot
            contrib /= len(gtraits)                        # mean over traits
            t_stat = stats.ttest_1samp(contrib, 0.0).statistic
            bf = float(bayesfactor_ttest(t_stat, n))
            dR2 = float(contrib.sum())                     # = ΔR²_mean
            R.setdefault(gname, {})[arch] = dict(bf_increase=bf, t=float(t_stat), n=n, dR2=dR2)
            print(f"{gname:9s} {arch:8s} ΔR²(3-1)={dR2:+.4f}  t={t_stat:+.2f}  BF10={bf:.3g}")
    # merge into delta json
    p = f"{OUTDIR}/datascaling_bootstrap_delta.json"
    d = json.load(open(p))
    for g in R:
        for a in R[g]:
            d[g][a].update(R[g][a])
    json.dump(d, open(p, "w"), indent=2)
    print(f"\nUpdated {p} with bf_increase")


if __name__ == "__main__":
    main()
