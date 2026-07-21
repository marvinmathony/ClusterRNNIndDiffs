#!/usr/bin/env python3
"""Canonical (non-bootstrap) LOO decoding + tests per (trait, arch). Produces the BAR
(canonical R2), the Bayes-factor TEST, and the permutation p for the LOO-R2 panel. The
+/-SD error bars come from the companion compute_loo_r2_traintest_bootstrap_perfold.py.

For each real participant i (held out once): fit the leakage-free readout on the other
n-1 REAL participants -- per-fold PCA dim-match (vanilla h only), per-fold standardisation,
per-fold RidgeCV alpha -- predict i, then average yhat_i over the 10 seeds -> one honest
out-of-sample prediction per participant.

Per (trait, arch):
  canonical_r2 = 1 - SS_res/SS_tot          BAR; 'chance' = always predict trait mean ybar
  bf10_greater = ONE-SIDED JZS correlation BF, bayesfactor_pearson(r(yhat,y), n, 'greater')
                 -- the reported TEST of "decodes above chance" (positive association).
  perm_p       = permutation p (permute y vs the fixed yhat, one-sided, 10000 draws) --
                 robust cross-check, reported in the paper.
  bf10_twosided, bf10_ereduction -- TRANSPARENCY ONLY, do not use (see the two traps below).

Two LOO traps this design avoids (each makes a naive test ANTI-correlate with real signal):
  1. Per-participant error reduction e_i=(y_i-ybar)^2-(y_i-yhat_i)^2 is heavy-tailed for
     signal traits (weak t despite large R2) and carries a tiny CONSISTENT LOO offset for
     null traits (huge |t|) -> its one-sample t-BF is pathological. Use the correlation,
     not e_i.  (bf10_ereduction is kept only to demonstrate the failure.)
  2. Under LOO the NULL of r(yhat,y) is NOT 0 but strongly NEGATIVE: intercept-only
     yhat_i=(sum(y)-y_i)/(n-1) is exactly anti-linear in y_i (r=-1), and a shrunk ridge
     (null trait) gives r ~ -0.4..-0.6. A TWO-SIDED BF flags that artifact as evidence;
     the ONE-SIDED ('greater') BF tests positive association = decoding. NB pingouin's
     one-sided bayesfactor_pearson is numerically unstable for |r| >~ 0.5 (spurious huge
     BF for strong-negative-r null traits), so trust bf10_greater only where |r| <~ 0.5
     (covers every positive claim); perm_p is robust everywhere and is the primary arbiter.

Bar = canonical R2 (full-data plug-in), NOT the bootstrap mean (bootstrap-mean of a
nonlinear stat is biased low). Run via compute_loo_r2_canonical_bf.sbatch (cpu_p, joblib,
BLAS=1); ~1 min.

Outputs (all under {FULL}/decoding/):
  loo_r2_canonical_bf.csv  -- long: canonical_r2, pearson_r, bf10_greater, perm_p,
                              bf10_twosided, bf10_ereduction, per (target, arch)
  loo_r2_canonical_bf_yhat.npz  -- per-participant yhat and y
  loo_r2_panel.csv  -- WIDE, what the figure reads: idrnn_/vanilla_ {r2, sd (from _perfold),
                       bf (=bf10_greater), perm_p}, one row per trait
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("THAL_FULL", "final_plots/thalmann_z3_3task_full")
os.environ.setdefault("THAL_DATA", "data_thalmann_3task_full")
os.environ.setdefault("N_SEEDS", "10")

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import pingouin as pg
from pingouin import bayesfactor_pearson

import seed_averaged_representation as S
from decode_thalmann_canonical import load_targets, ALL_KEYS

FULL = os.environ["THAL_FULL"]; DATA = os.environ["THAL_DATA"]
Z_DIM = 3
ALPHAS = (0.001, 0.01, 0.1, 1, 10, 100, 1000)
EPS = 1e-8
N_JOBS = int(os.environ.get("N_JOBS", os.environ.get("SLURM_CPUS_PER_TASK", "4")))


def loo_yhat(Z, y):
    """Canonical LOO predictions: for each i, fit the leakage-free readout on the real
    other n-1 participants and predict i. Everything (PCA dim-match, standardisation,
    RidgeCV alpha) is fit on the train rows only. Returns yhat (n,)."""
    n = len(y)
    yhat = np.empty(n)
    idx = np.arange(n)
    for i in range(n):
        tr = idx[idx != i]                            # the real other n-1 (no resampling)
        Ztr, ytr = Z[tr], y[tr]
        z_te = Z[i]
        if Ztr.shape[1] > Z_DIM:                      # dim-match: PCA fit on TRAIN only
            pca = PCA(n_components=Z_DIM).fit(Ztr)
            Ztr = pca.transform(Ztr)
            z_te = pca.transform(z_te[None])[0]
        mz, sz = Ztr.mean(0), Ztr.std(0) + EPS
        my, sy = ytr.mean(),  ytr.std()  + EPS
        model = RidgeCV(alphas=list(ALPHAS)).fit((Ztr - mz) / sz, (ytr - my) / sy)
        yhat[i] = model.predict(((z_te - mz) / sz)[None])[0] * sy + my
    return yhat


def run_unit(reps_arch, y_full):
    """One (trait, arch) unit: seed-averaged canonical yhat, R2, and JZS BF vs chance."""
    m = np.isfinite(y_full)
    for rep in reps_arch:
        m &= np.all(np.isfinite(rep), axis=1)
    if m.sum() < 30:
        return None
    yv = y_full[m].astype(float); n = len(yv)
    yhat = np.mean([loo_yhat(rep[m], yv) for rep in reps_arch], axis=0)   # seed-averaged
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    e = (yv - yv.mean()) ** 2 - (yv - yhat) ** 2                          # per-participant reduction
    r2 = float(np.sum(e) / ss_tot)                                       # == 1 - SS_res/SS_tot (the bar)
    # Primary test: JZS correlation BF between the LOO prediction and the truth (robust to the
    # heavy-tail / LOO-offset pathologies of the e_i t-test; matches Panel b's machinery).
    r = float(np.corrcoef(yhat, yv)[0, 1])
    # One-sided ('greater') correlation BF is the correct test: under LOO the null of r is NOT 0 but
    # strongly NEGATIVE (a shrunk predictor -> yhat ~ the LOO mean (S-y_i)/(n-1), exactly anti-linear
    # in y_i -> r=-1 in the intercept-only limit), so a two-sided BF flags that artifact as evidence.
    bf_greater = float(bayesfactor_pearson(r, n, alternative="greater"))
    bf_twosided = float(bayesfactor_pearson(r, n))                       # transparency: shows the trap
    # Permutation cross-check (user's scheme): permute y vs the fixed yhat -> null centred ~0, robust.
    rng = np.random.default_rng(0)
    perm = np.array([np.corrcoef(rng.permutation(yv), yhat)[0, 1] for _ in range(10000)])
    perm_p = float((np.sum(perm >= r) + 1) / (len(perm) + 1))
    bf_ered = float(pg.ttest(e, 0.0)["BF10"].iloc[0])                    # pathological here; record only
    return dict(n=n, canonical_r2=r2, pearson_r=r, bf10_greater=bf_greater,
                bf10_twosided=bf_twosided, perm_p=perm_p, bf10_ereduction=bf_ered, yhat=yhat, y=yv)


def main():
    print(f"N_JOBS={N_JOBS}")
    reps = {"idrnn": [S._idrnn_z(d) for d in S._seed_dirs("idrnn")],
            "vanilla": [S._vanilla_h(d) for d in S._seed_dirs("vanilla")]}
    print(f"loaded seeds: idrnn={len(reps['idrnn'])} vanilla={len(reps['vanilla'])}")
    subids = np.load(os.path.join(DATA, "subids_full.npy")).astype(int)
    targets = load_targets(subids)

    units = [(k, a) for k in ALL_KEYS for a in ("idrnn", "vanilla")]
    res = Parallel(n_jobs=N_JOBS)(
        delayed(run_unit)(reps[a], targets[k].values.astype(float)) for (k, a) in units)

    rows = []; yh = {}
    print(f"{'target':<14}{'arch':<9}{'R2':>9}{'r':>8}{'BF_gt':>13}{'perm_p':>10}")
    for (k, a), d in zip(units, res):
        if d is None:
            continue
        rows.append(dict(target=k, arch=a, n=d["n"], canonical_r2=d["canonical_r2"],
                         pearson_r=d["pearson_r"], bf10_greater=d["bf10_greater"],
                         perm_p=d["perm_p"], bf10_twosided=d["bf10_twosided"],
                         bf10_ereduction=d["bf10_ereduction"]))
        yh[f"{a}_{k}_yhat"] = d["yhat"]; yh[f"{a}_{k}_y"] = d["y"]
        print(f"{k:<14}{a:<9}{d['canonical_r2']:>+9.3f}{d['pearson_r']:>+8.2f}"
              f"{d['bf10_greater']:>13.3g}{d['perm_p']:>10.4f}")
    out = f"{FULL}/decoding/loo_r2_canonical_bf.csv"
    long = pd.DataFrame(rows)
    long.to_csv(out, index=False)
    np.savez(f"{FULL}/decoding/loo_r2_canonical_bf_yhat.npz", **yh)
    print(f"\nSaved {out}  (+ _yhat.npz)")

    # Merged wide panel CSV for Panel c: canonical R2 (bar) + one-sided BF10 (annotation) +
    # perm_p (reported in paper) from this run, joined with the bootstrap SD (error bar) from
    # loo_r2_traintest_bootstrap_perfold.csv. One row per trait; idrnn_*/vanilla_* columns.
    ren = {"canonical_r2": "r2", "bf10_greater": "bf", "perm_p": "perm_p"}
    wide = long.pivot(index="target", columns="arch", values=list(ren))
    wide.columns = [f"{arch}_{ren[val]}" for val, arch in wide.columns]
    sd = pd.read_csv(f"{FULL}/decoding/loo_r2_traintest_bootstrap_perfold.csv").set_index("target")
    wide["idrnn_sd"] = sd["idrnn_sd"]; wide["vanilla_sd"] = sd["vanilla_sd"]
    panel_out = f"{FULL}/decoding/loo_r2_panel.csv"
    wide.reindex(ALL_KEYS).to_csv(panel_out)
    print(f"Saved {panel_out}")


if __name__ == "__main__":
    main()
