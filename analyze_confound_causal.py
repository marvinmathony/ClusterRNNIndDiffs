#!/usr/bin/env python3
"""Causal-direction tests for the coverage confound (vanilla readouts).

Competing accounts for vanilla's pooled-cohort trait signal:
  A (confound): the readout exploits the coverage axis (has_s2), which proxies
                traits via selective attrition.
  B (genuine):  two sessions of extraction data genuinely increase trait signal.

TEST 1 — coverage-axis ablation at CONSTANT data (all 238, full extraction):
  project out the per-fold coverage direction (returners-vs-non mean difference,
  train-fit) from vanilla's 15-dim h before the dim-matched readout. A predicts
  collapse; B predicts survival (1 of 15 dims cannot erase a distributed code).
  WM is n/a (its sample is 100% returners — structurally unconfoundable).
  NOT run for IDRNN: in a 3-dim z whose exploration axis itself correlates with
  attrition (r(PC1, has_s2) = −0.21), the 'coverage direction' aligns with the
  trait axis, so ablation amputates real signal — invalid test for compact codes
  (IDRNN's causal controls are the cohort-restriction and S1-extraction tests).

TEST 2 — extraction dose within the FIXED both-sessions cohort (n=175):
  same participants, S1-extracted vs both-session-extracted h. Isolates the pure
  'more data' effect with coverage held constant. If B holds for a trait, the
  both-session column must beat the S1 column.

Output: final_plots/thalmann_z3_s2_3task_full/decoding/confound_causal_tests.npz
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

FULL = "final_plots/thalmann_z3_s2_3task_full"
ALPHAS = (0.001, 0.01, 0.1, 1, 10, 100, 1000); EPS = 1e-8; ZD = 3
ABL_TRAITS  = ["PANAS_PA", "PANAS_NA", "CEI", "BIG5_open"]           # WM n/a (see above)
DOSE_TRAITS = ["PANAS_PA", "PANAS_NA", "CEI", "BIG5_open", "WM_composite"]

Zi   = np.load("final_plots/thalmann_z3_ds_t012/decoding/_bootreps_idrnn.npy")
Hall = np.load("final_plots/thalmann_z3_ds_t012/decoding/_bootreps_vanilla_rawh.npy")
Hs1  = np.load(f"{FULL}/decoding/_bootreps_vanilla_rawh_s1extract.npy")
tg   = pd.read_csv(f"{FULL}/decoding/trait_targets.csv")
meta = pd.read_csv("data_thalmann_s2/df_all.csv").set_index("subid").loc[tg["subid"]]
s2   = (meta["has_s2"] == 1).values.astype(float)
both = (meta["has_s1"] == 1).values & (s2 == 1)


def loo_yhat(Z, y, cov=None):
    """Standard per-fold readout; if cov is given, the coverage direction is removed
    INSIDE the per-fold PCA score space (i.e. from the representation the decoder
    actually sees), not from the raw latents — PCA is fit on the original train
    rows first, so the basis is identical to the baseline readout's."""
    n = len(y); yhat = np.empty(n); idx = np.arange(n)
    for i in range(n):
        tr = idx[idx != i]; Ztr, ytr, zte = Z[tr].copy(), y[tr], Z[i].copy()
        if Ztr.shape[1] > ZD:
            p = PCA(n_components=ZD).fit(Ztr); Ztr = p.transform(Ztr); zte = p.transform(zte[None])[0]
        if cov is not None:
            c = cov[tr]
            assert (c == 1).any() and (c == 0).any(), "coverage constant in fold"
            w = Ztr[c == 1].mean(0) - Ztr[c == 0].mean(0); w /= (np.linalg.norm(w) + EPS)
            Ztr = Ztr - np.outer(Ztr @ w, w); zte = zte - (zte @ w) * w
        mz, sz = Ztr.mean(0), Ztr.std(0) + EPS; my, sy = ytr.mean(), ytr.std() + EPS
        yhat[i] = RidgeCV(alphas=list(ALPHAS)).fit((Ztr - mz) / sz, (ytr - my) / sy
                                                   ).predict(((zte - mz) / sz)[None])[0] * sy + my
    return yhat


def r2(y, yh):
    return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)


print("TEST 1 — coverage-axis ablation (vanilla, all 238, full extraction):")
abl = []
for k in ABL_TRAITS:
    y = tg[k].values.astype(float); fin = np.isfinite(y); yv = y[fin]
    base = r2(yv, np.mean([loo_yhat(Hall[s][fin], yv) for s in range(10)], axis=0))
    ablt = r2(yv, np.mean([loo_yhat(Hall[s][fin], yv, cov=s2[fin]) for s in range(10)], axis=0))
    abl.append((base, ablt))
    print(f"  {k:<12} base {base:+.4f} → −covAxis {ablt:+.4f}")

print("TEST 2 — extraction dose (fixed both-sessions cohort, n=175):")
dose = []
for k in DOSE_TRAITS:
    y = tg[k].values.astype(float); sel = both & np.isfinite(y); yv = y[sel]
    r_s1  = r2(yv, np.mean([loo_yhat(Hs1[s][sel],  yv) for s in range(10)], axis=0))
    r_all = r2(yv, np.mean([loo_yhat(Hall[s][sel], yv) for s in range(10)], axis=0))
    dose.append((r_s1, r_all))
    print(f"  {k:<12} S1-extr {r_s1:+.4f} → both-extr {r_all:+.4f}")

# ── TEST 3: session-specific extraction at IDENTICAL data quantity ───────────
# S1 and S2 both have 111 blocks. h_S2 is recovered algebraically from the caches:
# h_all = sum_all/n_all, h_S1 = sum_S1/n_S1  →  h_S2 = (h_all·n_all − h_S1·n_S1)/n_S2.
# Environmental-variance account: S2-only >> S1-only for openness/WM despite equal
# data. Generic-reliability account: S1-only ≈ S2-only. Δh probes change-on-re-exposure.
xin = np.load("data_sub_t012_full/xin_train.npy")
valid = (xin[..., 0] != -100.0)                       # (238, 222, 200)
S1_BLOCKS = np.zeros(222, bool); S1_BLOCKS[0:30] = True; S1_BLOCKS[60] = True; S1_BLOCKS[62:142] = True
n_all = valid.reshape(238, -1).sum(1).astype(np.float64)
n_s1  = valid[:, S1_BLOCKS].reshape(238, -1).sum(1).astype(np.float64)
n_s2  = n_all - n_s1
Hs2 = np.zeros_like(Hs1)
ok = n_s2 > 0
for s in range(Hall.shape[0]):
    Hs2[s][ok] = (Hall[s][ok] * n_all[ok, None] - Hs1[s][ok] * n_s1[ok, None]) / n_s2[ok, None]
Hdelta = Hs2 - Hs1

print("TEST 3 — session-specific extraction (fixed both-sessions cohort, n=175; 111 blocks each):")
test3 = {}
for k in DOSE_TRAITS:
    y = tg[k].values.astype(float); sel = both & np.isfinite(y); yv = y[sel]
    vals = []
    for lbl, R in (("S1only", Hs1), ("S2only", Hs2), ("delta", Hdelta), ("both", Hall)):
        yh = np.mean([loo_yhat(R[s][sel], yv) for s in range(10)], axis=0)
        vals.append(r2(yv, yh))
    test3[k] = vals
    print(f"  {k:<12} S1only {vals[0]:+.4f}   S2only {vals[1]:+.4f}   "
          f"delta {vals[2]:+.4f}   both {vals[3]:+.4f}")

# ── TEST 4: session-concatenated summary [h̄_S1, h̄_S2] (30-dim → PCA-3) ─────────
# Fairness upgrade over the grand average (which collapses sessions): concatenation
# preserves session-specific structure, and per-fold PCA on the concat upweights
# directions CONSISTENT across sessions (cross-session reliability enters the
# representation itself). Same dim-match (PCA→3), same cohort (175): vanilla's
# best-case participant summary.
Hcat = np.concatenate([Hs1, Hs2], axis=2)             # (10, 238, 30)
print("TEST 4 — session-concatenated [h_S1, h_S2] (fixed both-sessions cohort, n=175):")
test4 = {}
for k in DOSE_TRAITS:
    y = tg[k].values.astype(float); sel = both & np.isfinite(y); yv = y[sel]
    yh = np.mean([loo_yhat(Hcat[s][sel], yv) for s in range(10)], axis=0)
    test4[k] = r2(yv, yh)
    print(f"  {k:<12} concat {test4[k]:+.4f}")

# attrition–exploration-axis alignment (why the ablation test is invalid for 3-dim z)
c = np.load("data_sub_t012_full/c_train.npy"); tid = np.load("data_sub_t012_full/task_ids_per_block.npy")
def swr(bl):
    s = p = 0
    for b in range(bl.shape[0]):
        ch = bl[b]; ch = ch[ch >= 0]
        if len(ch) < 2: continue
        s += int((ch[1:] != ch[:-1]).sum()); p += len(ch) - 1
    return s / p if p else np.nan
sw_hor = np.array([swr(c[i][tid == 2]) for i in range(c.shape[0])])
cols = [(Zi[s] - Zi[s].mean(0)) / (Zi[s].std(0) + 1e-9) for s in range(Zi.shape[0])]
sc = PCA(n_components=5).fit_transform(np.hstack(cols))
j = int(np.argmax([abs(pearsonr(sc[:, k_], sw_hor)[0]) for k_ in range(5)]))
v = sc[:, j]; v = v if pearsonr(v, sw_hor)[0] >= 0 else -v
r_axis_s2 = float(pearsonr(v, s2)[0])
print(f"r(IDRNN exploration axis, has_s2) = {r_axis_s2:+.3f}")

np.savez(f"{FULL}/decoding/confound_causal_tests.npz",
         abl_traits=np.array(ABL_TRAITS, dtype=object), abl=np.array(abl),
         dose_traits=np.array(DOSE_TRAITS, dtype=object), dose=np.array(dose),
         test3_traits=np.array(DOSE_TRAITS, dtype=object),
         test3=np.array([test3[k] for k in DOSE_TRAITS]),
         test4=np.array([test4[k] for k in DOSE_TRAITS]),
         r_axis_has_s2=np.array([r_axis_s2]))
print(f"Saved {FULL}/decoding/confound_causal_tests.npz")
