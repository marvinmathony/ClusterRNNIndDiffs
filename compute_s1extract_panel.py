#!/usr/bin/env python3
"""Panel-c data for the S1-EXTRACTION convention + the coverage-confound argument figure.

Convention: models are FULLY INFORMED (pooled S1+S2 canonical ds_t012 runs), but
per-participant representations are extracted from SESSION-1 BLOCKS ONLY for vanilla
(trial-averaged h; cached by analyze_s1extract_decoding.py), so every participant's
summary covers the identical block set and the coverage axis cannot exist. IDRNN's
representation is the trained lookup z (consolidated during training). Cohort =
has_s1 participants (n=236); per-trait n further reduced by finite targets.

Outputs (final_plots/thalmann_z3_s2_3task_full/decoding/):
  loo_r2_panel_s1extract.csv  — drop-in for cell 88 panel c: per target,
      idrnn_r2 / vanilla_r2 (canonical, seed-averaged ŷ), idrnn_sd / vanilla_sd
      (train-resample bootstrap SD, ensemble order, B=500), idrnn_bf / vanilla_bf
      (one-sided JZS), idrnn_perm_p / vanilla_perm_p.
  confound_panel.npz — everything the standalone confound-argument notebook cell
      plots: PC-variance + r(PC, has_s2) spectra, has_s2 decodability per
      representation, r(has_s2, trait) with BFs, and vanilla/IDRNN trait R² under
      the three extraction/cohort regimes (mixed-full / both-sessions-full / S1-extract).
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
from joblib import Parallel, delayed
from pingouin import bayesfactor_pearson

FULL = "final_plots/thalmann_z3_s2_3task_full"
ALPHAS = (0.001, 0.01, 0.1, 1, 10, 100, 1000); EPS = 1e-8; ZD = 3
B = int(os.environ.get("B_BOOT", "500"))
N_JOBS = int(os.environ.get("N_JOBS", os.environ.get("SLURM_CPUS_PER_TASK", "4")))
ALL_KEYS = ["PANAS_PA", "PANAS_NA", "STICSA", "PHQ", "CEI", "BIG5_open", "AxDep",
            "posMood", "negMood", "Exp", "WM_composite", "WM_OS", "WM_SS", "WM_WMU"]


def loo_yhat(Z, y, T=None):
    """Canonical LOO ŷ (T=None) or bootstrap-train LOO ŷ (T = (n, n-1) resample)."""
    n = len(y); yhat = np.empty(n); idx = np.arange(n)
    for i in range(n):
        tr = idx[idx != i] if T is None else T[i]
        Ztr, ytr, zte = Z[tr], y[tr], Z[i]
        if Ztr.shape[1] > ZD:
            p = PCA(n_components=ZD).fit(Ztr); Ztr = p.transform(Ztr); zte = p.transform(zte[None])[0]
        mz, sz = Ztr.mean(0), Ztr.std(0) + EPS; my, sy = ytr.mean(), ytr.std() + EPS
        yhat[i] = RidgeCV(alphas=list(ALPHAS)).fit((Ztr - mz) / sz, (ytr - my) / sy
                                                   ).predict(((zte - mz) / sz)[None])[0] * sy + my
    return yhat


def _r2(y, yh):
    return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)


def draw_T(n, rng):
    raw = rng.integers(0, n - 1, size=(n, n - 1))
    ii = np.arange(n)[:, None]
    return raw + (raw >= ii)


def canonical_unit(Zs, y):
    """Seed-averaged canonical ŷ → R², one-sided BF, perm p (identical to the
    main panel's compute_loo_r2_canonical_bf machinery)."""
    yhat = np.mean([loo_yhat(Z, y) for Z in Zs], axis=0)
    r2 = float(_r2(y, yhat)); r = float(np.corrcoef(yhat, y)[0, 1])
    bf = float(bayesfactor_pearson(r, len(y), alternative="greater"))
    rng = np.random.default_rng(0)
    perm = np.array([np.corrcoef(rng.permutation(y), yhat)[0, 1] for _ in range(10000)])
    perm_p = float((np.sum(perm >= r) + 1) / (len(perm) + 1))
    return dict(r2=r2, bf=bf, perm_p=perm_p)


def boot_unit(Zs, y, seed):
    """Ensemble-order train-resample bootstrap SD (B replicates, T shared across seeds)."""
    rng = np.random.default_rng(seed)
    n = len(y); ens = np.empty(B)
    for b in range(B):
        T = draw_T(n, rng)
        YH = np.stack([loo_yhat(Z, y, T) for Z in Zs])
        ens[b] = _r2(y, YH.mean(0))
    return float(np.nanstd(ens))


def main():
    print(f"B={B} N_JOBS={N_JOBS}")
    Zi   = np.load("final_plots/thalmann_z3_ds_t012/decoding/_bootreps_idrnn.npy")
    Hall = np.load("final_plots/thalmann_z3_ds_t012/decoding/_bootreps_vanilla_rawh.npy")
    Hs1  = np.load(f"{FULL}/decoding/_bootreps_vanilla_rawh_s1extract.npy")
    tg   = pd.read_csv(f"{FULL}/decoding/trait_targets.csv")
    meta = pd.read_csv("data_thalmann_s2/df_all.csv").set_index("subid").loc[tg["subid"]]
    has_s1 = (meta["has_s1"] == 1).values
    s2 = meta["has_s2"].values.astype(float)
    both = has_s1 & (s2 == 1)
    REPS = {"idrnn": Zi, "vanilla": Hs1}          # the panel-c convention
    print(f"cohort has_s1 n={has_s1.sum()}, both n={both.sum()}")

    # ── panel-c table: canonical + bootstrap SD, S1-extraction convention ────
    units = [(k, a) for k in ALL_KEYS for a in ("idrnn", "vanilla")]

    def unit_sel(k):
        return has_s1 & np.isfinite(tg[k].values.astype(float))

    canon = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(canonical_unit)([Z[unit_sel(k)] for Z in REPS[a]],
                                tg[k].values.astype(float)[unit_sel(k)])
        for (k, a) in units)
    seeds = np.random.SeedSequence(0).spawn(len(units))
    sds = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(boot_unit)([Z[unit_sel(k)] for Z in REPS[a]],
                           tg[k].values.astype(float)[unit_sel(k)], sq)
        for (k, a), sq in zip(units, seeds))

    res = {u: (c, s) for u, c, s in zip(units, canon, sds)}
    rows = []
    for k in ALL_KEYS:
        ci, si = res[(k, "idrnn")]; cv, sv = res[(k, "vanilla")]
        rows.append(dict(target=k, n=int(unit_sel(k).sum()),
                         idrnn_r2=ci["r2"], vanilla_r2=cv["r2"],
                         idrnn_sd=si, vanilla_sd=sv,
                         idrnn_bf=ci["bf"], vanilla_bf=cv["bf"],
                         idrnn_perm_p=ci["perm_p"], vanilla_perm_p=cv["perm_p"]))
        print(f"{k:<14} idrnn {ci['r2']:+.4f}±{si:.4f} (BF {ci['bf']:.3g})   "
              f"vanilla_s1x {cv['r2']:+.4f}±{sv:.4f} (BF {cv['bf']:.3g})")
    out = f"{FULL}/decoding/loo_r2_panel_s1extract.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")

    # ── confound-argument figure data ─────────────────────────────────────────
    npz = {}
    # (1) PC spectra: variance ratio + |r(PC, has_s2)| for vanilla-full and idrnn (mixed cohort)
    for tag, R in (("vanfull", Hall), ("idrnn", Zi)):
        vr, rc = [], []
        for s in range(R.shape[0]):
            X = R[s] - R[s].mean(0)
            p = PCA(n_components=min(5, X.shape[1])).fit(X)
            sc = p.transform(X)
            vr.append(p.explained_variance_ratio_)
            rc.append([abs(pearsonr(sc[:, j], s2)[0]) for j in range(sc.shape[1])])
        npz[f"pc_var_{tag}"] = np.array(vr); npz[f"pc_rs2_{tag}"] = np.array(rc)
    # (2) has_s2 decodability per representation (LOO, dim-matched readout)
    for tag, R, sel in (("vanfull", Hall, np.ones(len(s2), bool)),
                        ("idrnn", Zi, np.ones(len(s2), bool)),
                        ("vans1x", Hs1, has_s1)):
        yb = s2[sel]
        yh = np.mean([loo_yhat(R[s][sel], yb) for s in range(R.shape[0])], axis=0)
        npz[f"cov_decode_{tag}"] = np.array([pearsonr(yb, yh)[0]])
    # (3) retention–trait correlations (+ two-sided BF)
    CT = ["PANAS_PA", "PANAS_NA", "CEI", "BIG5_open"]
    rt, bt_ = [], []
    for k in CT:
        y = tg[k].values.astype(float); m = np.isfinite(y)
        r = float(pearsonr(y[m], s2[m])[0]); rt.append(r)
        bt_.append(float(bayesfactor_pearson(r, int(m.sum()))))
    npz["ret_traits"] = np.array(CT, dtype=object); npz["ret_r"] = np.array(rt); npz["ret_bf"] = np.array(bt_)
    # (4) readout consequences: trait R² per regime for both archs
    RT = ["PANAS_PA", "PANAS_NA", "CEI", "BIG5_open", "WM_composite"]
    regimes = {"mixed_full": (Hall, Zi, np.ones(len(s2), bool)),
               "both_full":  (Hall, Zi, both),
               "s1_extract": (Hs1,  Zi, has_s1)}
    for rg, (Hv, Zr, sel0) in regimes.items():
        rv, ri = [], []
        for k in RT:
            y = tg[k].values.astype(float); sel = sel0 & np.isfinite(y); yv = y[sel]
            rv.append(_r2(yv, np.mean([loo_yhat(Hv[s][sel], yv) for s in range(Hv.shape[0])], axis=0)))
            ri.append(_r2(yv, np.mean([loo_yhat(Zr[s][sel], yv) for s in range(Zr.shape[0])], axis=0)))
        npz[f"regime_{rg}_vanilla"] = np.array(rv); npz[f"regime_{rg}_idrnn"] = np.array(ri)
    npz["regime_traits"] = np.array(RT, dtype=object)
    np.savez(f"{FULL}/decoding/confound_panel.npz", **npz)
    print(f"Saved {FULL}/decoding/confound_panel.npz")


if __name__ == "__main__":
    main()
