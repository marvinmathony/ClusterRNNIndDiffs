#!/usr/bin/env python3
"""Seed-robust exploration-axis trait analysis for the Thalmann IDRNN (z_dim=3).

Individual latent dimensions are basis-arbitrary across independent trainings, so
"dim 2" in one seed is not "dim 2" in another. We therefore (a) select, in EACH
seed, the dimension that is the exploration axis by a PRECISE, TRAIT-INDEPENDENT
rule, and (b) cross-check that rule against alternatives + a pooled-PCA fixed
dimension. Everything used for selection is behavioural (switch/entropy), never
the questionnaire traits, so the trait read-outs are not circular.

SELECTION RULE (primary, "horizon-switch anchor"):
  The exploration axis is directed exploration, whose behavioural signature is the
  horizon-task switch rate. Per seed: d* = argmax_d |corr(z_d, switch_horizon)|,
  sign-flipped so corr(z_{d*}, switch_horizon) > 0. Seed-symmetric (no reference
  seed), theory-motivated, trait-independent.

Cross-checks:
  - Rule B: argmax cosine( 6-d behavioural fingerprint [per-task switch + entropy],
            canonical seed-999 fingerprint ).
  - Rule C: same but reference = leave-one-out consensus fingerprint (seed-symmetric).
  - PCA  : standardise each seed's 3 dims, concat -> X(236 x 30), PCA. The PC whose
            fingerprint best matches horizon switching is a FIXED, basis-free
            exploration latent; report its variance + traits.

Writes per-seed rows to final_plots/thalmann_z3_3task_full/decoding/z2_seed_matched.csv
and the consensus PCA latent to .../z2_consensus_pca.npy, plus a JSON summary.
"""
import os, json
import numpy as np, pandas as pd, pingouin as pg
from scipy.stats import pearsonr, ttest_1samp
from sklearn.decomposition import PCA

# Env-overridable so the pooled S1+S2 variant (THAL_FULL=final_plots/thalmann_z3_s2_3task_full,
# THAL_DATA=data_sub_t012_full) can run without touching the S1-only outputs.
FULL = os.environ.get("THAL_FULL", "final_plots/thalmann_z3_3task_full")
DATA = os.environ.get("THAL_DATA", "data_thalmann_3task_full")
CAN  = f"{FULL}/canonical/idrnn"
DEC  = f"{FULL}/decoding"
c    = np.load(f"{DATA}/c_train.npy")
tid  = np.load(f"{DATA}/task_ids_per_block.npy")
N    = c.shape[0]
KS   = {0: 2, 1: 4, 2: 2}   # n choices per task (2-armed, restless, horizon)


def switch_rate(bl):
    s = p = 0
    for b in range(bl.shape[0]):
        ch = bl[b]; ch = ch[ch >= 0]
        if len(ch) < 2: continue
        s += int((ch[1:] != ch[:-1]).sum()); p += len(ch) - 1
    return s / p if p else np.nan


def entropy(bl, k):
    ch = bl[bl >= 0].astype(int)
    if len(ch) == 0: return np.nan
    pr = np.bincount(ch, minlength=k) / len(ch); pr = pr[pr > 0]
    return float(-(pr * np.log(pr)).sum())


# behavioural summaries (seed-independent)
SW = {k: np.array([switch_rate(c[i][tid == k]) for i in range(N)]) for k in (0, 1, 2)}
EN = {k: np.array([entropy(c[i][tid == k], KS[k]) for i in range(N)]) for k in (0, 1, 2)}
sw_all = np.array([switch_rate(c[i]) for i in range(N)])
sw_hor = SW[2]


def rr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return pearsonr(a[m], b[m])[0] if m.sum() > 2 else np.nan


def fingerprint(v):
    """6-d trait-independent behavioural fingerprint: per-task switch + entropy."""
    return np.array([rr(v, SW[k]) for k in (0, 1, 2)] + [rr(v, EN[k]) for k in (0, 1, 2)])


def pcorr(d, x, y, cov):
    dd = d[[x, y, cov]].dropna()
    if len(dd) < 5: return np.nan, np.nan
    out = pg.partial_corr(dd, x=x, y=y, covar=cov)
    return float(out["r"].values[0]), float(out["p-val"].values[0])


# ---- load all seeds -------------------------------------------------------
import glob
seed_dirs = sorted(glob.glob(f"{CAN}/runs/seed_*"), key=lambda p: int(p.split("seed_")[1]))
seeds = [int(p.split("seed_")[1]) for p in seed_dirs]
Z = {s: np.load(f"{d}/step1_z_lookup.npy") for s, d in zip(seeds, seed_dirs)}
assert all(z.shape[0] == N for z in Z.values()), "subject mismatch"

tg = pd.read_csv(f"{DEC}/trait_targets.csv")
WM, OP, CEI = tg["WM_composite"].values, tg["BIG5_open"].values, tg["CEI"].values
ALL_KEYS = ["PANAS_PA", "PANAS_NA", "STICSA", "PHQ", "CEI", "BIG5_open", "AxDep", "posMood",
            "negMood", "Exp", "WM_composite", "WM_OS", "WM_SS", "WM_WMU"]
TRAIT = {k: tg[k].values for k in ALL_KEYS}

# canonical fingerprint (seed 999, dim 2) for Rule B
canon_fp = fingerprint(Z[999][:, 2]); canon_fp /= np.linalg.norm(canon_fp)


def select_horizon(z):
    """Rule A: argmax|corr(dim, horizon switch)|, sign so corr>0."""
    cors = np.array([rr(z[:, d], sw_hor) for d in range(z.shape[1])])
    d = int(np.argmax(np.abs(cors))); sgn = 1.0 if cors[d] >= 0 else -1.0
    return d, sgn


def select_cos(z, ref):
    """Cosine match of fingerprint to a reference (Rule B / C)."""
    best = None
    for d in range(z.shape[1]):
        fp = fingerprint(z[:, d]); fp = fp / (np.linalg.norm(fp) + 1e-9)
        for sgn in (1.0, -1.0):
            cos = float(np.dot(sgn * fp, ref))
            if best is None or cos > best[0]: best = (cos, d, sgn)
    return best[1], best[2], best[0]


# leave-one-out consensus fingerprint (Rule C, seed-symmetric)
per_seed_expfp = {}
for s in seeds:
    d, sgn = select_horizon(Z[s])
    fp = fingerprint(sgn * Z[s][:, d]); per_seed_expfp[s] = fp / np.linalg.norm(fp)

rows = []
at_rows = []          # per-seed correlation of the matched exploration dim with EVERY trait
for s in seeds:
    z = Z[s]
    dA, sgnA = select_horizon(z)
    loo = np.mean([per_seed_expfp[o] for o in seeds if o != s], axis=0)
    loo /= np.linalg.norm(loo)
    dC, sgnC, cosC = select_cos(z, loo)
    dB, sgnB, cosB = select_cos(z, canon_fp)
    ze = sgnA * z[:, dA]                       # PRIMARY exploration latent for this seed
    # WITHIN-SEED PCA: rotate this seed's 3 dims to its own principal axes, then pick
    # the exploration component by the same trait-independent horizon-switch anchor.
    zs = (z - z.mean(0)) / (z.std(0) + 1e-9)
    pca_s = PCA(n_components=z.shape[1]).fit(zs)
    sc = pca_s.transform(zs)
    jx = int(np.argmax([abs(rr(sc[:, j], sw_hor)) for j in range(sc.shape[1])]))
    zp = sc[:, jx]
    if rr(zp, sw_hor) < 0: zp = -zp
    ws_var = float(pca_s.explained_variance_ratio_[jx])
    df = pd.DataFrame({"ze": ze, "WM": WM, "open": OP, "CEI": CEI,
                       "sw": sw_all})
    p_open_wm, _ = pcorr(df, "ze", "open", "WM")
    p_cei_wm, _  = pcorr(df, "ze", "CEI", "WM")
    p_open_sw, _ = pcorr(df, "ze", "open", "sw")     # latent | switch
    p_sw_open, _ = pcorr(df.rename(columns={"sw": "SW"}), "SW", "open", "ze")  # switch | latent
    # per-task partial r(ze, switch | entropy)
    pr_task = {}
    for k in (0, 1, 2):
        dk = pd.DataFrame({"ze": ze, "switch": SW[k], "entropy": EN[k]})
        r, _ = pcorr(dk, "ze", "switch", "entropy"); pr_task[k] = r
    rows.append(dict(seed=s, dimA=dA, dimB=dB, dimC=dC, agreeAB=int(dA == dB),
                     agreeAC=int(dA == dC), cosB=cosB, cosC=cosC,
                     r_wm=rr(ze, WM), r_open=rr(ze, OP), r_cei=rr(ze, CEI),
                     r_sw=rr(ze, sw_all), r_sw2a=rr(ze, SW[0]), r_swre=rr(ze, SW[1]),
                     r_swho=rr(ze, sw_hor),
                     pr_open_wm=p_open_wm, pr_cei_wm=p_cei_wm,
                     pr_open_sw=p_open_sw, pr_sw_open=p_sw_open,
                     pr_sw2a=pr_task[0], pr_swre=pr_task[1], pr_swho=pr_task[2],
                     ws_pc=jx, ws_var=ws_var,
                     ws_r_wm=rr(zp, WM), ws_r_open=rr(zp, OP), ws_r_cei=rr(zp, CEI),
                     ws_r_swho=rr(zp, sw_hor)))
    # per-seed correlation of THIS seed's matched exploration dim (ze) with every trait
    at_rows.append({"seed": s, **{k: rr(ze, TRAIT[k]) for k in ALL_KEYS}})

R = pd.DataFrame(rows)
pd.DataFrame(at_rows).to_csv(f"{DEC}/z2_seed_matched_alltraits.csv", index=False)
R.to_csv(f"{DEC}/z2_seed_matched.csv", index=False)

# ---- pooled PCA: fixed, basis-free exploration latent ---------------------
cols = []
for s in seeds:
    z = Z[s]
    zs = (z - z.mean(0)) / (z.std(0) + 1e-9)   # standardise each seed's dims
    cols.append(zs)
X = np.hstack(cols)                            # 236 x 30
pca = PCA(n_components=min(10, X.shape[1])).fit(X)
scores = pca.transform(X)                      # 236 x k
# Consensus axis = PC1 BY DEFINITION (2026-07-12 convention): fixed ex ante by the
# latents' own variance structure — no component selection, less exploratory freedom.
# (In every run to date the switch-anchored argmax was PC1 anyway; the anchor is now
# used only to fix the arbitrary PCA sign below.)
jexp = 0
pcv = scores[:, jexp]
if rr(pcv, sw_hor) < 0: pcv = -pcv
np.save(f"{DEC}/z2_consensus_pca.npy", pcv)
pca_summary = dict(
    exp_pc=jexp + 1, var_explained=[round(float(v), 3) for v in pca.explained_variance_ratio_[:5]],
    fingerprint=[round(float(x), 3) for x in fingerprint(pcv)],
    r_wm=round(rr(pcv, WM), 3), r_open=round(rr(pcv, OP), 3), r_cei=round(rr(pcv, CEI), 3),
    r_sw_horizon=round(rr(pcv, sw_hor), 3),
    top_pcs={f"PC{j+1}": dict(r_wm=round(rr(scores[:, j], WM), 3),
                              r_open=round(rr(scores[:, j], OP), 3),
                              r_cei=round(rr(scores[:, j], CEI), 3),
                              r_swho=round(rr(scores[:, j], sw_hor), 3)) for j in range(3)})

# ---- pooled-PCA VALIDATION: not overfitting (LOO out-of-sample) + de-attenuation ----
# sign-aligned, standardised per-seed exploration latent (raw-dim horizon anchor)
EXP = []
for s in seeds:
    d, sgn = select_horizon(Z[s]); v = sgn * Z[s][:, d]; EXP.append((v - v.mean()) / (v.std() + 1e-9))
EXP = np.column_stack(EXP)                                  # 236 x 10
mean_lat = EXP.mean(1)                                      # simple consensus (zero cross-subject fitting)
np.save(f"{DEC}/z2_seedavg_mean.npy", mean_lat)             # PCA-FREE seed-averaged latent (for scatter illustration)
pw = [rr(EXP[:, i], EXP[:, j]) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
rbar = float(np.mean(pw)); k = len(seeds); sb = k * rbar / (1 + (k - 1) * rbar)
r_single = float(np.mean([rr(EXP[:, i], OP) for i in range(len(seeds))]))
# leave-one-SUBJECT-out 30-col pooled PCA: refit on 235, project held-out subject (no leakage)
loo = np.full(N, np.nan)
for i in range(N):
    idx = np.arange(N) != i
    pp = PCA(n_components=5).fit(X[idx]); st = pp.transform(X[idx])
    j = int(np.argmax([abs(rr(st[:, m], sw_hor[idx])) for m in range(st.shape[1])]))
    sgn = 1.0 if rr(st[:, j], sw_hor[idx]) >= 0 else -1.0
    loo[i] = sgn * pp.transform(X[i:i + 1])[0, j]
pca_validation = dict(
    simple_mean=dict(r_wm=round(rr(mean_lat, WM), 3), r_open=round(rr(mean_lat, OP), 3), r_cei=round(rr(mean_lat, CEI), 3)),
    pooled_pca_loo=dict(r_wm=round(rr(loo, WM), 3), r_open=round(rr(loo, OP), 3), r_cei=round(rr(loo, CEI), 3)),
    cross_seed_reliability=round(rbar, 3), spearman_brown_10=round(sb, 3),
    single_seed_open=round(r_single, 3), deattenuated_open=round(r_single / np.sqrt(rbar), 3))
pca_summary["validation"] = pca_validation

# ---- report ---------------------------------------------------------------
print("=== per-seed selection (Rule A = horizon-switch anchor) ===")
print(R[["seed", "dimA", "dimB", "dimC", "agreeAB", "agreeAC", "r_wm", "r_open", "r_cei"]]
      .to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
print(f"\nrule agreement: A==B {R.agreeAB.sum()}/10   A==C {R.agreeAC.sum()}/10")

print("\n=== seed-averaged (matched dim, Rule A) mean +/- SE, t-test vs 0 ===")
for k, lab in [("r_wm", "r(WM)"), ("r_open", "r(openness)"), ("r_cei", "r(curiosity)"),
               ("pr_open_wm", "r(open|WM)"), ("pr_cei_wm", "r(CEI|WM)"),
               ("pr_open_sw", "r(open|switch)"), ("pr_sw_open", "r(switch|open)"),
               ("pr_sw2a", "pr(sw|ent) 2a"), ("pr_swre", "pr(sw|ent) re"),
               ("pr_swho", "pr(sw|ent) ho"), ("r_sw", "r(switch_all)")]:
    v = R[k].dropna().values; t, p = ttest_1samp(v, 0.0)
    print(f"  {lab:16s} mean={v.mean():+.3f} SE={v.std(ddof=1)/np.sqrt(len(v)):.3f}  t={t:+6.2f} p={p:.1e}")

print("\n=== behavioural-summary openness baselines (seed-independent) ===")
print(f"  r(open, overall switch) = {rr(OP, sw_all):+.3f}")
print(f"  r(open, horizon switch) = {rr(OP, sw_hor):+.3f}")

# ---- within-seed PCA: per-seed exploration component, then combine --------
ws_summary = {}
print("\n=== WITHIN-SEED PCA (per-seed component, analysed individually, then combined) ===")
print(R[["seed", "ws_pc", "ws_var", "ws_r_wm", "ws_r_open", "ws_r_cei"]]
      .to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
for k, lab in [("ws_r_wm", "r(WM)"), ("ws_r_open", "r(openness)"), ("ws_r_cei", "r(curiosity)")]:
    v = R[k].dropna().values; t, p = ttest_1samp(v, 0.0)
    ws_summary[k] = [round(float(v.mean()), 4), round(float(v.std(ddof=1)/np.sqrt(len(v))), 4)]
    print(f"  {lab:14s} mean={v.mean():+.3f} SE={v.std(ddof=1)/np.sqrt(len(v)):.3f}  t={t:+6.2f} p={p:.1e}")

print("\n=== pooled-PCA fixed exploration latent ===")
print(json.dumps(pca_summary, indent=2))

print("\n=== THREE-METHOD CONVERGENCE (trait r with exploration axis) ===")
print(f"{'method':32s} {'WM':>8} {'openness':>9} {'curiosity':>10}")
print(f"{'raw-dim structure-matched':32s} {R.r_wm.mean():>+8.3f} {R.r_open.mean():>+9.3f} {R.r_cei.mean():>+10.3f}")
print(f"{'within-seed PCA (combined)':32s} {R.ws_r_wm.mean():>+8.3f} {R.ws_r_open.mean():>+9.3f} {R.ws_r_cei.mean():>+10.3f}")
print(f"{'pooled-across-seed PCA (fixed)':32s} {pca_summary['r_wm']:>+8.3f} {pca_summary['r_open']:>+9.3f} {pca_summary['r_cei']:>+10.3f}")

json.dump({"selection_rule": "horizon-switch anchor (argmax|corr(dim,switch_horizon)|, sign>0)",
           "seed_avg": {k: [round(float(R[k].dropna().mean()), 4),
                            round(float(R[k].dropna().std(ddof=1)/np.sqrt(R[k].dropna().size)), 4)]
                        for k in ["r_wm", "r_open", "r_cei", "pr_open_wm", "pr_cei_wm",
                                  "pr_open_sw", "pr_sw_open", "pr_sw2a", "pr_swre", "pr_swho", "r_sw"]},
           "within_seed_pca": ws_summary,
           "open_summary_baselines": {"overall_switch": round(rr(OP, sw_all), 4),
                                      "horizon_switch": round(rr(OP, sw_hor), 4)},
           "pca": pca_summary},
          open(f"{DEC}/z2_seed_matched_summary.json", "w"), indent=2)
print(f"\nwrote {DEC}/z2_seed_matched.csv, z2_consensus_pca.npy, z2_seed_matched_summary.json")
