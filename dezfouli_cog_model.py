"""
dezfouli_cog_model.py — Cognitive-model baselines for the Dezfouli 2-armed
bandit dataset (3 diagnostic groups × 12 reward-contingency blocks; binary
reward, binary choice; variable block lengths).

Fits four baselines per outer-CV fold:
  1. CogModel_EM  — proper cognitive model (QLP), hierarchical-Bayesian EM
  2. CogModel_CP  — proper cognitive model (QLP), common-process (one θ
                    shared across train subjects)
  3. IllSpec_EM   — ill-specified Q-learning (QL), hierarchical-Bayesian EM
  4. IllSpec_CP   — ill-specified Q-learning (QL), common process

Specified model — QLP (3 params per subject):
  Q_{t+1}(a_t)  = (1 − φ)·Q_t(a_t) + φ·r_t
  k_t(a)        = κ if a == a_{t-1} else 0   (k_t(a)=0 on first trial)
  π_t(a)        ∝ exp(β·Q_t(a) + k_t(a))
  Per-block reset.   Free parameters: φ ∈ (0,1), β > 0, κ ∈ ℝ.

Ill-specified model — QL (2 params): drop κ. Same Q update; π ∝ exp(β·Q).

CSV filenames match what analyze_outer_cv.py already reads
(cog_model_{,cp_}results.csv, ill_specified_{map,cp}_results.csv).

Usage:
    python dezfouli_cog_model.py --fold 0
"""

import os
import argparse
from math import exp, log
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR_BASE = "data_dezfouli"
RAW_CSV       = "data/for_plos.csv"
N_BLOCKS      = 12
A             = 2
Q_INIT        = 0.5     # binary rewards in [0,1]; 0.5 is a neutral midpoint


# ══════════════════════════════════════════════════════════════════════════════
# QLP — specified model (Q-learning + perseveration)
# ══════════════════════════════════════════════════════════════════════════════
N_PARAMS_QLP = 3


def unpack_qlp(pv):
    pv = np.asarray(pv, dtype=float)
    return dict(
        phi   = 1.0 / (1.0 + np.exp(-np.clip(pv[0], -15, 15))),
        beta  = np.exp(np.clip(pv[1], -10, 10)),
        kappa = float(pv[2]),
    )


def ll_qlp_subject(pv, c_p, r_p, valid_p):
    """Log-likelihood for one subject under QLP.

    c_p, r_p  : (N_BLOCKS, T_max) int / float, with -100 padding outside the
                valid trial range.
    valid_p   : (N_BLOCKS,) int — valid-trial count per block.
    """
    p = unpack_qlp(pv)
    phi, beta, kappa = p["phi"], p["beta"], p["kappa"]

    ll = 0.0
    for b in range(N_BLOCKS):
        L = int(valid_p[b])
        if L == 0:
            continue
        Q = np.full(A, Q_INIT, dtype=np.float64)
        prev = -1
        for t in range(L):
            u = beta * Q.copy()
            if prev >= 0:
                u[prev] += kappa
            um   = u.max()
            logZ = um + log(exp(u[0] - um) + exp(u[1] - um))
            c    = int(c_p[b, t])
            ll  += u[c] - logZ
            r    = float(r_p[b, t])
            Q[c] = Q[c] + phi * (r - Q[c])
            prev = c
    return ll


# ══════════════════════════════════════════════════════════════════════════════
# QL — ill-specified model (no perseveration)
# ══════════════════════════════════════════════════════════════════════════════
N_PARAMS_QL = 2


def unpack_ql(pv):
    pv = np.asarray(pv, dtype=float)
    return dict(
        phi  = 1.0 / (1.0 + np.exp(-np.clip(pv[0], -15, 15))),
        beta = np.exp(np.clip(pv[1], -10, 10)),
    )


def ll_ql_subject(pv, c_p, r_p, valid_p):
    p = unpack_ql(pv)
    phi, beta = p["phi"], p["beta"]

    ll = 0.0
    for b in range(N_BLOCKS):
        L = int(valid_p[b])
        if L == 0:
            continue
        Q = np.full(A, Q_INIT, dtype=np.float64)
        for t in range(L):
            u    = beta * Q
            um   = u.max()
            logZ = um + log(exp(u[0] - um) + exp(u[1] - um))
            c    = int(c_p[b, t])
            ll  += u[c] - logZ
            r    = float(r_p[b, t])
            Q[c] = Q[c] + phi * (r - Q[c])
    return ll


# ══════════════════════════════════════════════════════════════════════════════
# Specs
# ══════════════════════════════════════════════════════════════════════════════
QLP_SPEC = dict(
    name        = "qlp",
    n_params    = N_PARAMS_QLP,
    ll_subject  = ll_qlp_subject,
    init_m      = np.array([0.0, 0.0, 0.0]),                  # φ≈0.5, β≈1, κ=0
    init_v      = np.array([4.0, 4.0, 4.0]),
    param_names = ["logit_φ", "log_β", "κ"],
)

QL_SPEC = dict(
    name        = "ql",
    n_params    = N_PARAMS_QL,
    ll_subject  = ll_ql_subject,
    init_m      = np.array([0.0, 0.0]),                       # φ≈0.5, β≈1
    init_v      = np.array([4.0, 4.0]),
    param_names = ["logit_φ", "log_β"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════
def build_subid_to_rawid(df_raw):
    """Replicate the integer-coding done in load_dezfouli.py (cat.codes)."""
    df_raw = df_raw.copy()
    df_raw["subid"] = df_raw["ID"].astype("category").cat.codes
    pairs = df_raw[["subid", "ID"]].drop_duplicates().sort_values("subid")
    return dict(zip(pairs["subid"].astype(int), pairs["ID"]))


def load_subject_data(subids, df_raw, subid_to_rawid):
    """Returns:
      c_arr  (P, N_BLOCKS, T_max)  int   choices  (-100 = padding)
      r_arr  (P, N_BLOCKS, T_max)  float rewards  (-100 = padding)
      valid  (P, N_BLOCKS)         int   valid-trial count per block
    """
    P = len(subids)
    block_lens = (
        df_raw.groupby(["ID", "block"]).size().reset_index(name="L")
    )
    T_max = int(block_lens["L"].max())

    c_arr = np.full((P, N_BLOCKS, T_max), -100, dtype=np.int64)
    r_arr = np.full((P, N_BLOCKS, T_max), -100.0, dtype=np.float64)
    valid = np.zeros((P, N_BLOCKS), dtype=np.int64)

    for p_idx, sid in enumerate(subids):
        raw_id = subid_to_rawid[int(sid)]
        df_s = df_raw[df_raw["ID"] == raw_id]
        for b_raw, grp in df_s.groupby("block"):
            b = int(b_raw) - 1
            grp = grp.reset_index(drop=True)
            L = len(grp)
            valid[p_idx, b] = L
            if L == 0:
                continue
            choices = grp["key"].map({"R1": 0, "R2": 1}).astype(int).values
            rewards = grp["reward"].astype(float).values
            c_arr[p_idx, b, :L] = choices
            r_arr[p_idx, b, :L] = rewards
    return c_arr, r_arr, valid


# ══════════════════════════════════════════════════════════════════════════════
# Generic EM + CP machinery
# ══════════════════════════════════════════════════════════════════════════════
def neg_log_posterior(pv, c_p, r_p, valid_p, m, v, ll_subject):
    ll = ll_subject(pv, c_p, r_p, valid_p)
    log_prior = float(np.sum(norm.logpdf(pv, loc=m, scale=np.sqrt(v))))
    return -(ll + log_prior)


def hessian_diag(fun, x, eps=1e-4):
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    f0 = fun(x)
    diag = np.zeros(n)
    for j in range(n):
        xp = x.copy(); xp[j] += eps
        xm = x.copy(); xm[j] -= eps
        diag[j] = (fun(xp) - 2 * f0 + fun(xm)) / (eps ** 2)
    return diag


def estimate_individual(spec, m, v, c_p, r_p, valid_p,
                         n_restarts=1, maxiter=200):
    ll_subject = spec["ll_subject"]; n_p = spec["n_params"]
    obj = lambda pv: neg_log_posterior(pv, c_p, r_p, valid_p, m, v, ll_subject)

    res0 = minimize(obj, x0=m.copy(), method="L-BFGS-B",
                    options=dict(maxiter=maxiter, ftol=1e-6))
    best_val, best_x = res0.fun, res0.x.copy()
    for _ in range(n_restarts - 1):
        x0 = m + np.random.randn(n_p) * np.sqrt(v) * 0.5
        res = minimize(obj, x0=x0, method="L-BFGS-B",
                       options=dict(maxiter=maxiter, ftol=1e-6))
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x.copy()
    H_diag = hessian_diag(obj, best_x)
    var_i  = np.where(H_diag > 1e-6, 1.0 / H_diag, v)
    var_i  = np.clip(var_i, 1e-4, 10.0)
    return best_x, var_i


def update_group_prior(h_all, var_all):
    m = h_all.mean(0)
    v = (h_all ** 2 + var_all).mean(0) - m ** 2
    v = np.clip(v, 1e-4, 10.0)
    return m, v


def run_em(spec, c_arr, r_arr, valid, n_iter=20, seed=0):
    m = spec["init_m"].copy()
    v = spec["init_v"].copy()
    P = c_arr.shape[0]
    n_p = spec["n_params"]
    h_all   = np.zeros((P, n_p))
    var_all = np.zeros((P, n_p))
    m_hist, v_hist = [m.copy()], [v.copy()]

    np.random.seed(seed)
    for it in range(n_iter):
        for p in range(P):
            h_all[p], var_all[p] = estimate_individual(
                spec, m, v, c_arr[p], r_arr[p], valid[p])
        m, v = update_group_prior(h_all, var_all)
        m_hist.append(m.copy()); v_hist.append(v.copy())
        if it % 5 == 0 or it == n_iter - 1:
            print(f"  [{spec['name']}] EM iter {it+1:3d}/{n_iter}  "
                  f"m={np.array2string(m, precision=3, suppress_small=True)}")
    return m, v, h_all, var_all, np.array(m_hist), np.array(v_hist)


def marginal_loglik_mc(spec, m, v, c_p, r_p, valid_p,
                        n_samples=500, seed=0):
    rng = np.random.default_rng(seed)
    std = np.sqrt(v)
    samples = rng.normal(loc=m, scale=std, size=(n_samples, len(m)))
    ll_subject = spec["ll_subject"]
    lls = np.array([ll_subject(samples[s], c_p, r_p, valid_p)
                    for s in range(n_samples)])
    return float(np.logaddexp.reduce(lls) - np.log(n_samples))


def fit_cp(spec, c_arr, r_arr, valid, n_restarts=3, maxiter=500, seed=0):
    """Common-process fit: single θ shared across all training subjects."""
    ll_subject = spec["ll_subject"]
    n_p = spec["n_params"]
    P   = c_arr.shape[0]

    def neg_joint(pv):
        return -sum(ll_subject(pv, c_arr[p], r_arr[p], valid[p])
                    for p in range(P))

    np.random.seed(seed)
    best_val, best_x = np.inf, spec["init_m"].copy()
    for r in range(n_restarts):
        x0 = (spec["init_m"] if r == 0
              else spec["init_m"] + np.random.randn(n_p) * np.sqrt(spec["init_v"]) * 0.5)
        res = minimize(neg_joint, x0=x0, method="L-BFGS-B",
                       options=dict(maxiter=maxiter, ftol=1e-6))
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x.copy()
    print(f"  [{spec['name']}] CP fit: joint_nll={best_val:.1f}  "
          f"θ={np.array2string(best_x, precision=3, suppress_small=True)}")
    return best_x


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════
def eval_em_on_test(spec, m, v, c_arr, r_arr, valid, test_ids,
                     n_mc_samples=500, seed=0):
    """MAP + Laplace per test subj → marginal NLL via MC."""
    P = len(test_ids)
    n_p = spec["n_params"]
    h_te   = np.zeros((P, n_p))
    var_te = np.zeros((P, n_p))
    ll_map = np.zeros(P)
    ll_mrg = np.zeros(P)

    for p in range(P):
        h_te[p], var_te[p] = estimate_individual(
            spec, m, v, c_arr[p], r_arr[p], valid[p],
            n_restarts=3, maxiter=500)
        ll_map[p] = spec["ll_subject"](h_te[p], c_arr[p], r_arr[p], valid[p])
        ll_mrg[p] = marginal_loglik_mc(
            spec, m, v, c_arr[p], r_arr[p], valid[p],
            n_samples=n_mc_samples, seed=seed + int(test_ids[p]))
    return h_te, var_te, ll_map, ll_mrg


def eval_cp_on_test(spec, theta_cp, c_arr, r_arr, valid, test_ids):
    P = len(test_ids)
    ll = np.zeros(P)
    for p in range(P):
        ll[p] = spec["ll_subject"](theta_cp, c_arr[p], r_arr[p], valid[p])
    return ll


def save_results_csv(out_csv, test_ids, ll_total, valid_te, model_name):
    """Save per-subject NLL.  normalized_likelihood = mean NLL per valid trial."""
    n_valid_per = valid_te.sum(axis=1).clip(min=1)
    pd.DataFrame({
        "subid": np.asarray(test_ids, dtype=int),
        "normalized_likelihood": -ll_total / n_valid_per,
        "total_nll":             -ll_total,
        "n_valid":               n_valid_per,
        "model":                 model_name,
    }).to_csv(out_csv, index=False)
    print(f"  saved → {out_csv}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--n_em_iter", type=int, default=20)
    ap.add_argument("--n_mc_samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["qlp", "ql", "all"], default="all",
                    help="Limit to one model family")
    args = ap.parse_args()

    FOLD_DIR = f"{DATA_DIR_BASE}/fold{args.fold}"
    print(f"=== Dezfouli cog-model baselines — fold {args.fold} ===")

    train_ids = pd.read_csv(f"{FOLD_DIR}/df_train.csv")["subid"].astype(int).values
    test_ids  = pd.read_csv(f"{FOLD_DIR}/df_test.csv")["subid"].astype(int).values
    print(f"  train subjects: {len(train_ids)}, test subjects: {len(test_ids)}")

    df_raw = pd.read_csv(RAW_CSV)
    sid_to_raw = build_subid_to_rawid(df_raw)
    c_tr, r_tr, valid_tr = load_subject_data(train_ids, df_raw, sid_to_raw)
    c_te, r_te, valid_te = load_subject_data(test_ids,  df_raw, sid_to_raw)

    specs = ([QLP_SPEC, QL_SPEC] if args.only == "all"
             else [QLP_SPEC] if args.only == "qlp" else [QL_SPEC])

    for spec in specs:
        name = spec["name"]
        print(f"\n{'='*72}\nModel: {name}\n{'='*72}")

        # ── EM (individual-differences hierarchical fit) ──────────────────────
        print(f"\n-- [{name}] EM on training fold --")
        m, v, h_tr, var_tr, m_hist, v_hist = run_em(
            spec, c_tr, r_tr, valid_tr,
            n_iter=args.n_em_iter, seed=args.seed)
        print(f"\n-- [{name}] Final group prior (unconstrained) --")
        for i, pname in enumerate(spec["param_names"]):
            print(f"  {pname:10s}  m={m[i]:+.4f}  v={v[i]:.4f}")

        print(f"\n-- [{name}] Test-fold MAP + Laplace + MC marginal --")
        h_te, var_te, ll_map, ll_mrg = eval_em_on_test(
            spec, m, v, c_te, r_te, valid_te, test_ids,
            n_mc_samples=args.n_mc_samples, seed=args.seed)
        n_valid_per = valid_te.sum(axis=1).clip(min=1)
        print(f"  [{name}_EM] marginal NLL/trial: "
              f"mean={(-ll_mrg / n_valid_per).mean():.4f}  "
              f"sd={(-ll_mrg / n_valid_per).std():.4f}")

        # CSV filenames match what analyze_outer_cv.py already reads.
        csv_em_name = ("cog_model_results.csv" if name == "qlp"
                       else "ill_specified_map_results.csv")
        save_results_csv(f"{FOLD_DIR}/{csv_em_name}", test_ids, ll_mrg, valid_te,
                          "CogModel_EM" if name == "qlp" else "IllSpec_EM")

        # ── CP (common-process / single shared θ) ─────────────────────────────
        print(f"\n-- [{name}] CP fit (one shared θ over train) --")
        theta_cp = fit_cp(spec, c_tr, r_tr, valid_tr,
                          n_restarts=3, maxiter=500, seed=args.seed)
        ll_cp_te = eval_cp_on_test(spec, theta_cp,
                                     c_te, r_te, valid_te, test_ids)
        print(f"  [{name}_CP] NLL/trial: "
              f"mean={(-ll_cp_te / n_valid_per).mean():.4f}  "
              f"sd={(-ll_cp_te / n_valid_per).std():.4f}")

        csv_cp_name = ("cog_model_cp_results.csv" if name == "qlp"
                       else "ill_specified_cp_results.csv")
        save_results_csv(f"{FOLD_DIR}/{csv_cp_name}", test_ids, ll_cp_te, valid_te,
                          "CogModel_CP" if name == "qlp" else "IllSpec_CP")

        # ── Save full fitted state for optional reuse ─────────────────────────
        np.savez(f"{FOLD_DIR}/{name}_em_state.npz",
                 m=m, v=v, m_hist=m_hist, v_hist=v_hist,
                 h_train=h_tr, var_train=var_tr,
                 h_test=h_te,  var_test=var_te,
                 train_ids=train_ids, test_ids=test_ids,
                 ll_map=ll_map, ll_mrg=ll_mrg,
                 theta_cp=theta_cp, ll_cp_te=ll_cp_te)

    print(f"\nDone. Results in {FOLD_DIR}/")
