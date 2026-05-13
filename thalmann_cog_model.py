"""
thalmann_cog_model.py — Cognitive-model baselines for the Thalmann two-task dataset.

Fits four baselines per outer-CV fold:
  1. CogModel_EM — proper cognitive model, hierarchical-Bayesian EM
  2. CogModel_CP — proper cognitive model, common-process (one θ shared across train)
  3. IllSpec_EM  — ill-specified Q-learning, hierarchical-Bayesian EM
  4. IllSpec_CP  — ill-specified Q-learning, common process

Proper cognitive model (7 params per subject):
  - Task 0 (2-armed, Gershman 2018): Kalman filter (no diffusion), logistic choice
      with x = [E1-E2, √V1-√V2], params β0, β1, β2.
  - Task 1 (restless, Speekenbrink & Konstantinidis 2015): Kalman filter with
      diffusion decay (eq 6), UCB softmax choice, params τ, β_ucb, λ, C_decay.

Ill-specified Q-learning (2 params per subject):
  - Pure value tracking Q[c] ← Q[c] + α·(r̃ − Q[c]) with normalized r̃ = r/100.
  - Softmax choice P(c) ∝ exp(β·Q[c]); shared α, β across both tasks.
  - Per-block reset for 2-armed; single trajectory for restless.

Fixed generative constants (rewards in raw [1,100] scale):
  - 2-armed : σ²_innov = 0,    σ²_noise = 16,  E(0)=50, V(0)=100
  - restless: σ²_innov = 7.84, σ²_noise = 16,  E(0)=50, V(0)=100

Usage:
    python thalmann_cog_model.py --fold 0
"""

import os
import argparse
from math import exp, log, sqrt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR_BASE = "data_thalmann"

AB_N_BLOCKS  = 30
AB_BLOCK_LEN = 10
RB_N_TRIALS  = 200

SIG2_INNOV_AB = 0.0
SIG2_NOISE_AB = 16.0
SIG2_INNOV_RB = 7.84
SIG2_NOISE_RB = 16.0
E_INIT        = 50.0
V_INIT        = 100.0

# Normalisation constant for Q-learning (keeps α meaningful on [0,1])
REWARD_MAX = 100.0


# ══════════════════════════════════════════════════════════════════════════════
# Proper cognitive model (Gershman + Speekenbrink/UCB)
# ══════════════════════════════════════════════════════════════════════════════
N_PARAMS_COG = 7


def unpack_cog(pv):
    pv = np.asarray(pv, dtype=float)
    return dict(
        b0   = pv[0],
        b1   = pv[1],
        b2   = pv[2],
        tau  = np.exp(np.clip(pv[3], -10, 10)),
        buc  = pv[4],
        lam  = 1.0 / (1.0 + np.exp(-np.clip(pv[5], -15, 15))),
        cdec = pv[6],
    )


def ll_2armed(pv, ab_c_p, ab_r_p):
    """2-armed bandit log-likelihood (Gershman logistic), vectorised over blocks."""
    p = unpack_cog(pv)
    b0, b1, b2 = p["b0"], p["b1"], p["b2"]

    B        = AB_N_BLOCKS
    E        = np.full((B, 2), E_INIT, dtype=np.float64)
    V        = np.full((B, 2), V_INIT, dtype=np.float64)
    row_idx  = np.arange(B)
    ll       = 0.0

    for t in range(AB_BLOCK_LEN):
        dE    = E[:, 0] - E[:, 1]
        dSV   = np.sqrt(V[:, 0]) - np.sqrt(V[:, 1])
        logit = np.clip(b0 + b1 * dE + b2 * dSV, -30.0, 30.0)
        c     = ab_c_p[:, t]
        signed = (1 - 2 * c) * logit
        ll    += -np.log1p(np.exp(-signed)).sum()

        r       = ab_r_p[:, t]
        prior_V = V[row_idx, c] + SIG2_INNOV_AB
        K       = prior_V / (prior_V + SIG2_NOISE_AB)
        E[row_idx, c] = E[row_idx, c] + K * (r - E[row_idx, c])
        V[row_idx, c] = (1.0 - K) * prior_V
    return ll


def ll_restless(pv, rb_c_p, rb_r_p):
    """Restless bandit log-likelihood: Kalman+diffusion + UCB softmax. Scalar loop."""
    p    = unpack_cog(pv)
    tau  = p["tau"]
    buc  = p["buc"]
    lam  = p["lam"]
    cdec = p["cdec"]

    e0 = e1 = e2 = e3 = E_INIT
    v0 = v1 = v2 = v3 = V_INIT
    ll = 0.0
    c_arr, r_arr = rb_c_p, rb_r_p
    s2i, s2n = SIG2_INNOV_RB, SIG2_NOISE_RB

    for t in range(RB_N_TRIALS):
        u0 = tau * e0 + buc * sqrt(v0 + s2i)
        u1 = tau * e1 + buc * sqrt(v1 + s2i)
        u2 = tau * e2 + buc * sqrt(v2 + s2i)
        u3 = tau * e3 + buc * sqrt(v3 + s2i)
        um = max(u0, u1, u2, u3)
        log_Z = um + log(exp(u0-um) + exp(u1-um) + exp(u2-um) + exp(u3-um))

        c = int(c_arr[t])
        if   c == 0: ll += u0 - log_Z
        elif c == 1: ll += u1 - log_Z
        elif c == 2: ll += u2 - log_Z
        else:        ll += u3 - log_Z

        r = r_arr[t]
        if c == 0:
            pv_c = v0 + s2i; K = pv_c / (pv_c + s2n)
            e0 += K * (r - e0); v0 = (1.0 - K) * pv_c
            v1 += s2i; v2 += s2i; v3 += s2i
        elif c == 1:
            pv_c = v1 + s2i; K = pv_c / (pv_c + s2n)
            e1 += K * (r - e1); v1 = (1.0 - K) * pv_c
            v0 += s2i; v2 += s2i; v3 += s2i
        elif c == 2:
            pv_c = v2 + s2i; K = pv_c / (pv_c + s2n)
            e2 += K * (r - e2); v2 = (1.0 - K) * pv_c
            v0 += s2i; v1 += s2i; v3 += s2i
        else:
            pv_c = v3 + s2i; K = pv_c / (pv_c + s2n)
            e3 += K * (r - e3); v3 = (1.0 - K) * pv_c
            v0 += s2i; v1 += s2i; v2 += s2i

        one_m = 1.0 - lam
        e0 = lam * e0 + one_m * cdec
        e1 = lam * e1 + one_m * cdec
        e2 = lam * e2 + one_m * cdec
        e3 = lam * e3 + one_m * cdec
    return ll


def ll_cog(pv, ab_c_p, ab_r_p, rb_c_p, rb_r_p):
    """Joint log-likelihood across both tasks — proper cognitive model."""
    return ll_2armed(pv, ab_c_p, ab_r_p) + ll_restless(pv, rb_c_p, rb_r_p)


# ══════════════════════════════════════════════════════════════════════════════
# Ill-specified Q-learning model (value-based, no uncertainty tracking)
# ══════════════════════════════════════════════════════════════════════════════
N_PARAMS_ILLSPEC = 2


def unpack_illspec(pv):
    pv = np.asarray(pv, dtype=float)
    return dict(
        alpha = 1.0 / (1.0 + np.exp(-np.clip(pv[0], -15, 15))),
        beta  = np.exp(np.clip(pv[1], -10, 10)),
    )


def ll_q_2armed(pv, ab_c_p, ab_r_p):
    """Q-learning on 2-armed bandit (per-block reset). Vectorised over blocks."""
    p = unpack_illspec(pv)
    alpha, beta = p["alpha"], p["beta"]

    B   = AB_N_BLOCKS
    Q   = np.full((B, 2), 0.5, dtype=np.float64)     # normalised reward centre
    rix = np.arange(B)
    ll  = 0.0

    for t in range(AB_BLOCK_LEN):
        u     = beta * Q                              # (B, 2)
        um    = u.max(axis=1, keepdims=True)
        logZ  = um.squeeze(1) + np.log(np.exp(u - um).sum(axis=1))
        c     = ab_c_p[:, t]
        ll   += (u[rix, c] - logZ).sum()

        r_tilde = ab_r_p[:, t] / REWARD_MAX
        Q[rix, c] = Q[rix, c] + alpha * (r_tilde - Q[rix, c])
    return ll


def ll_q_restless(pv, rb_c_p, rb_r_p):
    """Q-learning on restless bandit (single trajectory, 4 arms). Scalar loop."""
    p = unpack_illspec(pv)
    alpha, beta = p["alpha"], p["beta"]

    q0 = q1 = q2 = q3 = 0.5
    ll = 0.0

    for t in range(RB_N_TRIALS):
        u0 = beta * q0; u1 = beta * q1; u2 = beta * q2; u3 = beta * q3
        um = max(u0, u1, u2, u3)
        logZ = um + log(exp(u0-um) + exp(u1-um) + exp(u2-um) + exp(u3-um))
        c = int(rb_c_p[t])
        if   c == 0: ll += u0 - logZ
        elif c == 1: ll += u1 - logZ
        elif c == 2: ll += u2 - logZ
        else:        ll += u3 - logZ

        r_tilde = rb_r_p[t] / REWARD_MAX
        if   c == 0: q0 += alpha * (r_tilde - q0)
        elif c == 1: q1 += alpha * (r_tilde - q1)
        elif c == 2: q2 += alpha * (r_tilde - q2)
        else:        q3 += alpha * (r_tilde - q3)
    return ll


def ll_illspec(pv, ab_c_p, ab_r_p, rb_c_p, rb_r_p):
    """Joint log-likelihood across both tasks — Q-learning ill-specified model."""
    return ll_q_2armed(pv, ab_c_p, ab_r_p) + ll_q_restless(pv, rb_c_p, rb_r_p)


# ══════════════════════════════════════════════════════════════════════════════
# Model specs — collect (name, n_params, init_m, init_v, ll_fn)
# ══════════════════════════════════════════════════════════════════════════════
COG_SPEC = dict(
    name     = "cog",
    n_params = N_PARAMS_COG,
    ll_fn    = ll_cog,
    init_m   = np.array([0.0, 0.05, 0.1, np.log(0.1), 0.3, 4.0, 50.0]),
    init_v   = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0]),
    param_names = ["β0", "β1", "β2", "log_τ", "β_ucb", "logit_λ", "C_decay"],
)

ILLSPEC_SPEC = dict(
    name     = "illspec",
    n_params = N_PARAMS_ILLSPEC,
    ll_fn    = ll_illspec,
    init_m   = np.array([0.0, 0.0]),                   # α≈0.5, β≈1
    init_v   = np.array([4.0, 4.0]),                   # broad prior
    param_names = ["logit_α", "log_β"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════
def load_subject_data(subids, df_ab_raw, df_rb_raw):
    P = len(subids)
    ab_c = np.zeros((P, AB_N_BLOCKS, AB_BLOCK_LEN), dtype=np.int64)
    ab_r = np.zeros((P, AB_N_BLOCKS, AB_BLOCK_LEN), dtype=np.float64)
    rb_c = np.zeros((P, RB_N_TRIALS), dtype=np.int64)
    rb_r = np.zeros((P, RB_N_TRIALS), dtype=np.float64)

    ab_sorted = df_ab_raw.sort_values(["ID", "block", "trial"])
    rb_sorted = df_rb_raw.sort_values(["ID", "trial"])

    for p_idx, sid in enumerate(subids):
        df_ab_s = ab_sorted[ab_sorted["ID"] == sid]
        for b_raw, grp in df_ab_s.groupby("block"):
            b = int(b_raw) - 1
            grp = grp.sort_values("trial")
            ab_c[p_idx, b, :] = grp["chosen"].values.astype(np.int64)
            ab_r[p_idx, b, :] = grp["reward"].values.astype(np.float64)

        df_rb_s = rb_sorted[rb_sorted["ID"] == sid].sort_values("trial")
        rb_c[p_idx] = df_rb_s["chosen"].values.astype(np.int64)
        rb_r[p_idx] = df_rb_s["reward"].values.astype(np.float64)
    return ab_c, ab_r, rb_c, rb_r


# ══════════════════════════════════════════════════════════════════════════════
# Generic EM + CP machinery (pass spec dict)
# ══════════════════════════════════════════════════════════════════════════════
def neg_log_posterior(pv, ab_c_p, ab_r_p, rb_c_p, rb_r_p, m, v, ll_fn):
    ll = ll_fn(pv, ab_c_p, ab_r_p, rb_c_p, rb_r_p)
    log_prior = np.sum(norm.logpdf(pv, loc=m, scale=np.sqrt(v)))
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


def estimate_individual(spec, m, v, ab_c_p, ab_r_p, rb_c_p, rb_r_p,
                         n_restarts=1, maxiter=200):
    ll_fn = spec["ll_fn"]; n_p = spec["n_params"]
    obj = lambda pv: neg_log_posterior(pv, ab_c_p, ab_r_p, rb_c_p, rb_r_p,
                                        m, v, ll_fn)
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


def run_em(spec, ab_c, ab_r, rb_c, rb_r, n_iter=20, seed=0):
    m = spec["init_m"].copy()
    v = spec["init_v"].copy()
    P = ab_c.shape[0]
    n_p = spec["n_params"]
    h_all   = np.zeros((P, n_p))
    var_all = np.zeros((P, n_p))
    m_hist, v_hist = [m.copy()], [v.copy()]

    np.random.seed(seed)
    for it in range(n_iter):
        for p in range(P):
            h_all[p], var_all[p] = estimate_individual(
                spec, m, v, ab_c[p], ab_r[p], rb_c[p], rb_r[p])
        m, v = update_group_prior(h_all, var_all)
        m_hist.append(m.copy()); v_hist.append(v.copy())
        if it % 5 == 0 or it == n_iter - 1:
            print(f"  [{spec['name']}] EM iter {it+1:3d}/{n_iter}  "
                  f"m={np.array2string(m, precision=3, suppress_small=True)}")
    return m, v, h_all, var_all, np.array(m_hist), np.array(v_hist)


def marginal_loglik_mc(spec, m, v, ab_c_p, ab_r_p, rb_c_p, rb_r_p,
                        n_samples=500, seed=0):
    rng = np.random.default_rng(seed)
    std = np.sqrt(v)
    samples = rng.normal(loc=m, scale=std, size=(n_samples, len(m)))
    ll_fn = spec["ll_fn"]
    lls = np.array([ll_fn(samples[s], ab_c_p, ab_r_p, rb_c_p, rb_r_p)
                    for s in range(n_samples)])
    return float(np.logaddexp.reduce(lls) - np.log(n_samples))


def fit_cp(spec, ab_c, ab_r, rb_c, rb_r, n_restarts=3, maxiter=500, seed=0):
    """Common-process fit: single θ shared across all training subjects."""
    ll_fn = spec["ll_fn"]
    n_p   = spec["n_params"]
    P     = ab_c.shape[0]

    def neg_joint(pv):
        return -sum(ll_fn(pv, ab_c[p], ab_r[p], rb_c[p], rb_r[p]) for p in range(P))

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
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════
N_VALID_TRIALS = AB_N_BLOCKS * AB_BLOCK_LEN + RB_N_TRIALS   # 500


def eval_em_on_test(spec, m, v, ab_c_te, ab_r_te, rb_c_te, rb_r_te,
                     test_ids, n_mc_samples=500, seed=0):
    """MAP + Laplace per test subj → marginal NLL via MC."""
    P = len(test_ids)
    n_p = spec["n_params"]
    h_te   = np.zeros((P, n_p))
    var_te = np.zeros((P, n_p))
    ll_map = np.zeros(P)
    ll_mrg = np.zeros(P)

    for p in range(P):
        h_te[p], var_te[p] = estimate_individual(
            spec, m, v, ab_c_te[p], ab_r_te[p], rb_c_te[p], rb_r_te[p],
            n_restarts=3, maxiter=500)
        ll_map[p] = spec["ll_fn"](h_te[p],
                                   ab_c_te[p], ab_r_te[p], rb_c_te[p], rb_r_te[p])
        ll_mrg[p] = marginal_loglik_mc(
            spec, m, v, ab_c_te[p], ab_r_te[p], rb_c_te[p], rb_r_te[p],
            n_samples=n_mc_samples, seed=seed + p)
    return h_te, var_te, ll_map, ll_mrg


def eval_cp_on_test(spec, theta_cp, ab_c_te, ab_r_te, rb_c_te, rb_r_te, test_ids):
    """Evaluate a single CP θ on all test subjects."""
    P = len(test_ids)
    ll = np.zeros(P)
    for p in range(P):
        ll[p] = spec["ll_fn"](theta_cp,
                               ab_c_te[p], ab_r_te[p], rb_c_te[p], rb_r_te[p])
    return ll


def save_results_csv(out_csv, test_ids, ll_total, model_name):
    """Save per-subject NLL (mean per valid trial + total). Total LL is negated."""
    pd.DataFrame({
        "subid": test_ids,
        "normalized_likelihood": -ll_total / N_VALID_TRIALS,   # mean NLL per trial
        "total_nll":             -ll_total,
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
    ap.add_argument("--only", choices=["cog", "illspec", "all"], default="all",
                    help="Limit to one model family")
    args = ap.parse_args()

    FOLD_DIR = f"{DATA_DIR_BASE}/fold{args.fold}"
    print(f"=== Thalmann cog-model baselines — fold {args.fold} ===")

    train_ids = pd.read_csv(f"{FOLD_DIR}/df_train.csv")["subid"].values
    test_ids  = pd.read_csv(f"{FOLD_DIR}/df_test.csv")["subid"].values
    print(f"  train subjects: {len(train_ids)}, test subjects: {len(test_ids)}")

    df_ab = pd.read_csv("data/final2armedBanditSession1.csv")
    df_rb = pd.read_csv("data/finalRestlessSession1.csv")
    ab_c_tr, ab_r_tr, rb_c_tr, rb_r_tr = load_subject_data(train_ids, df_ab, df_rb)
    ab_c_te, ab_r_te, rb_c_te, rb_r_te = load_subject_data(test_ids,  df_ab, df_rb)

    specs = [COG_SPEC, ILLSPEC_SPEC] if args.only == "all" else \
            [COG_SPEC] if args.only == "cog" else [ILLSPEC_SPEC]

    for spec in specs:
        name = spec["name"]
        print(f"\n{'='*72}\nModel: {name}\n{'='*72}")

        # ── EM (individual-differences hierarchical fit) ──────────────────────
        print(f"\n-- [{name}] EM on training fold --")
        m, v, h_tr, var_tr, m_hist, v_hist = run_em(
            spec, ab_c_tr, ab_r_tr, rb_c_tr, rb_r_tr,
            n_iter=args.n_em_iter, seed=args.seed)
        print(f"\n-- [{name}] Final group prior (unconstrained) --")
        for i, pname in enumerate(spec["param_names"]):
            print(f"  {pname:10s}  m={m[i]:+.4f}  v={v[i]:.4f}")

        print(f"\n-- [{name}] Test-fold MAP + Laplace + MC marginal --")
        h_te, var_te, ll_map, ll_mrg = eval_em_on_test(
            spec, m, v, ab_c_te, ab_r_te, rb_c_te, rb_r_te, test_ids,
            n_mc_samples=args.n_mc_samples, seed=args.seed)
        print(f"  [{name}_EM] marginal NLL/trial: "
              f"mean={-ll_mrg.mean()/N_VALID_TRIALS:.4f}  "
              f"sd={(-ll_mrg/N_VALID_TRIALS).std():.4f}")

        csv_name = "cog_model_results.csv" if name == "cog" else "illspec_em_results.csv"
        save_results_csv(f"{FOLD_DIR}/{csv_name}", test_ids, ll_mrg,
                          "CogModel_EM" if name == "cog" else "IllSpec_EM")

        # ── CP (common-process) fit ────────────────────────────────────────────
        print(f"\n-- [{name}] CP fit (one shared θ over train) --")
        theta_cp = fit_cp(spec, ab_c_tr, ab_r_tr, rb_c_tr, rb_r_tr,
                          n_restarts=3, maxiter=500, seed=args.seed)
        ll_cp_te = eval_cp_on_test(spec, theta_cp,
                                     ab_c_te, ab_r_te, rb_c_te, rb_r_te, test_ids)
        print(f"  [{name}_CP] NLL/trial: "
              f"mean={-ll_cp_te.mean()/N_VALID_TRIALS:.4f}  "
              f"sd={(-ll_cp_te/N_VALID_TRIALS).std():.4f}")

        csv_name_cp = "cog_model_cp_results.csv" if name == "cog" \
                       else "illspec_cp_results.csv"
        save_results_csv(f"{FOLD_DIR}/{csv_name_cp}", test_ids, ll_cp_te,
                          "CogModel_CP" if name == "cog" else "IllSpec_CP")

        # ── Save full state for optional reuse ─────────────────────────────────
        np.savez(f"{FOLD_DIR}/{name}_em_state.npz",
                 m=m, v=v, m_hist=m_hist, v_hist=v_hist,
                 h_train=h_tr, var_train=var_tr,
                 h_test=h_te,  var_test=var_te,
                 train_ids=train_ids, test_ids=test_ids,
                 ll_map=ll_map, ll_mrg=ll_mrg,
                 theta_cp=theta_cp, ll_cp_te=ll_cp_te)

    print(f"\nDone. Results in {FOLD_DIR}/")
