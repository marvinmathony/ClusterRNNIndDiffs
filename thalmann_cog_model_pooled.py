"""thalmann_cog_model_pooled.py — cog-model baselines on the POOLED S1+S2 data.

Same models as thalmann_cog_model.py (CogModel EM/CP, IllSpec EM/CP) but each
subject's data is RAGGED across sessions: up to 60 two-armed blocks (S1+S2) and
up to 2 restless trajectories (S1+S2); only 175/238 have S2.  Per-subject NLL is
normalised by that subject's own valid-trial count.

Reads fold subject ids from data_thalmann_s2/fold{F}/df_{train,test}.csv.
Writes the same CSV names (cog_model_results, cog_model_cp_results,
illspec_em_results, illspec_cp_results) into that fold dir.

Usage: python thalmann_cog_model_pooled.py --fold 0 [--only all]
"""
import os, argparse
from math import exp, log, sqrt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

AB_BLOCK_LEN = 10; RB_N_TRIALS = 200
SIG2_INNOV_AB, SIG2_NOISE_AB = 0.0, 16.0
SIG2_INNOV_RB, SIG2_NOISE_RB = 7.84, 16.0
E_INIT, V_INIT, REWARD_MAX = 50.0, 100.0, 100.0
N_PARAMS_COG, N_PARAMS_ILLSPEC = 7, 2


def unpack_cog(pv):
    pv = np.asarray(pv, float)
    return dict(b0=pv[0], b1=pv[1], b2=pv[2], tau=np.exp(np.clip(pv[3], -10, 10)),
                buc=pv[4], lam=1/(1+np.exp(-np.clip(pv[5], -15, 15))), cdec=pv[6])


def ll_2armed(pv, ab_c, ab_r):
    """ab_c/ab_r: (B,10) for this subject's B two-armed blocks (any B)."""
    p = unpack_cog(pv); b0, b1, b2 = p["b0"], p["b1"], p["b2"]
    B = ab_c.shape[0]
    if B == 0:
        return 0.0
    E = np.full((B, 2), E_INIT); V = np.full((B, 2), V_INIT); rix = np.arange(B); ll = 0.0
    for t in range(AB_BLOCK_LEN):
        logit = np.clip(b0 + b1*(E[:, 0]-E[:, 1]) + b2*(np.sqrt(V[:, 0])-np.sqrt(V[:, 1])), -30, 30)
        c = ab_c[:, t]; ll += -np.log1p(np.exp(-(1-2*c)*logit)).sum()
        r = ab_r[:, t]; pV = V[rix, c] + SIG2_INNOV_AB; K = pV/(pV+SIG2_NOISE_AB)
        E[rix, c] += K*(r-E[rix, c]); V[rix, c] = (1-K)*pV
    return ll


def ll_restless_one(pv, c_arr, r_arr):
    p = unpack_cog(pv); tau, buc, lam, cdec = p["tau"], p["buc"], p["lam"], p["cdec"]
    e = [E_INIT]*4; v = [V_INIT]*4; ll = 0.0; s2i, s2n = SIG2_INNOV_RB, SIG2_NOISE_RB
    for t in range(len(c_arr)):
        u = [tau*e[i]+buc*sqrt(v[i]+s2i) for i in range(4)]; um = max(u)
        logZ = um + log(sum(exp(ui-um) for ui in u)); c = int(c_arr[t]); ll += u[c]-logZ
        pV = v[c]+s2i; K = pV/(pV+s2n); e[c] += K*(r_arr[t]-e[c]); v[c] = (1-K)*pV
        for i in range(4):
            if i != c: v[i] += s2i
            e[i] = lam*e[i] + (1-lam)*cdec
    return ll


def ll_restless(pv, rb_list):
    return sum(ll_restless_one(pv, c, r) for c, r in rb_list)


def ll_cog(pv, ab_c, ab_r, rb_list):
    return ll_2armed(pv, ab_c, ab_r) + ll_restless(pv, rb_list)


def unpack_ill(pv):
    pv = np.asarray(pv, float)
    return 1/(1+np.exp(-np.clip(pv[0], -15, 15))), np.exp(np.clip(pv[1], -10, 10))


def ll_q_2armed(pv, ab_c, ab_r):
    a, beta = unpack_ill(pv); B = ab_c.shape[0]
    if B == 0:
        return 0.0
    Q = np.full((B, 2), 0.5); rix = np.arange(B); ll = 0.0
    for t in range(AB_BLOCK_LEN):
        u = beta*Q; um = u.max(1, keepdims=True); logZ = um.squeeze(1)+np.log(np.exp(u-um).sum(1))
        c = ab_c[:, t]; ll += (u[rix, c]-logZ).sum()
        Q[rix, c] += a*(ab_r[:, t]/REWARD_MAX - Q[rix, c])
    return ll


def ll_q_restless_one(pv, c_arr, r_arr):
    a, beta = unpack_ill(pv); q = [0.5]*4; ll = 0.0
    for t in range(len(c_arr)):
        u = [beta*qi for qi in q]; um = max(u); logZ = um+log(sum(exp(ui-um) for ui in u))
        c = int(c_arr[t]); ll += u[c]-logZ; q[c] += a*(r_arr[t]/REWARD_MAX - q[c])
    return ll


def ll_q_restless(pv, rb_list):
    return sum(ll_q_restless_one(pv, c, r) for c, r in rb_list)


def ll_illspec(pv, ab_c, ab_r, rb_list):
    return ll_q_2armed(pv, ab_c, ab_r) + ll_q_restless(pv, rb_list)


COG_SPEC = dict(name="cog", n_params=N_PARAMS_COG, ll_fn=ll_cog,
                init_m=np.array([0., .05, .1, np.log(.1), .3, 4., 50.]),
                init_v=np.array([1., 1., 1., 1., 1., 1., 100.]))
ILLSPEC_SPEC = dict(name="illspec", n_params=N_PARAMS_ILLSPEC, ll_fn=ll_illspec,
                    init_m=np.array([0., 0.]), init_v=np.array([4., 4.]))


# ── ragged pooled loading ──────────────────────────────────────────────────────
def load_subject_data(subids, df_ab, df_rb):
    """Returns per-subject: AB_C[list of (B,10)], AB_R[...], RB[list of [(c,r),...]],
    NVALID[list of int]."""
    AB_C, AB_R, RB, NVALID = [], [], [], []
    for sid in subids:
        a = df_ab[df_ab["ID"] == sid]
        blocks_c, blocks_r = [], []
        for (sess, blk), g in a.groupby(["session", "block"]):
            g = g.sort_values("trial")
            if len(g) >= AB_BLOCK_LEN:
                blocks_c.append(g["chosen"].values[:AB_BLOCK_LEN].astype(np.int64))
                blocks_r.append(g["reward"].values[:AB_BLOCK_LEN].astype(np.float64))
        ab_c = np.array(blocks_c) if blocks_c else np.zeros((0, AB_BLOCK_LEN), np.int64)
        ab_r = np.array(blocks_r) if blocks_r else np.zeros((0, AB_BLOCK_LEN), np.float64)
        r = df_rb[df_rb["ID"] == sid]; trajs = []
        for sess, g in r.groupby("session"):
            g = g.sort_values("trial")
            if len(g) >= RB_N_TRIALS:
                trajs.append((g["chosen"].values[:RB_N_TRIALS].astype(np.int64),
                              g["reward"].values[:RB_N_TRIALS].astype(np.float64)))
        AB_C.append(ab_c); AB_R.append(ab_r); RB.append(trajs)
        NVALID.append(ab_c.shape[0]*AB_BLOCK_LEN + len(trajs)*RB_N_TRIALS)
    return AB_C, AB_R, RB, np.array(NVALID)


def neg_post(pv, ac, ar, rb, m, v, ll_fn):
    return -(ll_fn(pv, ac, ar, rb) + np.sum(norm.logpdf(pv, m, np.sqrt(v))))


def hess_diag(fun, x, eps=1e-4):
    x = np.asarray(x, float); f0 = fun(x); d = np.zeros(len(x))
    for j in range(len(x)):
        xp = x.copy(); xp[j] += eps; xm = x.copy(); xm[j] -= eps
        d[j] = (fun(xp)-2*f0+fun(xm))/eps**2
    return d


def estimate_individual(spec, m, v, ac, ar, rb, n_restarts=1, maxiter=200):
    obj = lambda pv: neg_post(pv, ac, ar, rb, m, v, spec["ll_fn"])
    res = minimize(obj, m.copy(), method="L-BFGS-B", options=dict(maxiter=maxiter, ftol=1e-6))
    bv, bx = res.fun, res.x.copy()
    for _ in range(n_restarts-1):
        r = minimize(obj, m+np.random.randn(spec["n_params"])*np.sqrt(v)*.5,
                     method="L-BFGS-B", options=dict(maxiter=maxiter, ftol=1e-6))
        if r.fun < bv: bv, bx = r.fun, r.x.copy()
    var = np.where(hess_diag(obj, bx) > 1e-6, 1.0/np.clip(hess_diag(obj, bx), 1e-6, None), v)
    return bx, np.clip(var, 1e-4, 10.0)


def run_em(spec, AC, AR, RB, n_iter=20, seed=0):
    m, v = spec["init_m"].copy(), spec["init_v"].copy(); P = len(AC); n_p = spec["n_params"]
    h = np.zeros((P, n_p)); va = np.zeros((P, n_p)); np.random.seed(seed)
    for it in range(n_iter):
        for p in range(P):
            h[p], va[p] = estimate_individual(spec, m, v, AC[p], AR[p], RB[p])
        m = h.mean(0); v = np.clip((h**2+va).mean(0)-m**2, 1e-4, 10.0)
        if it % 5 == 0 or it == n_iter-1:
            print(f"  [{spec['name']}] EM {it+1}/{n_iter} m={np.array2string(m,precision=3,suppress_small=True)}")
    return m, v, h, va


def marg_ll_mc(spec, m, v, ac, ar, rb, n_samples=500, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.normal(m, np.sqrt(v), size=(n_samples, len(m)))
    lls = np.array([spec["ll_fn"](S[s], ac, ar, rb) for s in range(n_samples)])
    return float(np.logaddexp.reduce(lls)-np.log(n_samples))


def fit_cp(spec, AC, AR, RB, n_restarts=3, maxiter=500, seed=0):
    nj = lambda pv: -sum(spec["ll_fn"](pv, AC[p], AR[p], RB[p]) for p in range(len(AC)))
    np.random.seed(seed); bv, bx = np.inf, spec["init_m"].copy()
    for r in range(n_restarts):
        x0 = spec["init_m"] if r == 0 else spec["init_m"]+np.random.randn(spec["n_params"])*np.sqrt(spec["init_v"])*.5
        res = minimize(nj, x0, method="L-BFGS-B", options=dict(maxiter=maxiter, ftol=1e-6))
        if res.fun < bv: bv, bx = res.fun, res.x.copy()
    print(f"  [{spec['name']}] CP joint_nll={bv:.1f}")
    return bx


def save_csv(out, ids, ll_total, nvalid, model):
    pd.DataFrame({"subid": ids, "normalized_likelihood": -ll_total/nvalid,
                  "total_nll": -ll_total, "model": model}).to_csv(out, index=False)
    print(f"  saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--n_em_iter", type=int, default=20)
    ap.add_argument("--n_mc_samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["cog", "illspec", "all"], default="all")
    args = ap.parse_args()
    FD = f"data_thalmann_s2/fold{args.fold}"
    print(f"=== POOLED cog baselines — fold {args.fold} ===")
    tr = pd.read_csv(f"{FD}/df_train.csv")["subid"].values
    te = pd.read_csv(f"{FD}/df_test.csv")["subid"].values
    df_ab = pd.concat([pd.read_csv("data/final2armedBanditSession1.csv"),
                       pd.read_csv("data/final2armedBanditSession2.csv")], ignore_index=True)
    df_rb = pd.concat([pd.read_csv("data/finalRestlessSession1.csv"),
                       pd.read_csv("data/finalRestlessSession2.csv")], ignore_index=True)
    AC_tr, AR_tr, RB_tr, NV_tr = load_subject_data(tr, df_ab, df_rb)
    AC_te, AR_te, RB_te, NV_te = load_subject_data(te, df_ab, df_rb)
    print(f"  train {len(tr)} test {len(te)}; nvalid/te min/median/max="
          f"{NV_te.min()}/{int(np.median(NV_te))}/{NV_te.max()}")
    specs = [COG_SPEC, ILLSPEC_SPEC] if args.only == "all" else \
            [COG_SPEC] if args.only == "cog" else [ILLSPEC_SPEC]
    for spec in specs:
        nm = spec["name"]; print(f"\n=== {nm} ===")
        m, v, h, va = run_em(spec, AC_tr, AR_tr, RB_tr, n_iter=args.n_em_iter, seed=args.seed)
        ll_mrg = np.array([marg_ll_mc(spec, m, v, AC_te[p], AR_te[p], RB_te[p],
                                      n_samples=args.n_mc_samples, seed=args.seed+p) for p in range(len(te))])
        save_csv(f"{FD}/{'cog_model_results' if nm=='cog' else 'illspec_em_results'}.csv",
                 te, ll_mrg, NV_te, "CogModel_EM" if nm == "cog" else "IllSpec_EM")
        theta = fit_cp(spec, AC_tr, AR_tr, RB_tr, seed=args.seed)
        ll_cp = np.array([spec["ll_fn"](theta, AC_te[p], AR_te[p], RB_te[p]) for p in range(len(te))])
        save_csv(f"{FD}/{'cog_model_cp_results' if nm=='cog' else 'illspec_cp_results'}.csv",
                 te, ll_cp, NV_te, "CogModel_CP" if nm == "cog" else "IllSpec_CP")
    print(f"\nDone -> {FD}/")
