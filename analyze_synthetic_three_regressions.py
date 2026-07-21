"""
analyze_synthetic_three_regressions.py — R1 / R2 / R3 analysis on a synthetic
uniform-α dataset, mirroring the cross-task regret logic from
analyze_thalmann_logit_updates.py but adapted for the single-task setting.

Target: per-session true α from data_dataset{ID}/params_dict.pkl['Q_MAP'].

Three single-feature predictors of α (Pearson r reported with bootstrap CI):
  R1 — raw observed behaviour       (mean reward per session, no model)
  R2 — exact-env model rollout      (replay session's own reward sequence)
  R3 — marginalised-env model rollout (envs sampled fresh from gen_reward_seq)

Models compared: IDRNN, Vanilla+h, Vanilla.

Inputs:
  plots_dataset{ID}/step1_vs_vanilla/latents_idrnn_step1_bestseed*.pt
  plots_dataset{ID}/step1_vs_vanilla/latents_vanilla_bestseed*.pt
  data_dataset{ID}/params_dict.pkl, df_train.csv, rewards_train.npy

Outputs:
  plots_dataset{ID}/step1_three_regressions.{npz,png}

Usage: python analyze_synthetic_three_regressions.py --dataset_id 0
"""
import os, glob, json, pickle, argparse, random as pyrandom
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modelsandtraining import Decoder, AblatedRNN


# Inlined from sim_Q_data.gen_reward_seq to avoid pulling in TensorFlow at
# import time. Match the upstream RNG calls exactly so behaviour is identical:
# the first numpy call is a 3-element Dirichlet draw, then per-N: a numpy
# choice over envs, pyrandom.uniform for the env-specific params (low/high),
# and a 2*T numpy rand draw for the binary rewards.
def gen_reward_seq(seed=1979, T=200, interval=50, N=None):
    np.random.seed(seed)
    pyrandom.seed(seed)
    np.random.dirichlet([1, 1, 1])
    environments = ["low", "normal", "high"]
    rewards = np.zeros((N, 2, T), dtype=np.float32)
    for n in range(N):
        selected_env = np.random.choice(environments)
        if selected_env == "low":
            pHigh = pyrandom.uniform(0.1, 0.3)
            gap   = pyrandom.uniform(0.05, 0.15)
            pLow  = pHigh - gap
        elif selected_env == "normal":
            pHigh = 0.8
            pLow  = 0.2
        else:  # "high"
            pHigh = pyrandom.uniform(0.7, 0.95)
            gap   = pyrandom.uniform(0.2, 0.4)
            pLow  = max(0.05, pHigh - gap)
        u = np.random.rand(2 * T).reshape(2, T)
        for t in range(T):
            if t % (interval * 2) < interval:
                rewards[n, 0, t] = 1 if u[0, t] < pHigh else 0
                rewards[n, 1, t] = 1 if u[1, t] < pLow  else 0
            else:
                rewards[n, 1, t] = 1 if u[0, t] < pHigh else 0
                rewards[n, 0, t] = 1 if u[1, t] < pLow  else 0
    return rewards


N_RNG_SEEDS = 100   # rollouts per session
A           = 2


# ── Encoder for the 4-dim CIR-with-reward feature (matches sim_Q_data) ──────
def encode_prev(prev_c, prev_r):
    """Single-trial feature vector reflecting the previous (c, r) pair.
    Format matches generate_xin in sim_Q_data.py:
      feat[0] = 1 if (prev_c==0 & prev_r==1)
      feat[1] = 1 if (prev_c==1 & prev_r==1)
      feat[2] = 1 if (prev_c==0 & any reward)
      feat[3] = 1 if (prev_c==1 & any reward)
    """
    x = np.zeros(4, dtype=np.float32)
    if prev_c == 0 and prev_r == 1: x[0] = 1; x[2] = 1
    if prev_c == 1 and prev_r == 1: x[1] = 1; x[3] = 1
    if prev_c == 0 and prev_r == 0: x[2] = 1
    if prev_c == 1 and prev_r == 0: x[3] = 1
    return x


# ── Rollouts ────────────────────────────────────────────────────────────────
@torch.no_grad()
def rollout_idrnn(z_vec, env_2xT, rng, decoder, device, reward_max=1.0):
    """On-policy IDRNN rollout. env_2xT: (2, T) per-arm rewards (Bernoulli 0/1
    for synthetic data). Returns (rewards_received, choices) length T."""
    T = env_2xT.shape[1]
    z_t = torch.as_tensor(z_vec, dtype=torch.float32, device=device).unsqueeze(0)
    h = decoder.z2h0(z_t).unsqueeze(0)
    prev_x = np.zeros(4, dtype=np.float32)
    rewards = np.zeros(T, dtype=np.float32)
    choices = np.zeros(T, dtype=np.int64)
    for t in range(T):
        x_in = torch.as_tensor(prev_x, dtype=torch.float32,
                               device=device).unsqueeze(0).unsqueeze(0)
        logits, h = decoder(x_in, z_t, hidden=h)
        p = F.softmax(logits[0, 0], dim=-1).detach().cpu().numpy()
        arm = int(rng.choice(A, p=p))
        r   = float(env_2xT[arm, t])
        choices[t] = arm
        rewards[t] = r
        prev_x = encode_prev(arm, int(round(r)))
    return rewards, choices


@torch.no_grad()
def rollout_vanilla(env_2xT, h_init, rng, vmodel, device, hidden_size):
    """On-policy Vanilla rollout. h_init: None for zero-init, or (hidden,) for +h."""
    T = env_2xT.shape[1]
    if h_init is None:
        h = torch.zeros(1, 1, hidden_size, device=device)
    else:
        h = torch.as_tensor(h_init, dtype=torch.float32,
                            device=device).reshape(1, 1, hidden_size)
    prev_x = np.zeros(4, dtype=np.float32)
    rewards = np.zeros(T, dtype=np.float32)
    choices = np.zeros(T, dtype=np.int64)
    for t in range(T):
        x_in = torch.as_tensor(prev_x, dtype=torch.float32,
                               device=device).unsqueeze(0).unsqueeze(0)
        logits, h, _ = vmodel.dec(x_in, h0=h)
        p = F.softmax(logits[0, 0], dim=-1).detach().cpu().numpy()
        arm = int(rng.choice(A, p=p))
        r   = float(env_2xT[arm, t])
        choices[t] = arm
        rewards[t] = r
        prev_x = encode_prev(arm, int(round(r)))
    return rewards, choices


# ── Stats helpers ───────────────────────────────────────────────────────────
def r_with_ci(x, y, n_boot=1000, seed=0):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan, (np.nan, np.nan), 0
    xv, yv = x[m], y[m]
    r0 = float(np.corrcoef(xv, yv)[0, 1])
    rng = np.random.default_rng(seed)
    n = len(xv)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a, b = xv[idx], yv[idx]
        if a.std() == 0 or b.std() == 0:
            boots[i] = np.nan
        else:
            boots[i] = np.corrcoef(a, b)[0, 1]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return r0, (float(lo), float(hi)), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", type=int, default=0)
    ap.add_argument("--n_rng_seeds", type=int, default=N_RNG_SEEDS)
    ap.add_argument("--n_marg_envs", type=int, default=N_RNG_SEEDS)
    ap.add_argument("--dgp", type=str, default=None,
                    help="DGP subdir prefix, e.g. 'norev'.")
    ap.add_argument("--marg_interval", type=int, default=50,
                    help="Reversal interval used for the marg-env sample. "
                         "Should match the training-data interval.")
    ap.add_argument("--add_vanH_lasttime", action="store_true",
                    help="Also run Vanilla+h R2/R3 with h_init = LAST-timestep "
                         "hidden state (instead of time-mean).  Forwards the "
                         "canonical vanilla model on xin_train.npy to get "
                         "h_last per subject, runs the rollouts, and adds "
                         "R2_vanH_lastT / R3_vanH_lastT (+ bootstraps) to "
                         "the output npz.")
    args = ap.parse_args()

    if args.dgp:
        DATA_DIR = f"data_{args.dgp}_dataset{args.dataset_id}"
        PLOT_DIR = f"plots_{args.dgp}_dataset{args.dataset_id}"
    else:
        DATA_DIR = f"data_dataset{args.dataset_id}"
        PLOT_DIR = f"plots_dataset{args.dataset_id}"
    LATENT_DIR = f"{PLOT_DIR}/step1_vs_vanilla"
    os.makedirs(PLOT_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # ── Load IDRNN ────────────────────────────────────────────────────────────
    idrnn_files = glob.glob(f"{LATENT_DIR}/latents_idrnn_step1_bestseed*.pt")
    assert idrnn_files, f"No step-1 IDRNN latents in {LATENT_DIR}"
    idrnn_path = max(idrnn_files, key=os.path.getmtime)
    print(f"Loading IDRNN: {idrnn_path}")
    saved = torch.load(idrnn_path, map_location=DEVICE, weights_only=False)
    Z_DIM       = saved["z_dim"]
    HIDDEN      = saved["hidden"]
    BASE_IN_DIM = saved["base_in_dim"]
    A_loaded    = saved["A"]
    assert A_loaded == A, f"A mismatch: loaded {A_loaded}, expected {A}"
    emb         = np.asarray(saved["z"])
    n_sessions  = emb.shape[0]
    print(f"  z_dim={Z_DIM}, hidden={HIDDEN}, n_sessions={n_sessions}")
    state = saved["model_state"]
    decoder = Decoder(in_dim=BASE_IN_DIM, z_dim=Z_DIM, hid=HIDDEN, A=A)
    decoder.load_state_dict({k[8:]: v for k, v in state.items()
                             if k.startswith("decoder.")})
    decoder.to(DEVICE).eval()

    # ── Load Vanilla ──────────────────────────────────────────────────────────
    vanilla_files = glob.glob(f"{LATENT_DIR}/latents_vanilla_bestseed*.pt")
    assert vanilla_files, f"No Vanilla latents in {LATENT_DIR}"
    vanilla_path = max(vanilla_files, key=os.path.getmtime)
    print(f"Loading Vanilla: {vanilla_path}")
    v_saved = torch.load(vanilla_path, map_location=DEVICE, weights_only=False)
    V_HIDDEN = v_saved["hidden"]
    vmodel = AblatedRNN(hid=V_HIDDEN, in_dim=BASE_IN_DIM, A=A,
                        block_structure=False, n_tasks=None, task_emb_dim=0).to(DEVICE)
    vmodel.load_state_dict(v_saved["model_state"])
    vmodel.eval()
    v_h_raw = np.asarray(v_saved["h"])   # (n_sessions, V_HIDDEN) — time-mean
    print(f"  V_HIDDEN={V_HIDDEN}")

    v_h_lastT_raw = None
    if args.add_vanH_lasttime:
        xin_train_np = np.load(f"{DATA_DIR}/xin_train.npy")
        xin_train = torch.from_numpy(xin_train_np).float().to(DEVICE)
        with torch.no_grad():
            _, _, hidden_tr = vmodel(xin_train)
        if hidden_tr.dim() == 4: hidden_tr = hidden_tr.squeeze(1)
        v_h_lastT_raw = hidden_tr[:, -1, :].cpu().numpy()
        print(f"  Computed v_h_lastT_raw, shape={v_h_lastT_raw.shape}")

    # ── Dimension-match h_init to IDRNN's z_dim via rank-1 PC reconstruction ─
    # Other representational panels (b, d) decode vanilla from PC1 of h to match
    # IDRNN's z_dim=1.  Apply the same constraint here: project h onto its top-1
    # principal component, then reconstruct back to the V_HIDDEN-dim space so
    # the rollout's GRU receives a vector of the expected shape but constrained
    # to a 1-D subspace.  Across subjects, h_init now varies along ONE direction
    # only, just like IDRNN's z varies along ONE dim.
    from sklearn.decomposition import PCA as _PCA
    pca_h = _PCA(n_components=Z_DIM)
    v_h = pca_h.inverse_transform(pca_h.fit_transform(v_h_raw))   # (n_sub, V_HIDDEN)
    print(f"  v_h dim-matched: PC1 explains {pca_h.explained_variance_ratio_[0]*100:.1f}% of h variance")
    v_h_lastT = None
    if v_h_lastT_raw is not None:
        pca_h_last = _PCA(n_components=Z_DIM)
        v_h_lastT = pca_h_last.inverse_transform(pca_h_last.fit_transform(v_h_lastT_raw))
        print(f"  v_h_lastT dim-matched: PC1 explains "
              f"{pca_h_last.explained_variance_ratio_[0]*100:.1f}% of h_last variance")

    # ── Load TRUE α per session (from generate_parameter_lists, not MAP fit) ─
    true_alpha_csv = f"{DATA_DIR}/true_parameter_values.csv"
    if os.path.exists(true_alpha_csv):
        true_df = pd.read_csv(true_alpha_csv)
        true_alpha = true_df["alphaP_list"].values.astype(float)
        true_beta  = true_df["beta_list"].values.astype(float)
        print(f"  Loaded TRUE α from {true_alpha_csv}")
    else:
        with open(f"{DATA_DIR}/params_dict.pkl", "rb") as f:
            pdict = pickle.load(f)
        true_alpha = np.array([pdict["Q_MAP"][i][0] for i in range(n_sessions)])
        true_beta  = np.array([pdict["Q_MAP"][i][1] for i in range(n_sessions)])
        print(f"  No true_parameter_values.csv — falling back to MAP from params_dict.pkl")
    print(f"  α: range=[{true_alpha.min():.3f}, {true_alpha.max():.3f}], "
          f"mean={true_alpha.mean():.3f}, sd={true_alpha.std():.3f}")
    if np.allclose(true_beta, true_beta[0]):
        print(f"  β: constant at {true_beta[0]:.3f}")
    else:
        print(f"  β: varies — range=[{true_beta.min():.3f}, {true_beta.max():.3f}]")

    # ── Observed sequences ────────────────────────────────────────────────────
    df = pd.read_csv(f"{DATA_DIR}/df_train.csv")
    obs = df.pivot(index="session", columns="trial",
                   values=["c", "r"]).sort_index()
    obs_c = obs["c"].to_numpy(dtype=np.int64)   # (n_sessions, T)
    obs_r = obs["r"].to_numpy(dtype=np.float32)
    T = obs_c.shape[1]
    print(f"  observed: {obs_c.shape[0]} sessions × {T} trials")

    # Build per-session env: (2, T) reward array (rewards_train.npy if available)
    rewards_path = f"{DATA_DIR}/rewards_train.npy"
    if os.path.exists(rewards_path):
        rewards = np.load(rewards_path)
        # Shape could be (n_sessions, 2, T) or (2, T) shared
        if rewards.ndim == 2:
            envs_train = np.broadcast_to(rewards[None], (n_sessions, 2, T)).copy()
        elif rewards.ndim == 3 and rewards.shape[0] == n_sessions:
            envs_train = rewards.astype(np.float32)
        else:
            envs_train = None
        if envs_train is not None:
            print(f"  rewards_train.npy → envs_train shape {envs_train.shape}")
    else:
        envs_train = None

    if envs_train is None:
        print("  rewards_train.npy missing/unrecognised — "
              "reconstructing per-session env from observed (c, r) only "
              "(non-chosen arm = NaN). R2 cannot use counterfactuals.")
        envs_train = np.full((n_sessions, 2, T), np.nan, dtype=np.float32)
        for s in range(n_sessions):
            for t in range(T):
                envs_train[s, obs_c[s, t], t] = obs_r[s, t]

    # ── R1: observed mean regret per session ─────────────────────────────────
    # Regret_t = max(reward1_t, reward2_t) − received_t.
    # Higher α → faster learning → lower regret → expected negative correlation.
    if envs_train is not None:
        # max-arm reward per trial available from rewards_train (counterfactuals)
        max_arm = envs_train.max(axis=1)   # (n_sessions, T)
        R1 = (max_arm - obs_r).mean(axis=1)
        print(f"  R1 (observed mean regret): mean={R1.mean():.3f}, sd={R1.std():.3f}")
    else:
        # Should not happen with rewards_train.npy present, but fall back to
        # mean reward (sign flipped) so the analysis still runs.
        R1 = obs_r.mean(axis=1)
        print(f"  R1 (raw mean reward, regret unavailable): mean={R1.mean():.3f}, "
              f"sd={R1.std():.3f}")

    # ── Build marginalised env set (shared across sessions for R3) ────────────
    print(f"\nSampling {args.n_marg_envs} marginalised envs from gen_reward_seq "
          f"(interval={args.marg_interval}) ...")
    MARG = gen_reward_seq(seed=99 + args.dataset_id, T=T, N=args.n_marg_envs,
                          interval=args.marg_interval)
    print(f"  MARG envs shape: {MARG.shape}  (N, 2, T)")

    # ── Per-session rollouts: IDRNN exact, IDRNN marg, Vanilla, Vanilla+h ────
    print(f"\nRunning rollouts for {n_sessions} sessions × "
          f"{args.n_rng_seeds} RNG seeds × {args.n_marg_envs} marg envs ...")
    # R0 = "No Marg." — a SINGLE rollout on the subject's exact env (the k=0
    # term of the R2 loop, i.e. R2 with K=1; matches Dezfouli's R0 rung).
    sim_idrnn_R0 = np.full(n_sessions, np.nan)
    sim_van0_R0  = np.full(n_sessions, np.nan)
    sim_vanH_R0  = np.full(n_sessions, np.nan)
    sim_vanH_lastT_R0 = np.full(n_sessions, np.nan)
    sim_idrnn_R2 = np.full(n_sessions, np.nan)
    sim_idrnn_R3 = np.full(n_sessions, np.nan)
    sim_van0_R2  = np.full(n_sessions, np.nan)
    sim_van0_R3  = np.full(n_sessions, np.nan)
    sim_vanH_R2  = np.full(n_sessions, np.nan)
    sim_vanH_R3  = np.full(n_sessions, np.nan)
    sim_vanH_lastT_R2 = np.full(n_sessions, np.nan)
    sim_vanH_lastT_R3 = np.full(n_sessions, np.nan)

    for s in range(n_sessions):
        z_vec = emb[s]
        h_init_v = v_h[s] if np.all(np.isfinite(v_h[s])) else None

        env_exact = envs_train[s]   # (2, T) — may contain NaN if reconstructed
        usable_exact = np.all(np.isfinite(env_exact))
        max_exact = env_exact.max(axis=0) if usable_exact else None  # (T,)

        # IDRNN
        if usable_exact:
            r2_sum = 0.0
            for k in range(args.n_rng_seeds):
                rng = np.random.default_rng(k * 9973 + s + 7)
                rew, _ = rollout_idrnn(z_vec, env_exact, rng, decoder, DEVICE)
                reg = (max_exact - rew).mean()
                if k == 0:
                    sim_idrnn_R0[s] = reg          # No Marg. — single rollout
                r2_sum += reg
            sim_idrnn_R2[s] = r2_sum / args.n_rng_seeds
        r3_sum = 0.0
        for k in range(args.n_marg_envs):
            rng = np.random.default_rng(k * 7919 + s + 11)
            rew, _ = rollout_idrnn(z_vec, MARG[k], rng, decoder, DEVICE)
            r3_sum += (MARG[k].max(axis=0) - rew).mean()
        sim_idrnn_R3[s] = r3_sum / args.n_marg_envs

        # Vanilla zero
        if usable_exact:
            r2_sum = 0.0
            for k in range(args.n_rng_seeds):
                rng = np.random.default_rng(k * 9973 + s + 13)
                rew, _ = rollout_vanilla(env_exact, None, rng,
                                         vmodel, DEVICE, V_HIDDEN)
                reg = (max_exact - rew).mean()
                if k == 0:
                    sim_van0_R0[s] = reg           # No Marg. — single rollout
                r2_sum += reg
            sim_van0_R2[s] = r2_sum / args.n_rng_seeds
        r3_sum = 0.0
        for k in range(args.n_marg_envs):
            rng = np.random.default_rng(k * 7919 + s + 17)
            rew, _ = rollout_vanilla(MARG[k], None, rng,
                                     vmodel, DEVICE, V_HIDDEN)
            r3_sum += (MARG[k].max(axis=0) - rew).mean()
        sim_van0_R3[s] = r3_sum / args.n_marg_envs

        # Vanilla+h
        if h_init_v is not None:
            if usable_exact:
                r2_sum = 0.0
                for k in range(args.n_rng_seeds):
                    rng = np.random.default_rng(k * 9973 + s + 19)
                    rew, _ = rollout_vanilla(env_exact, h_init_v, rng,
                                             vmodel, DEVICE, V_HIDDEN)
                    reg = (max_exact - rew).mean()
                    if k == 0:
                        sim_vanH_R0[s] = reg       # No Marg. — single rollout
                    r2_sum += reg
                sim_vanH_R2[s] = r2_sum / args.n_rng_seeds
            r3_sum = 0.0
            for k in range(args.n_marg_envs):
                rng = np.random.default_rng(k * 7919 + s + 23)
                rew, _ = rollout_vanilla(MARG[k], h_init_v, rng,
                                         vmodel, DEVICE, V_HIDDEN)
                r3_sum += (MARG[k].max(axis=0) - rew).mean()
            sim_vanH_R3[s] = r3_sum / args.n_marg_envs

        # Vanilla+h LAST-timestep variant (same rollout structure, different h_init)
        if v_h_lastT is not None and np.all(np.isfinite(v_h_lastT[s])):
            h_init_last = v_h_lastT[s]
            if usable_exact:
                r2_sum = 0.0
                for k in range(args.n_rng_seeds):
                    rng = np.random.default_rng(k * 9973 + s + 29)
                    rew, _ = rollout_vanilla(env_exact, h_init_last, rng,
                                             vmodel, DEVICE, V_HIDDEN)
                    reg = (max_exact - rew).mean()
                    if k == 0:
                        sim_vanH_lastT_R0[s] = reg  # No Marg. — single rollout
                    r2_sum += reg
                sim_vanH_lastT_R2[s] = r2_sum / args.n_rng_seeds
            r3_sum = 0.0
            for k in range(args.n_marg_envs):
                rng = np.random.default_rng(k * 7919 + s + 31)
                rew, _ = rollout_vanilla(MARG[k], h_init_last, rng,
                                         vmodel, DEVICE, V_HIDDEN)
                r3_sum += (MARG[k].max(axis=0) - rew).mean()
            sim_vanH_lastT_R3[s] = r3_sum / args.n_marg_envs

        if (s + 1) % 25 == 0:
            print(f"  {s+1}/{n_sessions} sessions done")

    # ── Compute correlations with bootstrap CIs ──────────────────────────────
    BLOCKS = [
        ("R1 Observed",            "Human",     R1),
        ("R0 NoMarg IDRNN",        "IDRNN",     sim_idrnn_R0),
        ("R0 NoMarg Van+h",        "Vanilla+h", sim_vanH_R0),
        ("R0 NoMarg Van",          "Vanilla",   sim_van0_R0),
        ("R2 NoiseMarg IDRNN",     "IDRNN",     sim_idrnn_R2),
        ("R2 NoiseMarg Van+h",     "Vanilla+h", sim_vanH_R2),
        ("R2 NoiseMarg Van",       "Vanilla",   sim_van0_R2),
        ("R3 Env+NoiseMarg IDRNN", "IDRNN",     sim_idrnn_R3),
        ("R3 Env+NoiseMarg Van+h", "Vanilla+h", sim_vanH_R3),
        ("R3 Env+NoiseMarg Van",   "Vanilla",   sim_van0_R3),
    ]

    print("\nResults (Pearson r [95% CI] vs true α):")
    print(f"{'block':<22}{'n':>5}{'r [CI]':>26}")
    results = {}
    for name, _, x in BLOCKS:
        r, ci, n = r_with_ci(x, true_alpha, seed=1)
        results[name] = {"r": r, "ci": ci, "n": n}
        print(f"{name:<22}{n:>5}   {r:+.3f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")

    # ── Save arrays ───────────────────────────────────────────────────────────
    out_npz = os.path.join(PLOT_DIR, "step1_three_regressions.npz")
    save_kwargs = dict(
        true_alpha=true_alpha, true_beta=true_beta,
        R1_raw_mean_reward=R1,
        R0_idrnn=sim_idrnn_R0, R2_idrnn=sim_idrnn_R2, R3_idrnn=sim_idrnn_R3,
        R0_vanH=sim_vanH_R0,   R2_vanH=sim_vanH_R2,   R3_vanH=sim_vanH_R3,
        R0_van0=sim_van0_R0,   R2_van0=sim_van0_R2,   R3_van0=sim_van0_R3,
    )
    if args.add_vanH_lasttime:
        save_kwargs["R0_vanH_lastT"] = sim_vanH_lastT_R0
        save_kwargs["R2_vanH_lastT"] = sim_vanH_lastT_R2
        save_kwargs["R3_vanH_lastT"] = sim_vanH_lastT_R3
    np.savez(out_npz, **save_kwargs)
    print(f"Saved → {out_npz}")

    # ── Plot: scatter grid (7 panels) + headline bar ─────────────────────────
    MODEL_COLORS = {
        "Human":     "#666666",
        "IDRNN":     "#4C72B0",
        "Vanilla+h": "#8C564B",
        "Vanilla":   "#DD8452",
    }

    fig = plt.figure(figsize=(16, 15))
    gs  = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.1], hspace=0.5, wspace=0.35)

    # 10 scatter panels in rows 0-2 (R1 + R0/R2/R3 × 3 archs)
    for i, (name, model, x) in enumerate(BLOCKS):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        m = np.isfinite(x) & np.isfinite(true_alpha)
        ax.scatter(x[m], true_alpha[m], s=20, alpha=0.55,
                   color=MODEL_COLORS[model],
                   edgecolors="black", linewidths=0.3)
        if m.sum() >= 5 and x[m].std() > 0:
            slope, intercept = np.polyfit(x[m], true_alpha[m], 1)
            xs = np.linspace(x[m].min(), x[m].max(), 50)
            ax.plot(xs, slope * xs + intercept, color="#C44E52", lw=1.5, ls="--")
        r = results[name]["r"]
        ci = results[name]["ci"]
        ax.set_title(f"{name}\nr={r:+.3f} [{ci[0]:+.2f}, {ci[1]:+.2f}]",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("predictor", fontsize=9)
        ax.set_ylabel("true α", fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Headline bar in last row spanning all columns
    ax = fig.add_subplot(gs[3, :])
    names   = [b[0] for b in BLOCKS]
    models  = [b[1] for b in BLOCKS]
    rs      = [results[n]["r"] for n in names]
    lo_err  = [abs(results[n]["r"] - results[n]["ci"][0]) for n in names]
    hi_err  = [abs(results[n]["ci"][1] - results[n]["r"]) for n in names]
    colors  = [MODEL_COLORS[m] for m in models]
    xs      = np.arange(len(names))
    ax.bar(xs, rs, yerr=[lo_err, hi_err], capsize=4,
           color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=10, rotation=20, ha="right")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_ylabel("Pearson r vs true α", fontsize=11)
    ax.set_title("Headline — does marginalising over envs (R3) "
                 "improve recovery of true α?",
                 fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Synthetic uniform-α dataset {args.dataset_id} — "
        f"three-regression test vs true α  (n={n_sessions} sessions, "
        f"{args.n_rng_seeds} RNG / {args.n_marg_envs} marg-env seeds)",
        fontsize=12, fontweight="bold", y=0.998)
    out_png = os.path.join(PLOT_DIR, "step1_three_regressions.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_png}")
    print("\nDone.")


if __name__ == "__main__":
    main()
