"""analyze_dezfouli_three_regressions.py — panel-e for Dezfouli.

Mirrors analyze_synthetic_three_regressions.py but for the Dezfouli 2-armed
bandit (12 blocks × variable trials per block, 2-dim [prev_c, prev_r] inputs,
categorical 3-way `diag` target instead of continuous α).

Per-subject statistics:
  R1 — observed expected per-trial regret (inferred (good_arm, good_prob)
       per block; regret_t = good_prob - obs_r_t).
  R2 — exact-env rollout regret (fresh both-arm draws using each subject's
       inferred per-block (good_arm, good_prob); rollout policy chooses).
  R3 — marginalised-env rollout regret (envs sampled from the canonical
       Dezfouli generator via simulate_dezfouli_bandit; rollout policy
       chooses).

Models compared: IDRNN (z), Vanilla (no init), Vanilla+h_avg, Vanilla+h_last.

Inputs:
  final_plots/dezfouli{tag}/canonical/{idrnn,vanilla}/latents_*_canonical.pt
  data_dezfouli/{xin,c,choice_one_hot}_train.npy + df_train.csv (subid, diag)

Outputs:
  final_plots/dezfouli{tag}/three_regressions_dezfouli.npz
    Per-subject R1/R2/R3 for each model variant + diag labels.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from modelsandtraining import Decoder, AblatedRNN, IDRNN, LatentRNN_secondstep
from nested_cv.config import path_tag
from simulate_dezfouli_bandit import (
    simulate_dezfouli_bandit, N_BLOCKS, GOOD_PROBS, BAD_PROB,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DGP = "dezfouli"
DATA_DIR = "data_dezfouli"
A = 2

GOOD_PROB_CANDIDATES = np.array(GOOD_PROBS)   # (0.25, 0.125, 0.08)


# ── Per-subject env from raw data/for_plos.csv ─────────────────────────────
# best_action (TRUE/FALSE) directly identifies the good arm at every trial —
# no inference from arm-mean comparison needed. The good arm's *probability*
# is one of {0.25, 0.125, 0.08} from the spec; we snap the empirical mean
# reward on the good arm to the nearest candidate.
def per_subject_env_from_raw(train_subids):
    """For each train subject (by integer subid in `train_subids`), extract:
        block_lengths: (N, 12)            — trials per block
        good_arm:      (N, 12)  in {0,1}  — directly from `best_action`
        good_prob:     (N, 12)            — snapped to {0.25, 0.125, 0.08}
        best_action_seq: list[list[ndarray of bool]] — per (subject, block) the
                                            per-trial best_action sequence
                                            (used for R1).
    `train_subids` must be the integer subids in the same order as the rows
    of xin_train.npy (i.e. the order load_dezfouli.py wrote to disk).
    """
    df = pd.read_csv("data/for_plos.csv")
    df["choice_binary"] = df["key"].map({"R1": 0, "R2": 1}).astype(int)
    df["subid"] = df["ID"].astype("category").cat.codes
    # best_action can be a string "TRUE"/"FALSE" or already bool — normalise.
    if df["best_action"].dtype == bool:
        df["ba"] = df["best_action"]
    else:
        df["ba"] = df["best_action"].astype(str).str.upper().eq("TRUE")

    N = len(train_subids)
    block_lengths = np.zeros((N, N_BLOCKS), dtype=int)
    good_arm      = np.zeros((N, N_BLOCKS), dtype=int)
    good_prob     = np.zeros((N, N_BLOCKS), dtype=float)
    best_action_seq = [[None]*N_BLOCKS for _ in range(N)]

    for i, sid in enumerate(train_subids):
        sub_df = df[df["subid"] == int(sid)]
        for b_raw in range(1, N_BLOCKS + 1):
            b = b_raw - 1   # 0-indexed block array
            blk = sub_df[sub_df["block"] == b_raw]
            L = len(blk)
            block_lengths[i, b] = L
            if L == 0:
                good_arm[i, b]  = 0
                good_prob[i, b] = GOOD_PROBS[-1]
                best_action_seq[i][b] = np.zeros(0, dtype=bool)
                continue
            ba = blk["ba"].values.astype(bool)
            ch = blk["choice_binary"].values.astype(int)
            rw = blk["reward"].values.astype(float)
            # good_arm: arm chosen when best_action=TRUE (constant per block
            # since schedule is fixed). If best_action is all FALSE in this
            # block, fall back to the arm chosen when best_action=FALSE
            # (which equals NOT good_arm → invert).
            if ba.any():
                good_arm[i, b] = int(np.bincount(ch[ba], minlength=2).argmax())
            elif (~ba).any():
                # Any pick is wrong → good arm is the other one
                good_arm[i, b] = int(1 - np.bincount(ch[~ba], minlength=2).argmax())
            else:
                good_arm[i, b] = 0   # degenerate, shouldn't hit
            # good_prob: mean reward on the good arm in this block, snapped
            good_mask = (ch == good_arm[i, b])
            if good_mask.any():
                empirical = float(rw[good_mask].mean())
            else:
                empirical = GOOD_PROBS[-1]
            good_prob[i, b] = float(GOOD_PROB_CANDIDATES[
                np.argmin(np.abs(GOOD_PROB_CANDIDATES - empirical))
            ])
            best_action_seq[i][b] = ba
    return block_lengths, good_arm, good_prob, best_action_seq


# ── R1: expected per-trial regret using best_action + known generative model
def observed_expected_regret(best_action_seq, block_lengths, good_prob):
    """For each valid trial:
        E[regret_t | chose good] = good_prob_b - good_prob_b = 0
        E[regret_t | chose bad ] = good_prob_b - 0.05
    Equivalently: regret_t = (good_prob_b - 0.05) * I[best_action_t == FALSE].
    Returns per-subject mean over all valid trials."""
    N = len(best_action_seq)
    R1 = np.zeros(N, dtype=float)
    for i in range(N):
        regrets = []
        for b in range(N_BLOCKS):
            ba = best_action_seq[i][b]
            if ba is None or len(ba) == 0:
                continue
            chose_bad = ~ba
            block_regret = chose_bad.astype(float) * (good_prob[i, b] - BAD_PROB)
            regrets.append(block_regret)
        if regrets:
            R1[i] = float(np.concatenate(regrets).mean())
        else:
            R1[i] = np.nan
    return R1


# ── Subject-specific exact env (R2): both-arm rewards drawn fresh from the
#    subject's inferred per-block schedule. Block lengths match the subject. ─
def sample_exact_env_subject(seed, block_lengths, good_arm, good_prob):
    rng = np.random.default_rng(seed)
    max_T = int(block_lengths.max())
    rewards = np.full((N_BLOCKS, max_T, A), -100.0, dtype=np.float32)
    for b in range(N_BLOCKS):
        T = int(block_lengths[b]); ga = good_arm[b]; gp = good_prob[b]
        probs = [BAD_PROB, BAD_PROB]; probs[ga] = gp
        rewards[b, :T, 0] = (rng.random(T) < probs[0]).astype(np.float32)
        rewards[b, :T, 1] = (rng.random(T) < probs[1]).astype(np.float32)
    return rewards


# ── Rollouts ────────────────────────────────────────────────────────────────
def encode_prev_dezfouli(prev_c, prev_r):
    return np.array([float(prev_c), float(prev_r)], dtype=np.float32)


@torch.no_grad()
def rollout_idrnn_batch(z_vec, envs_4d, decoder, device, block_lengths,
                         torch_gen=None):
    """Batched IDRNN rollout — runs B independent rollouts in parallel.

    envs_4d: (B, 12, max_T, 2) — B distinct envs (or B copies of one env).
    z_vec:   (z_dim,)            — single subject's z, broadcast across batch.
    block_lengths: (12,)         — same for all B (subject-specific).
    Returns dict with mean per-trial regret AND mean per-trial perseverance
    (P(c_t == c_{t-1}) within block) per rollout: each value ndarray (B,)."""
    B = envs_4d.shape[0]
    envs = torch.as_tensor(envs_4d, dtype=torch.float32, device=device)
    z = torch.as_tensor(z_vec, dtype=torch.float32, device=device)
    z = z.unsqueeze(0).expand(B, -1).contiguous()    # (B, z_dim)

    if torch_gen is None:
        torch_gen = torch.Generator(device=device)

    cum_regret = torch.zeros(B, device=device)
    cum_stay   = torch.zeros(B, device=device)
    n_valid    = 0
    n_stay_opportunities = 0
    for b in range(N_BLOCKS):
        T = int(block_lengths[b])
        if T == 0: continue
        h = decoder.z2h0(z).unsqueeze(0)             # (1, B, hidden) — block reset
        prev_x = torch.zeros(B, 1, 2, device=device) # (B, 1, in_dim)
        env_block = envs[:, b, :T, :]                # (B, T, 2)
        prev_arm = None
        for t in range(T):
            logits, h = decoder(prev_x, z, hidden=h)     # logits (B, 1, A)
            p = F.softmax(logits[:, 0, :], dim=-1)        # (B, A)
            arm = torch.multinomial(p, num_samples=1, generator=torch_gen).squeeze(-1)  # (B,)
            r = env_block[torch.arange(B, device=device), t, arm]
            r_max = env_block[:, t, :].max(dim=-1).values
            cum_regret = cum_regret + (r_max - r)
            if prev_arm is not None:
                cum_stay = cum_stay + (arm == prev_arm).float()
                n_stay_opportunities += 1
            prev_arm = arm
            prev_x = torch.stack([arm.float(), r.round()], dim=-1).unsqueeze(1)
        n_valid += T
    return {
        "regret":      (cum_regret / max(n_valid, 1)).cpu().numpy(),
        "perseverance": (cum_stay   / max(n_stay_opportunities, 1)).cpu().numpy(),
    }


@torch.no_grad()
def rollout_vanilla_batch(envs_4d, h_init_batch, vmodel, device, hidden_size,
                            block_lengths, torch_gen=None):
    """Batched Vanilla rollout. h_init_batch:
       None                   → zero-init each block
       (B, hidden) tensor/np  → reset to this h at every block start
       For Dezfouli (12 blocks) we reset h to the supplied prior at each
       block boundary — the +h interpretation is 'subject prior at episode
       boundary'.
    Returns mean per-trial regret per rollout: ndarray (B,).
    """
    B = envs_4d.shape[0]
    envs = torch.as_tensor(envs_4d, dtype=torch.float32, device=device)

    if h_init_batch is None:
        h_init_b = torch.zeros(1, B, hidden_size, device=device)
    else:
        hi = torch.as_tensor(h_init_batch, dtype=torch.float32, device=device)
        if hi.dim() == 1:
            hi = hi.unsqueeze(0).expand(B, -1).contiguous()
        h_init_b = hi.unsqueeze(0)                   # (1, B, hidden)

    if torch_gen is None:
        torch_gen = torch.Generator(device=device)

    cum_regret = torch.zeros(B, device=device)
    cum_stay   = torch.zeros(B, device=device)
    n_valid = 0
    n_stay_opportunities = 0
    for b in range(N_BLOCKS):
        T = int(block_lengths[b])
        if T == 0: continue
        h = h_init_b.clone()                          # block reset
        prev_x = torch.zeros(B, 1, 2, device=device)
        env_block = envs[:, b, :T, :]
        prev_arm = None
        for t in range(T):
            logits, h, _ = vmodel.dec(prev_x, h0=h)
            p = F.softmax(logits[:, 0, :], dim=-1)
            arm = torch.multinomial(p, num_samples=1, generator=torch_gen).squeeze(-1)
            r = env_block[torch.arange(B, device=device), t, arm]
            r_max = env_block[:, t, :].max(dim=-1).values
            cum_regret = cum_regret + (r_max - r)
            if prev_arm is not None:
                cum_stay = cum_stay + (arm == prev_arm).float()
                n_stay_opportunities += 1
            prev_arm = arm
            prev_x = torch.stack([arm.float(), r.round()], dim=-1).unsqueeze(1)
        n_valid += T
    return {
        "regret":       (cum_regret / max(n_valid, 1)).cpu().numpy(),
        "perseverance": (cum_stay   / max(n_stay_opportunities, 1)).cpu().numpy(),
    }


# ── PC1-match Vanilla h to IDRNN z_dim ──────────────────────────────────────
def pc1_match(h_raw, z_dim):
    """PCA-rank-z reconstruction of Vanilla h, so h_init has the same
    effective dim as IDRNN z. Identity if h.shape[1] <= z_dim."""
    from sklearn.decomposition import PCA
    if h_raw.shape[1] <= z_dim:
        return h_raw.astype(np.float32)
    p = PCA(n_components=z_dim)
    h_low = p.inverse_transform(p.fit_transform(h_raw))
    return h_low.astype(np.float32)


# ── Model loading from canonical .pt bundles ───────────────────────────────
def _load_idrnn(pt_path, pre_step25_ckpt=None):
    """Load the canonical IDRNN bundle and rebuild its decoder. If
    `pre_step25_ckpt` is given, override the decoder weights from that
    checkpoint — this is the right pairing for step-1 lookup z's, since the
    decoder at end-of-step-2 was last trained against varied lookup z (step
    2.5 then adapted both encoder + decoder to near-constant encoder z, so
    using the post-step-2.5 decoder with step-1 z is an OOD pairing)."""
    bundle = torch.load(pt_path, map_location=DEVICE, weights_only=False)
    z_dim   = bundle["z_dim"]
    hidden  = bundle["hidden"]
    in_dim  = bundle["base_in_dim"]
    A_ = bundle["A"]
    dec = Decoder(in_dim=in_dim, z_dim=z_dim, hid=hidden, A=A_).to(DEVICE)
    if pre_step25_ckpt is not None and os.path.exists(pre_step25_ckpt):
        state = torch.load(pre_step25_ckpt, map_location=DEVICE,
                            weights_only=False)
        src = "pre-step-2.5 checkpoint"
    else:
        state = bundle["model_state"]
        src = "post-step-2.5 canonical bundle"
    dec_state = {k.replace("decoder.", "", 1): v
                 for k, v in state.items()
                 if k.startswith("decoder.")}
    dec.load_state_dict(dec_state, strict=False)
    dec.eval()
    print(f"  IDRNN decoder loaded from {src}  ({len(dec_state)} tensors)")
    return dec, bundle["z_train"], bundle.get("z_test"), z_dim, hidden


def _load_vanilla(pt_path):
    bundle = torch.load(pt_path, map_location=DEVICE, weights_only=False)
    hidden = bundle["hidden"]
    in_dim = bundle["base_in_dim"]
    A_ = bundle["A"]
    vmodel = AblatedRNN(hid=hidden, in_dim=in_dim, A=A_, block_structure=True).to(DEVICE)
    vmodel.load_state_dict(bundle["model_state"], strict=False)
    vmodel.eval()
    return (vmodel, bundle["h_avg_train"], bundle.get("h_avg_test"),
            bundle["h_last_train"], bundle.get("h_last_test"), hidden)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis_z", type=int, default=2)
    ap.add_argument("--metric", default="step1_specificity")
    ap.add_argument("--n_rng_seeds", type=int, default=100,
                    help="RNG seeds per subject for exact-env (R2) rollouts.")
    ap.add_argument("--n_marg_envs", type=int, default=100,
                    help="Fresh env draws per subject for marginal (R3) rollouts.")
    args = ap.parse_args()

    # ── Locate canonical .pt bundles (BOTH archs live under the {z,metric}
    #    -tagged path on Dezfouli) ──
    tag = path_tag(args.hypothesis_z, args.metric)
    idr_root  = f"final_plots/{DGP}{tag}/canonical/idrnn"
    van_root  = f"final_plots/{DGP}{tag}/canonical/vanilla"
    out_dir   = f"final_plots/{DGP}{tag}"
    os.makedirs(out_dir, exist_ok=True)

    idr_pt = os.path.join(idr_root, "latents_idrnn_canonical.pt")
    van_pt = os.path.join(van_root, "latents_vanilla_canonical.pt")
    if not os.path.exists(idr_pt) or not os.path.exists(van_pt):
        raise FileNotFoundError(
            f"Missing canonical .pt bundles:\n  idrnn: {idr_pt}\n  vanilla: {van_pt}\n"
            "Run extract_canonical_latents_dezfouli.py first."
        )

    # Locate the seed dir to grab step1_z_lookup + pre-step-2.5 decoder ckpt
    choice_path = os.path.join(idr_root, "..", "idrnn_choice.json")
    if os.path.exists(choice_path):
        choice = json.load(open(choice_path))
        seed = choice.get("selected_seed")
        canonical_runs_dir = choice.get("canonical_runs_dir") \
            or f"{idr_root}/runs"
        seed_dir = os.path.join(canonical_runs_dir, f"seed_{seed}")
    else:
        seed_dir = None
    step1_p = (os.path.join(seed_dir, "step1_z_lookup.npy")
               if seed_dir else None)
    pre_step25_ckpt = None
    if seed_dir:
        ckpts = [f for f in os.listdir(os.path.join(seed_dir, "checkpoints"))
                 if f.endswith("_pre_step25.pt")]
        if ckpts:
            pre_step25_ckpt = os.path.join(seed_dir, "checkpoints",
                                            sorted(ckpts)[-1])

    decoder, z_train_step2, _, z_dim, idr_hidden = _load_idrnn(
        idr_pt, pre_step25_ckpt=pre_step25_ckpt
    )
    (vmodel, h_avg_train, _,
     h_last_train, _, van_hidden) = _load_vanilla(van_pt)

    if step1_p is None or not os.path.exists(step1_p):
        raise FileNotFoundError(
            f"Step-1 z lookup not found at {step1_p}; "
            "cannot run rollouts with step-1 latents."
        )
    z_train = np.load(step1_p).astype(np.float32)
    print(f"  Loaded step-1 IDRNN z: {z_train.shape}  (vs step-2 encoder "
          f"shape {z_train_step2.shape})  source: {step1_p}")

    # ── Behavioural data (train cohort) + per-subject env from raw csv ────
    df_train  = pd.read_csv(f"{DATA_DIR}/df_train.csv")
    train_subids = df_train["subid"].astype(int).values
    N = len(train_subids)
    print(f"  Loaded {N} train subjects from df_train.csv")

    block_lengths, good_arm, good_prob, best_action_seq = per_subject_env_from_raw(
        train_subids
    )
    print(f"  Per-subject env (best_action-derived): block_lengths range "
          f"[{block_lengths.min()}, {block_lengths.max()}]; "
          f"good_prob counts → "
          f"{dict(zip(*np.unique(good_prob, return_counts=True)))}")

    # ── R1: expected per-trial regret from best_action + known generative model
    R1_regret = observed_expected_regret(best_action_seq, block_lengths, good_prob)
    print(f"  R1 (expected regret from best_action): mean={np.nanmean(R1_regret):.4f}, "
          f"sd={np.nanstd(R1_regret):.4f}")

    # ── R1: observed perseverance P(c_t == c_{t-1}) within blocks ──────────
    df_raw = pd.read_csv("data/for_plos.csv")
    df_raw["choice_binary"] = df_raw["key"].map({"R1": 0, "R2": 1}).astype(int)
    df_raw["subid"] = df_raw["ID"].astype("category").cat.codes
    R1_persev = np.full(N, np.nan, dtype=np.float32)
    for i, sid in enumerate(train_subids):
        sub = df_raw[df_raw["subid"] == int(sid)]
        stays = 0; opps = 0
        for b in range(1, N_BLOCKS + 1):
            ch = sub[sub["block"] == b]["choice_binary"].values
            if len(ch) < 2: continue
            stays += int((ch[1:] == ch[:-1]).sum())
            opps  += int(len(ch) - 1)
        R1_persev[i] = stays / max(opps, 1)
    print(f"  R1 (observed perseverance): mean={np.nanmean(R1_persev):.4f}, "
          f"sd={np.nanstd(R1_persev):.4f}")

    # ── PC1-match h_avg, h_last to z_dim ───────────────────────────────────
    h_avg_pc  = pc1_match(h_avg_train,  z_dim)
    h_last_pc = pc1_match(h_last_train, z_dim) if h_last_train is not None else None

    # ── Per-subject rollouts (batched). Each rollout returns BOTH regret
    #    and perseverance (P(c_t==c_{t-1}) within block). ─────────────────
    KEYS = ("idrnn", "van0", "vanH_avg", "vanH_last")
    METRICS = ("regret", "perseverance")
    R2 = {(m, k): np.full(N, np.nan, dtype=np.float32) for m in METRICS for k in KEYS}
    R3 = {(m, k): np.full(N, np.nan, dtype=np.float32) for m in METRICS for k in KEYS}

    K = args.n_rng_seeds
    M = args.n_marg_envs

    import time
    t0 = time.time()
    for i in range(N):
        bl_i = block_lengths[i]
        ga_i = good_arm[i]; gp_i = good_prob[i]
        max_T_i = int(bl_i.max())

        # ── R2: K rollouts of the SAME exact env (RNG variation only) ─────
        env_exact = sample_exact_env_subject(
            seed=10_000_000 * i, block_lengths=bl_i, good_arm=ga_i, good_prob=gp_i
        )
        envs_R2 = np.broadcast_to(env_exact[None], (K, *env_exact.shape)).copy()

        gen = torch.Generator(device=DEVICE)
        for tag, kwargs, seed_off in [
            ("idrnn",     {"call": "idrnn", "z_vec": z_train[i]},                    101),
            ("van0",      {"call": "vanilla", "h_init": None},                       202),
            ("vanH_avg",  {"call": "vanilla", "h_init": h_avg_pc[i]},                303),
            ("vanH_last", {"call": "vanilla", "h_init": h_last_pc[i] if h_last_pc is not None else None}, 404),
        ]:
            if kwargs.get("h_init") is None and tag == "vanH_last":
                continue
            gen.manual_seed(seed_off + i)
            if kwargs["call"] == "idrnn":
                out = rollout_idrnn_batch(kwargs["z_vec"], envs_R2, decoder,
                                            DEVICE, bl_i, gen)
            else:
                out = rollout_vanilla_batch(envs_R2, kwargs["h_init"], vmodel,
                                              DEVICE, van_hidden, bl_i, gen)
            R2[("regret",       tag)][i] = out["regret"].mean()
            R2[("perseverance", tag)][i] = out["perseverance"].mean()

        # ── R3: M distinct marginal envs (one rollout per env) ─────────────
        envs_R3 = np.zeros((M, N_BLOCKS, max_T_i, A), dtype=np.float32)
        for k in range(M):
            envs_R3[k], _, _ = simulate_dezfouli_bandit(
                seed=20_000_000 * i + k, block_lengths=bl_i
            )
        for tag, kwargs, seed_off in [
            ("idrnn",     {"call": "idrnn", "z_vec": z_train[i]},                    505),
            ("van0",      {"call": "vanilla", "h_init": None},                       606),
            ("vanH_avg",  {"call": "vanilla", "h_init": h_avg_pc[i]},                707),
            ("vanH_last", {"call": "vanilla", "h_init": h_last_pc[i] if h_last_pc is not None else None}, 808),
        ]:
            if kwargs.get("h_init") is None and tag == "vanH_last":
                continue
            gen.manual_seed(seed_off + i)
            if kwargs["call"] == "idrnn":
                out = rollout_idrnn_batch(kwargs["z_vec"], envs_R3, decoder,
                                            DEVICE, bl_i, gen)
            else:
                out = rollout_vanilla_batch(envs_R3, kwargs["h_init"], vmodel,
                                              DEVICE, van_hidden, bl_i, gen)
            R3[("regret",       tag)][i] = out["regret"].mean()
            R3[("perseverance", tag)][i] = out["perseverance"].mean()

        if (i + 1) % 10 == 0 or i == N - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N - i - 1)
            print(f"  [{i+1}/{N}] R3 idrnn regret={R3[('regret','idrnn')][i]:.3f} "
                  f"persev={R3[('perseverance','idrnn')][i]:.3f}  "
                  f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

    diag_labels = df_train["diag"].astype(str).values

    out_path = os.path.join(out_dir, "three_regressions_dezfouli.npz")
    save_kwargs = {
        "R1":         R1_regret,    # back-compat: R1 == regret
        "R1_regret":  R1_regret,
        "R1_perseverance": R1_persev,
        "diag":             diag_labels,
        "block_lengths":    block_lengths,
        "good_arm":         good_arm,
        "good_prob":        good_prob,
    }
    NAME_MAP = {"idrnn":"idrnn", "van0":"van0", "vanH_avg":"vanH", "vanH_last":"vanH_lastT"}
    for metric in METRICS:
        for tag in KEYS:
            short = NAME_MAP[tag]
            tag2 = f"R2_{short}" if metric == "regret" else f"R2_{short}_persev"
            tag3 = f"R3_{short}" if metric == "regret" else f"R3_{short}_persev"
            save_kwargs[tag2] = R2[(metric, tag)]
            save_kwargs[tag3] = R3[(metric, tag)]
    np.savez(out_path, **save_kwargs)
    print(f"\nSaved -> {out_path}")
    print("  npz keys:", sorted(save_kwargs.keys()))


if __name__ == "__main__":
    main()
