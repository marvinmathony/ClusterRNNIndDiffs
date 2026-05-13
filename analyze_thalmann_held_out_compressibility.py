#!/usr/bin/env python3
"""
Held-out compressibility test for IDRNN step-1 lookup model.

Uses the hp_v2 3-fold CV structure
(runs_thalmann_hp_v2_{COMBO}/fold{0,1,2}/seed_X/frozen_decoder/policy_model.pt).
Per fold, the model is trained on a subset of participants only — test
participants have *no* learned lookup row.

Procedure:
  1. Load each fold's frozen step-1 decoder.
  2. For each held-out test subject in that fold, INFER their z by gradient
     descent: freeze decoder, initialize z, run their actual training-task
     behaviour through the model, compute NLL on choices, backprop to z, update.
  3. With the inferred z, run on-policy rollouts on the restless-bandit
     simulator and compute LZW compressibility score = b_LZW / l_LZW.
  4. Pool held-out subjects across folds → one score per participant.
  5. Compare with human compressibility and questionnaire scales.

Outputs (plots_thalmann/):
  step1_held_out_compressibility.npz       — held-out comp scores + subids
  step1_held_out_compressibility.png       — humans vs held-out IDRNN scatter
  step1_held_out_questionnaire_corr.png    — held-out vs all-train comp×scale
"""

import os, json, glob
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

import sys
sys.path.insert(0, ".")
from modelsandtraining import Decoder
from simulate_restless_bandit import simulate_restless_bandit, T as SIM_T, K as SIM_K

# ── Config ─────────────────────────────────────────────────────────────────────
DGP    = "thalmann"
COMBO  = "uw00_lmbd01_eh5_h5_z5"
RUNS_DIR = f"runs_{DGP}_hp_v2_{COMBO}"
FOLDS  = [0, 1, 2]
PLOT_DIR = f"plots_{DGP}"
os.makedirs(PLOT_DIR, exist_ok=True)

A           = SIM_K
N_TRIALS    = SIM_T
N_ENV_SEEDS = 100   # rollouts per held-out subject
PCT         = 10
# Match training-time normalization (load_thalmann.py:42): r_input = r_raw / 100
REWARD_MAX  = 100.0

# Z-inference hyperparams
Z_INFER_EPOCHS = 400
Z_INFER_LR     = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

task_ids_global = torch.tensor(
    np.load(f"data_{DGP}/task_ids_per_block.npy"), dtype=torch.long
).to(DEVICE)
RESTLESS_TASK_ID = 1
assert task_ids_global.tolist().count(RESTLESS_TASK_ID) >= 1

COL_LOW   = "#4C72B0"
COL_HIGH  = "#C44E52"

# ── Questionnaire ──────────────────────────────────────────────────────────────
quest = pd.read_csv("data/finalQuestionnaireDataSession1.csv").set_index("ID")
SCALES = {
    "PANAS_PA":   [f"PANAS_{i}"  for i in [0,2,4,6,8,14,16,18]],
    "PANAS_NA":   [f"PANAS_{i}"  for i in [1,3,5,7,9,11,13,15]],
    "STICSA":     [f"STICSA_{i}" for i in range(22)],
    "PHQ":        [f"PHQ_9_{i}"  for i in range(10)],
    "CEI":        [f"CEI_{i}"    for i in range(4)],
    "BIG5_open":  [f"BIG_5_{i}"  for i in range(6)],
    "Motiv_slot": ["motiv_slot_0"],
    "Motiv_mem":  ["motiv_mem_0"],
}
for name, items in SCALES.items():
    quest[name] = quest[items].mean(axis=1)
scale_names = list(SCALES.keys())

# ── Helpers ────────────────────────────────────────────────────────────────────
def lzw_length(seq, alphabet_size):
    s = [int(x) for x in seq]
    table = {(c,): i for i, c in enumerate(range(alphabet_size))}
    next_code = alphabet_size
    out = 0
    w = ()
    for c in s:
        wc = w + (c,)
        if wc in table:
            w = wc
        else:
            out += 1
            table[wc] = next_code
            next_code += 1
            w = (c,)
    if w:
        out += 1
    return out

def baseline_lzw(length, alphabet_size, n_samples=500, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.mean([
        lzw_length(rng.integers(0, alphabet_size, size=length), alphabet_size)
        for _ in range(n_samples)
    ]))

@lru_cache(maxsize=None)
def make_env(seed):
    """Return continuous rewards (T, K) in raw scale; rollouts feed
    reward / REWARD_MAX into the model input to match training format."""
    return simulate_restless_bandit(seed)["rewards"].astype(np.float32)

# ── Load fold model + infer z for test subjects ───────────────────────────────
def load_fold_decoder(fold):
    """Return (decoder, task_emb_w, mc, train_z_mean, base_in_dim)."""
    seed_dirs = sorted(d for d in os.listdir(f"{RUNS_DIR}/fold{fold}")
                       if d.startswith("seed_"))
    sd0 = seed_dirs[0]
    fpath = f"{RUNS_DIR}/fold{fold}/{sd0}/frozen_decoder/policy_model.pt"
    state = torch.load(fpath, map_location=DEVICE)
    with open(f"{RUNS_DIR}/fold{fold}/{sd0}/config.json") as f:
        cfg = json.load(f)
    mc = cfg["model_config"]
    decoder = Decoder(in_dim=mc["dec_in_dim"], z_dim=mc["z_dim"],
                      hid=mc["hidden"], A=mc["A"]).to(DEVICE)
    decoder.load_state_dict(
        {k[8:]: v for k, v in state.items() if k.startswith("decoder.")})
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    task_emb_w = state["task_embedding.weight"].to(DEVICE)
    train_z = state["encoder.embed.weight"].to(DEVICE)  # (n_train, z_dim)
    base_in_dim = mc["dec_in_dim"] - mc.get("task_emb_dim", 0)
    return decoder, task_emb_w, mc, train_z, base_in_dim

def infer_z_for_subject(decoder, task_emb_w, mc, base_in_dim,
                        xin_subj, c_subj, init_z=None,
                        epochs=Z_INFER_EPOCHS, lr=Z_INFER_LR):
    """Fit a single z (frozen decoder) to one subject's full 31-block behaviour.

    xin_subj: (31, 200, base_in_dim), c_subj: (31, 200)
    Returns: numpy array of inferred z (z_dim,) and final NLL.
    """
    z_dim = mc["z_dim"]
    A_int = mc["A"]
    n_blocks, T, _ = xin_subj.shape

    z = (torch.zeros(z_dim, device=DEVICE) if init_z is None
         else torch.as_tensor(init_z, dtype=torch.float32, device=DEVICE).clone())
    z = z.requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)

    xin_t = torch.as_tensor(xin_subj, dtype=torch.float32, device=DEVICE)   # (B, T, in)
    c_t   = torch.as_tensor(c_subj, dtype=torch.long,   device=DEVICE)      # (B, T)
    valid = (c_t >= 0).float()                                              # (B, T)

    # Append per-block task embedding
    def block_input(z_vec):
        # Build (n_blocks, T, dec_in) by concatenating task embedding
        z_b = z_vec.unsqueeze(0).unsqueeze(0)  # (1, 1, z)
        out_blocks = []
        for b in range(n_blocks):
            tid  = task_ids_global[b]
            temb = task_emb_w[tid].unsqueeze(0).unsqueeze(0).expand(1, T, -1)
            xb   = xin_t[b].unsqueeze(0)                       # (1, T, in)
            out_blocks.append(torch.cat([xb, temb], dim=-1))   # (1, T, dec_in)
        return torch.cat(out_blocks, dim=0)                    # (B, T, dec_in)

    last_nll = float("inf")
    for ep in range(epochs):
        opt.zero_grad()
        seq = block_input(z)                                   # (B, T, dec_in)
        # decoder.forward expects (B, T, in_dim); broadcast z across blocks
        z_exp = z.unsqueeze(0).expand(n_blocks, -1)            # (B, z_dim)
        logits, _ = decoder(seq, z_exp)                        # (B, T, A)
        log_p = F.log_softmax(logits, dim=-1)
        chosen = log_p.gather(-1, c_t.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        nll = -(chosen * valid).sum() / valid.sum().clamp(min=1)
        nll.backward()
        opt.step()
        last_nll = nll.item()
    return z.detach().cpu().numpy(), last_nll

# ── On-policy rollout (same logic as analyze_thalmann_logit_updates.py) ──────
@torch.no_grad()
def rollout_choices(decoder, task_emb_w, base_in_dim, z_vec,
                     reward_schedule, task_id, rng):
    """reward_schedule: (T, K) continuous raw rewards. Input feature receives
    reward / REWARD_MAX, matching training format."""
    T = reward_schedule.shape[0]
    z_t  = torch.as_tensor(z_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    temb = task_emb_w[task_id].detach().cpu().numpy()
    h    = decoder.z2h0(z_t).unsqueeze(0)
    prev = np.concatenate([np.zeros(base_in_dim, dtype=np.float32), temb])
    choices = []
    for t in range(T):
        x = torch.as_tensor(prev, dtype=torch.float32,
                            device=DEVICE).unsqueeze(0).unsqueeze(0)
        logits, h = decoder(x, z_t, hidden=h)
        p = F.softmax(logits[0, 0], dim=-1).detach().cpu().numpy()
        arm        = rng.choice(A, p=p)
        reward_raw = float(reward_schedule[t, arm])
        oh = np.zeros(A, dtype=np.float32); oh[arm] = 1.0
        prev = np.concatenate([oh, [reward_raw / REWARD_MAX], temb])
        choices.append(arm)
    return np.array(choices)

# ── Main per-fold loop ────────────────────────────────────────────────────────
b_sim = baseline_lzw(N_TRIALS, A, n_samples=500, seed=0)
print(f"Baseline LZW length (random, T={N_TRIALS}, K={A}): {b_sim:.2f}")

held_out_records = []   # one row per held-out subject (across folds)
for fold in FOLDS:
    print(f"\n══ Fold {fold} ══")
    decoder, task_emb_w, mc, train_z, base_in_dim = load_fold_decoder(fold)
    z_dim = mc["z_dim"]
    train_z_mean = train_z.mean(0).detach().cpu().numpy()

    df_test  = pd.read_csv(f"data_{DGP}/fold{fold}/df_test.csv")
    xin_test = np.load(f"data_{DGP}/fold{fold}/xin_test.npy")  # (N_te, 31, T, in)
    c_test   = np.load(f"data_{DGP}/fold{fold}/c_test.npy")    # (N_te, 31, T)
    test_subids = df_test["subid"].values
    print(f"  n_test={len(test_subids)}, z_dim={z_dim}, "
          f"base_in_dim={base_in_dim}")

    for i_te, subid in enumerate(test_subids):
        # 1) Infer z by gradient descent on actual behaviour
        z_inferred, nll = infer_z_for_subject(
            decoder, task_emb_w, mc, base_in_dim,
            xin_test[i_te], c_test[i_te], init_z=train_z_mean,
        )

        # 2) On-policy rollouts in the restless bandit, compressibility
        rollout_lens = []
        for env_seed in range(N_ENV_SEEDS):
            sched = make_env(env_seed)
            rng   = np.random.default_rng(env_seed * 10000 + int(subid))
            ch = rollout_choices(decoder, task_emb_w, base_in_dim,
                                 z_inferred, sched, RESTLESS_TASK_ID, rng)
            rollout_lens.append(lzw_length(ch, A))
        comp_score = b_sim / float(np.mean(rollout_lens))

        held_out_records.append({
            "fold": fold, "subid": int(subid),
            "z_inferred": z_inferred.tolist(),
            "infer_nll": float(nll),
            "comp_score": comp_score,
        })
        if (i_te + 1) % 20 == 0:
            print(f"  fold {fold}: {i_te+1}/{len(test_subids)} done")
    print(f"  fold {fold} complete.")

ho_df = pd.DataFrame(held_out_records)

# Save raw (one row per (fold, subid))
np_path = os.path.join(PLOT_DIR, "step1_held_out_compressibility.npz")
np.savez(np_path,
         subids=ho_df["subid"].values,
         comp_scores=ho_df["comp_score"].values,
         folds=ho_df["fold"].values,
         z_inferred=np.stack(ho_df["z_inferred"].values))
print(f"\nSaved raw scores → {np_path}")
ho_df.drop(columns=["z_inferred"]).to_csv(
    os.path.join(PLOT_DIR, "step1_held_out_compressibility.csv"), index=False)

# ── Compare with human compressibility ───────────────────────────────────────
print("\nLoading human compressibility from restless data...")
raw_restless = pd.read_csv("data/finalRestlessSession1.csv").sort_values(
    ["ID", "trial"])
human_scores = {}
for sid, grp in raw_restless.groupby("ID"):
    ch = grp["chosen"].values.astype(int)
    if len(ch) < 50:
        continue
    l_h = lzw_length(ch, A)
    b_h = baseline_lzw(len(ch), A, n_samples=500, seed=hash(int(sid)) & 0xFFFF)
    human_scores[sid] = b_h / l_h

ho_df["human_comp"] = ho_df["subid"].map(human_scores)

# ── Correlations: held-out IDRNN compressibility × {human, questionnaire} ────
print("\nCorrelations (held-out IDRNN compressibility):")
mask = np.isfinite(ho_df["human_comp"].values) & np.isfinite(ho_df["comp_score"].values)
r_sp, p_sp = stats.spearmanr(ho_df.loc[mask, "human_comp"],
                              ho_df.loc[mask, "comp_score"])
r_pe, p_pe = stats.pearsonr(ho_df.loc[mask, "human_comp"],
                             ho_df.loc[mask, "comp_score"])
print(f"  vs human compressibility (n={mask.sum()}): "
      f"Spearman r={r_sp:+.3f} p={p_sp:.3e}  |  "
      f"Pearson r={r_pe:+.3f} p={p_pe:.3e}")

scale_records = []
for scale in scale_names:
    y = quest.reindex(ho_df["subid"].values)[scale].values.astype(float)
    m = np.isfinite(y) & np.isfinite(ho_df["comp_score"].values)
    if m.sum() < 10:
        continue
    rs, ps = stats.spearmanr(ho_df["comp_score"].values[m], y[m])
    rp, pp = stats.pearsonr(ho_df["comp_score"].values[m],  y[m])
    scale_records.append({"scale": scale, "n": int(m.sum()),
                          "r_sp": rs, "p_sp": ps,
                          "r_pe": rp, "p_pe": pp})
scale_df = pd.DataFrame(scale_records)
scale_df["p_sp_fdr"] = multipletests(scale_df["p_sp"], method="fdr_bh")[1]
scale_df["p_pe_fdr"] = multipletests(scale_df["p_pe"], method="fdr_bh")[1]
print("\nQuestionnaire correlations (held-out, FDR-BH):")
for _, row in scale_df.iterrows():
    s = ("***" if row["p_sp_fdr"] < 0.001
         else "**" if row["p_sp_fdr"] < 0.01
         else "*"  if row["p_sp_fdr"] < 0.05
         else "")
    print(f"  {row['scale']:<12}n={int(row['n']):3d}  "
          f"Spearman r={row['r_sp']:+.3f} p={row['p_sp']:.2e} "
          f"FDR={row['p_sp_fdr']:.2e}{s}  |  "
          f"Pearson r={row['r_pe']:+.3f} p={row['p_pe']:.2e}")
scale_df.to_csv(
    os.path.join(PLOT_DIR, "step1_held_out_questionnaire_corr.csv"),
    index=False,
)

# ── Plot: scatter (human vs held-out IDRNN comp) + bar (per-scale) ───────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel A: scatter
ax = axes[0]
m = np.isfinite(ho_df["human_comp"]) & np.isfinite(ho_df["comp_score"])
sub = ho_df[m]
sc = ax.scatter(sub["human_comp"], sub["comp_score"],
                c=sub["fold"], cmap="viridis", s=35, alpha=0.7,
                edgecolors="black", linewidths=0.3)
slope, intercept, *_ = stats.linregress(sub["human_comp"], sub["comp_score"])
xf = np.linspace(sub["human_comp"].min(), sub["human_comp"].max(), 100)
ax.plot(xf, slope*xf + intercept, color="black", lw=2, ls="--", label="OLS fit")
lo = min(sub["human_comp"].min(), sub["comp_score"].min())
hi = max(sub["human_comp"].max(), sub["comp_score"].max())
ax.plot([lo, hi], [lo, hi], color="grey", ls=":", lw=1, alpha=0.5,
        label="Identity")
plt.colorbar(sc, ax=ax, label="Fold")
ax.set_xlabel("Human compressibility (b_LZW / l_LZW)", fontsize=11)
ax.set_ylabel("Held-out IDRNN compressibility", fontsize=11)
ax.set_title(f"Humans vs held-out IDRNN  (n={m.sum()})\n"
             f"Spearman r={r_sp:+.3f} p={p_sp:.2e} | "
             f"Pearson r={r_pe:+.3f} p={p_pe:.2e}",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel B: per-scale Spearman r
ax = axes[1]
xs = np.arange(len(scale_df))
ax.bar(xs, scale_df["r_sp"].values, color="#4C72B0",
       edgecolor="black", linewidth=0.5)
for i, (rval, pf) in enumerate(zip(scale_df["r_sp"], scale_df["p_sp_fdr"])):
    s = ("***" if pf < 0.001 else "**" if pf < 0.01
         else "*" if pf < 0.05 else "")
    if s:
        ax.text(i, rval + 0.01 * np.sign(rval),
                s, ha="center",
                va="bottom" if rval >= 0 else "top",
                fontsize=10, fontweight="bold")
ax.axhline(0, color="grey", lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(scale_df["scale"].values, rotation=30, fontsize=9, ha="right")
ax.set_ylabel("Spearman r", fontsize=11)
ax.set_title("Held-out IDRNN comp × questionnaire scales\n"
             f"FDR-BH within {len(scale_df)} tests",
             fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle(
    f"Held-out compressibility test  ({COMBO}, {len(FOLDS)}-fold CV, "
    f"n={len(ho_df)} held-out subjects)",
    fontsize=12, fontweight="bold")
fig.tight_layout()
out = os.path.join(PLOT_DIR, "step1_held_out_compressibility.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {out}")

print("\nDone.")
