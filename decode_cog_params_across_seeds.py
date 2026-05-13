#!/usr/bin/env python3
"""
Decode cognitive model parameters (theta, w_value, w_uncert, w_lag, w_novelty)
from IDRNN / Vanilla latents across seeds, using LOOCV Ridge regression.

Epoch selection (in priority order):
  1. --epoch fixed value
  2. plots_sloutsky/analyze_across_seeds_summary{tag}.json  (key: "epochs")
  3. plots_sloutsky/decode_across_seeds_summary{tag}.json   (key: "best_epochs")
  4. Inline LOSO train-NLL (same as decode_across_seeds.py)

Diagnostic figures saved before regression:
  - dist_cog_params.png      : histograms of all 5 cog parameters
  - dist_latents.png         : per-dimension boxplots of IDRNN and Vanilla latents

Main results:
  - decode_cog_params_loso{tag}.png    : grouped bars (Pearson r per param)
  - novelty_scatter_loso{tag}.png      : LOOCV predicted vs true w_novelty per seed

Optional target transformation: --transform_targets {none, yeo-johnson}
  yeo-johnson applies PowerTransformer to each target independently.

Usage:
    python decode_cog_params_across_seeds.py
    python decode_cog_params_across_seeds.py --seeds 12,50,76
    python decode_cog_params_across_seeds.py --epoch 1500
    python decode_cog_params_across_seeds.py --transform_targets yeo-johnson
"""

import argparse
import os
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import make_pipeline

from compute_reconstruction_specificity import (
    load_model_config, create_model_from_config,
    load_model_checkpoint, compute_reconstruction_loss,
)
from sloutsky_cog_model import unpack_params

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=None,
                    help="Fixed epoch for all seeds (bypass epoch selection)")
parser.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds, e.g. '12,50,76,100,142'")
parser.add_argument("--transform_targets", choices=["none", "yeo-johnson"],
                    default="none",
                    help="Transform cog param targets before regression")
parser.add_argument("--n_pca", type=int, default=None,
                    help="Reduce latents to this many PCA components before Ridge "
                         "(recommended when latent dim > ~5). Default: use all dims.")
args = parser.parse_args()
FIXED_EPOCH       = args.epoch
FILTER_SEEDS      = (set(int(s) for s in args.seeds.split(",")) if args.seeds else None)
TRANSFORM_TARGETS = args.transform_targets
N_PCA             = args.n_pca

# ── Configuration ─────────────────────────────────────────────────────────────
DGP       = "sloutsky"
DATA_DIR  = f"data_{DGP}"
IDRNN_DIR = f"runs_{DGP}"
VAN_DIR   = f"runs_vanilla_{DGP}"
PLOT_DIR  = f"plots_{DGP}"
MIN_EPOCH = 100
MAX_EPOCH = 3000
os.makedirs(PLOT_DIR, exist_ok=True)

_seed_tag = ("_seeds" + "-".join(str(s) for s in sorted(FILTER_SEEDS))
             if FILTER_SEEDS else "")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load sequence data ────────────────────────────────────────────────────────
xin_test  = torch.from_numpy(np.load(f"{DATA_DIR}/xin_test.npy")).float().to(device)
xin_train = torch.from_numpy(np.load(f"{DATA_DIR}/xin_train.npy")).float().to(device)
c_train   = torch.from_numpy(np.load(f"{DATA_DIR}/c_train.npy")).float().to(device)

_enc_train_path = f"{DATA_DIR}/xin_enc_train.npy"
xin_enc_train   = (torch.from_numpy(np.load(_enc_train_path)).float().to(device)
                   if os.path.exists(_enc_train_path) else None)

B_test  = xin_test.shape[0]
B_train = xin_train.shape[0]

# ── Participant metadata ───────────────────────────────────────────────────────
df_test     = pd.read_csv(f"{DATA_DIR}/df_test.csv")
df_unique   = df_test.drop_duplicates("subid").sort_values("subid").reset_index(drop=True)
test_subids = sorted(df_test["subid"].unique())
assert len(test_subids) == B_test

GROUP_MAP    = {"young_child": 0, "old_child": 1, "adult": 2}
GROUP_ORDER  = ["young_child", "old_child", "adult"]
GROUP_LABELS = ["Young children", "Older children", "Adults"]
GROUP_COLORS = ["#5B9BD5", "#ED7D31", "#A5A5A5"]
age_labels   = np.array([df_unique.loc[df_unique["subid"] == s, "age"].values[0]
                          for s in test_subids])

# ── Cognitive model parameters ────────────────────────────────────────────────
em = np.load(f"{DATA_DIR}/em_results.npz", allow_pickle=True)
h_all_test        = em["h_all_test"]
participants_test = em["participants_test"]
assert list(participants_test) == test_subids

_cog = [unpack_params(h_all_test[i]) for i in range(B_test)]
COG_VARS = {
    "theta":    np.array([p["theta"]             for p in _cog]),
    "w_value":  np.array([p["b_value_train"]     for p in _cog]),
    "w_uncert": np.array([p["b_uncertain_train"] for p in _cog]),
    "w_lag":    np.array([p["b_lag_train"]       for p in _cog]),
    "w_novelty":np.array([p["b_novelty_train"]   for p in _cog]),
}
COG_LABELS = {
    "theta":    "θ",
    "w_value":  "w value",
    "w_uncert": "w uncert",
    "w_lag":    "w lag",
    "w_novelty":"w novelty",
}

# ── Epoch loading helpers ──────────────────────────────────────────────────────
def load_epochs_from_json(json_path, key):
    """Return {seed: epoch} from a summary JSON, or None if not usable."""
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        d = json.load(f)
    md = d.get(key, {})
    # analyze_across_seeds format: {"epochs": {"200": 1800, ...}}
    if "epochs" in md:
        return {int(k): v for k, v in md["epochs"].items()}
    # decode_across_seeds format: {"best_epochs": {"12": {"epoch": X}, ...}}
    if "best_epochs" in md:
        return {int(k): v["epoch"] for k, v in md["best_epochs"].items()
                if isinstance(v, dict) and "epoch" in v}
    return None


def get_epochs(base_dir, is_latent, key):
    """
    Load per-seed epochs in priority order:
      1. FIXED_EPOCH
      2. analyze_across_seeds_summary JSON
      3. decode_across_seeds_summary JSON
      4. decode_cog_params_summary JSON
      5. Inline LOSO
    Returns {seed: epoch}.
    """
    seeds = sorted([
        int(d.split("_")[1])
        for d in os.listdir(base_dir)
        if d.startswith("seed_")
        and (FILTER_SEEDS is None or int(d.split("_")[1]) in FILTER_SEEDS)
    ])
    if not seeds:
        return {}

    if FIXED_EPOCH is not None:
        valid = [s for s in seeds
                 if os.path.exists(os.path.join(base_dir, f"seed_{s}",
                                                "checkpoints",
                                                f"epoch{FIXED_EPOCH:04d}.pt"))]
        print(f"  [{key}] Fixed epoch {FIXED_EPOCH}, {len(valid)}/{len(seeds)} seeds found.")
        return {s: FIXED_EPOCH for s in valid}

    candidates = [
        os.path.join(PLOT_DIR, f"analyze_across_seeds_summary{_seed_tag}.json"),
        os.path.join(PLOT_DIR, f"decode_across_seeds_summary{_seed_tag}.json"),
        os.path.join(PLOT_DIR, f"decode_cog_params_summary{_seed_tag}.json"),
    ]
    for path in candidates:
        ep = load_epochs_from_json(path, key)
        if ep:
            ep = {s: e for s, e in ep.items() if s in seeds}
            if ep:
                print(f"  [{key}] Loaded epochs from {os.path.basename(path)}: {ep}")
                return ep

    # Inline LOSO fallback
    print(f"  [{key}] No summary JSON found. Running inline LOSO …")
    return _loso_inline(base_dir, seeds, is_latent)


# ── LOSO helpers ──────────────────────────────────────────────────────────────
def list_checkpoints(base_dir, seed):
    ckpt_dir = os.path.join(base_dir, f"seed_{seed}", "checkpoints")
    if not os.path.exists(ckpt_dir):
        return []
    return sorted([
        int(f.replace("epoch", "").replace(".pt", ""))
        for f in os.listdir(ckpt_dir)
        if f.startswith("epoch") and f.endswith(".pt")
    ])


def compute_nll_for_seed_epoch(base_dir, seed, epoch, is_latent,
                                xin_data, c_data, xin_enc, B):
    run_dir     = os.path.join(base_dir, f"seed_{seed}")
    ckpt_path   = os.path.join(run_dir, "checkpoints", f"epoch{epoch:04d}.pt")
    frozen_path = os.path.join(run_dir, "frozen_decoder", "policy_model.pt")
    if not os.path.exists(ckpt_path):
        return None
    if is_latent and not os.path.exists(frozen_path):
        return None
    try:
        model_config = load_model_config(run_dir)
        model = create_model_from_config(
            model_config, n_participants=B, device=device,
            frozen_decoder_path=frozen_path if is_latent else None
        )
        model = load_model_checkpoint(ckpt_path, model, device)
        if is_latent:
            x_enc = xin_enc if xin_enc is not None else xin_data
            with torch.no_grad():
                mu, _ = model.encoder(x_enc.unsqueeze(1), return_per_timestep=False)
            loss = compute_reconstruction_loss(
                model, xin_data, c_data, z_latent=mu, is_latent_model=True
            )
        else:
            loss = compute_reconstruction_loss(
                model, xin_data, c_data, is_latent_model=False
            )
        return float(loss.mean().item())
    except Exception as e:
        print(f"    NLL failed seed={seed} epoch={epoch}: {e}")
        return None


def _loso_inline(base_dir, seeds, is_latent):
    result = {}
    for seed in seeds:
        other = [s for s in seeds if s != seed]
        epochs_sets = {
            s: set(e for e in list_checkpoints(base_dir, s)
                   if MIN_EPOCH <= e <= MAX_EPOCH)
            for s in other
        }
        if not epochs_sets:
            continue
        common = sorted(set.intersection(*epochs_sets.values()))
        if not common:
            continue
        best_ep, best_nll = None, float("inf")
        for ep in common:
            nlls = [compute_nll_for_seed_epoch(base_dir, s, ep, is_latent,
                                               xin_train, c_train,
                                               xin_enc_train, B_train)
                    for s in other]
            nlls = [n for n in nlls if n is not None]
            if nlls and float(np.mean(nlls)) < best_nll:
                best_nll = float(np.mean(nlls))
                best_ep  = ep
        if best_ep is not None:
            result[seed] = best_ep
            print(f"    Seed {seed}: LOSO epoch = {best_ep}  (NLL = {best_nll:.4f})")
    return result


# ── Latent extraction ─────────────────────────────────────────────────────────
def get_latent_array(base_dir, seed, epoch, is_latent):
    """(B_test, D) — last-timestep mu (IDRNN) or time-avg hidden (Vanilla)."""
    run_dir  = os.path.join(base_dir, f"seed_{seed}")
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    frozen_path  = os.path.join(run_dir, "frozen_decoder", "policy_model.pt")
    ckpt_path    = os.path.join(run_dir, "checkpoints", f"epoch{epoch:04d}.pt")
    model_config = load_model_config(run_dir)
    model = create_model_from_config(
        model_config, n_participants=B_test, device=device,
        frozen_decoder_path=frozen_path if is_latent else None
    )
    model = load_model_checkpoint(ckpt_path, model, device)
    model.eval()
    with torch.no_grad():
        if is_latent:
            enc_in_dim = cfg.get("model_config", {}).get("enc_in_dim", cfg["in_dim"])
            xenc = xin_test if enc_in_dim == cfg["in_dim"] else xin_test
            mu, _ = model.encoder(xenc.unsqueeze(1), return_per_timestep=True)
            return mu.squeeze(1)[:, -1, :].cpu().numpy()
        else:
            _, _, hidden = model(xin_test)
            return hidden.mean(dim=1).cpu().numpy()


# ── Diagnostic: cog parameter distributions ───────────────────────────────────
def plot_cog_distributions():
    n = len(COG_VARS)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    fig.suptitle("Cognitive parameter distributions (test set)", fontsize=11)
    for ax, (vname, vals) in zip(axes, COG_VARS.items()):
        ax.hist(vals, bins=12, color="#4C72B0", edgecolor="white", alpha=0.85)
        sk = skew(vals)
        ku = kurtosis(vals, fisher=True)
        ax.set_title(f"{COG_LABELS[vname]}\nskew={sk:.2f}  kurt={ku:.2f}",
                     fontsize=9)
        ax.set_xlabel("value", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # warn about high skew
        if abs(sk) > 1.0:
            ax.set_facecolor("#fff3cd")
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, f"dist_cog_params{_seed_tag}.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


# ── Diagnostic: latent distributions ─────────────────────────────────────────
def plot_latent_distributions(lat_collection, label, color):
    """
    lat_collection: list of (B, D) arrays (one per seed).
    Plots per-dimension boxplots with all seeds overlaid.
    """
    if not lat_collection:
        return
    # Determine max z_dim
    z_dims = [lat.shape[1] for lat in lat_collection]
    z_dim  = max(z_dims)
    n_seeds = len(lat_collection)

    fig, axes = plt.subplots(1, z_dim, figsize=(2.5 * z_dim, 3.5), squeeze=False)
    axes = axes[0]
    fig.suptitle(f"{label}: latent dimension distributions\n"
                 f"({n_seeds} seeds, each seed = one boxplot)", fontsize=10)

    for dim in range(z_dim):
        ax   = axes[dim]
        data = [lat[:, dim] for lat in lat_collection if lat.shape[1] > dim]
        ax.boxplot(data, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.6),
                   medianprops=dict(color="k"),
                   flierprops=dict(marker=".", markersize=3, alpha=0.5))
        sk_vals = [skew(d) for d in data]
        ax.set_title(f"z{dim+1}\nmean skew={np.mean(sk_vals):.2f}", fontsize=8)
        ax.set_xlabel("seed", fontsize=7)
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels([str(i + 1) for i in range(len(data))], fontsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    tag = label.lower().replace(" ", "_")
    path = os.path.join(PLOT_DIR, f"dist_latents_{tag}{_seed_tag}.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


# ── LOOCV Ridge regression ────────────────────────────────────────────────────
def ridge_loocv(X, y_raw):
    """
    Returns (r, y_pred_aligned) where y_pred_aligned has the same length
    as y_raw (nan for rows excluded due to non-finite values).
    """
    X     = np.asarray(X, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    finite = np.isfinite(y_raw) & np.all(np.isfinite(X), axis=1)
    if finite.sum() < 4:
        return float("nan"), np.full(len(y_raw), np.nan)

    X_f, y_f = X[finite], y_raw[finite]

    # Optionally transform targets
    if TRANSFORM_TARGETS == "yeo-johnson":
        pt  = PowerTransformer(method="yeo-johnson", standardize=True)
        y_t = pt.fit_transform(y_f.reshape(-1, 1)).ravel()
    else:
        y_t = y_f.copy()

    alphas = np.logspace(-3, 3, 20)
    loo    = LeaveOneOut()
    y_pred = np.empty_like(y_t)

    for tr, te in loo.split(X_f):
        n_comp = (min(N_PCA, X_f.shape[1], len(tr) - 1)
                  if N_PCA is not None else None)
        steps = [StandardScaler()]
        if n_comp is not None:
            steps.append(PCA(n_components=n_comp))
        steps.append(RidgeCV(alphas=alphas, cv=min(5, len(tr))))
        pipe = make_pipeline(*steps)
        pipe.fit(X_f[tr], y_t[tr])
        y_pred[te] = pipe.predict(X_f[te])

    r, _ = pearsonr(y_t, y_pred)

    # Inverse-transform predictions back to original scale for the scatter plot
    if TRANSFORM_TARGETS == "yeo-johnson":
        y_pred_orig = pt.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    else:
        y_pred_orig = y_pred

    # Align back to full-length array
    y_pred_full = np.full(len(y_raw), np.nan)
    y_pred_full[finite] = y_pred_orig

    return float(r), y_pred_full


# ── Per-model analysis loop ───────────────────────────────────────────────────
def run_model_type(base_dir, is_latent, label):
    print(f"\n{'='*70}\n{label}  (is_latent={is_latent})\n{'='*70}")
    key          = "idrnn" if is_latent else "vanilla"
    seed_epochs  = get_epochs(base_dir, is_latent, key)
    seeds        = sorted(seed_epochs.keys())
    print(f"  Seeds: {seeds}")

    r_per_seed       = {v: [] for v in COG_VARS}
    novelty_pred_per_seed = {}   # seed → y_pred array (for scatter)
    lat_collection   = []

    for seed in seeds:
        epoch = seed_epochs[seed]
        print(f"\n--- Seed {seed}  epoch {epoch} ---")
        try:
            lat = get_latent_array(base_dir, seed, epoch, is_latent)
        except Exception as e:
            print(f"  Latent extraction failed: {e}")
            continue

        lat_collection.append(lat)

        for vname, vals in COG_VARS.items():
            r, y_pred = ridge_loocv(lat, vals)
            r_per_seed[vname].append(r)
            if vname == "w_novelty":
                novelty_pred_per_seed[seed] = y_pred

        print("  r: " + "  ".join(
            f"{v}={r_per_seed[v][-1]:+.3f}" for v in COG_VARS))

    # Latent distribution diagnostic
    plot_latent_distributions(lat_collection, label,
                              color="#4C72B0" if is_latent else "#DD8452")

    return {v: np.array(rs) for v, rs in r_per_seed.items()}, \
           novelty_pred_per_seed, seeds


# ── Run ───────────────────────────────────────────────────────────────────────
# Distribution diagnostic for cog params first (independent of seeds)
plot_cog_distributions()

r_idrnn, nov_pred_idrnn, seeds_idrnn = run_model_type(IDRNN_DIR, True,  "IDRNN")
r_van,   nov_pred_van,   seeds_van   = run_model_type(VAN_DIR,   False, "Vanilla")

# ── Plot 1: grouped bars ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
keys   = list(COG_VARS.keys())
x      = np.arange(len(keys))
w      = 0.35
rng    = np.random.default_rng(0)
jitter = 0.05

for i, k in enumerate(keys):
    iv = r_idrnn[k]
    vv = r_van[k]
    alpha = 0.95 if k == "w_novelty" else 0.55

    mi  = float(np.nanmean(iv)) if len(iv) else 0.0
    sei = (float(np.nanstd(iv) / np.sqrt(np.sum(~np.isnan(iv))))
           if np.sum(~np.isnan(iv)) > 1 else 0.0)
    mv  = float(np.nanmean(vv)) if len(vv) else 0.0
    sev = (float(np.nanstd(vv) / np.sqrt(np.sum(~np.isnan(vv))))
           if np.sum(~np.isnan(vv)) > 1 else 0.0)

    ax.bar(i - w/2, mi, w, color="#4C72B0", alpha=alpha,
           label="IDRNN"   if i == 0 else "_")
    ax.bar(i + w/2, mv, w, color="#DD8452", alpha=alpha,
           label="Vanilla" if i == 0 else "_")
    ax.errorbar([i - w/2, i + w/2], [mi, mv], yerr=[sei, sev],
                fmt="none", color="k", capsize=4, lw=1.2)
    for xi, accs in [(i - w/2, iv), (i + w/2, vv)]:
        jit = rng.uniform(-jitter, jitter, size=len(accs))
        ax.scatter(xi + jit, accs, color="k", s=18, zorder=5, alpha=0.7)

ax.axhline(0, color="k", lw=0.8, ls=":")
ax.set_xticks(x)
ax.set_xticklabels([COG_LABELS[k] for k in keys])
for lbl, k in zip(ax.get_xticklabels(), keys):
    if k == "w_novelty":
        lbl.set_fontweight("bold")

epoch_desc = (f"fixed epoch {FIXED_EPOCH}" if FIXED_EPOCH is not None
              else "LOSO train-NLL epoch selection")
pca_desc   = f", PCA→{N_PCA}" if N_PCA is not None else ""
tf_desc    = f", targets: {TRANSFORM_TARGETS}" if TRANSFORM_TARGETS != "none" else ""
ax.set_ylabel("Pearson r  (LOOCV Ridge, predicted vs true)", fontsize=11)
ax.set_title(f"Decoding cognitive parameters from latents\n"
             f"({epoch_desc}{pca_desc}{tf_desc}, mean ± SEM)", fontsize=11)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

tf_tag    = f"_{TRANSFORM_TARGETS}" if TRANSFORM_TARGETS != "none" else ""
pca_tag   = f"_pca{N_PCA}" if N_PCA is not None else ""
out_stem  = (f"decode_cog_params_epoch{FIXED_EPOCH}{_seed_tag}{pca_tag}{tf_tag}"
             if FIXED_EPOCH is not None
             else f"decode_cog_params_loso{_seed_tag}{pca_tag}{tf_tag}")
bar_path  = os.path.join(PLOT_DIR, f"{out_stem}.png")
fig.savefig(bar_path, dpi=150)
print(f"\nSaved: {bar_path}")
plt.close(fig)

# ── Plot 2: novelty scatter (predicted vs true) ───────────────────────────────
all_seeds_for_scatter = (
    [(s, "IDRNN",   nov_pred_idrnn, "#4C72B0") for s in seeds_idrnn
     if s in nov_pred_idrnn] +
    [(s, "Vanilla", nov_pred_van,   "#DD8452") for s in seeds_van
     if s in nov_pred_van]
)

n_idrnn  = len([s for s in seeds_idrnn if s in nov_pred_idrnn])
n_van    = len([s for s in seeds_van   if s in nov_pred_van])
n_cols   = max(n_idrnn, n_van, 1)
n_rows   = (1 if n_idrnn == 0 or n_van == 0 else 2)

fig2, axes2 = plt.subplots(n_rows, n_cols,
                            figsize=(3.5 * n_cols, 3.5 * n_rows),
                            squeeze=False)
fig2.suptitle("w_novelty: LOOCV predicted vs true (per seed)", fontsize=11)

true_nov = COG_VARS["w_novelty"]

def scatter_novelty(ax, seed, label, pred, color):
    finite = np.isfinite(true_nov) & np.isfinite(pred)
    if finite.sum() < 3:
        ax.set_visible(False)
        return
    for grp, gcol, glbl in zip(GROUP_ORDER, GROUP_COLORS, GROUP_LABELS):
        mask = (age_labels == grp) & finite
        ax.scatter(true_nov[mask], pred[mask],
                   color=gcol, s=35, alpha=0.8, edgecolors="none",
                   label=glbl)
    # regression line
    coef = np.polyfit(true_nov[finite], pred[finite], 1)
    xr   = np.linspace(true_nov[finite].min(), true_nov[finite].max(), 100)
    ax.plot(xr, np.poly1d(coef)(xr), "--", color=color, lw=1.5)
    r, _ = pearsonr(true_nov[finite], pred[finite])
    ax.set_title(f"{label} seed {seed}\nr = {r:.3f}", fontsize=9)
    ax.set_xlabel("true w_novelty", fontsize=8)
    ax.set_ylabel("predicted", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

row_idrnn = 0
row_van   = 1 if n_rows == 2 else 0

for col_i, seed in enumerate(s for s in seeds_idrnn if s in nov_pred_idrnn):
    scatter_novelty(axes2[row_idrnn, col_i], seed, "IDRNN",
                    nov_pred_idrnn[seed], "#4C72B0")
    if col_i == 0:
        axes2[row_idrnn, col_i].legend(fontsize=7, frameon=False)

for col_i, seed in enumerate(s for s in seeds_van if s in nov_pred_van):
    scatter_novelty(axes2[row_van, col_i], seed, "Vanilla",
                    nov_pred_van[seed], "#DD8452")

# Hide unused axes
for row in range(n_rows):
    for col in range(n_cols):
        if not axes2[row, col].collections and not axes2[row, col].lines:
            axes2[row, col].set_visible(False)

fig2.tight_layout()
scat_path = os.path.join(PLOT_DIR, f"novelty_scatter_{out_stem.split('decode_cog_params_')[1]}.png")
fig2.savefig(scat_path, dpi=150)
print(f"Saved: {scat_path}")
plt.close(fig2)

# ── Save summary ──────────────────────────────────────────────────────────────
summary = {
    "transform_targets": TRANSFORM_TARGETS,
    "idrnn": {
        "r_per_seed": {v: r_idrnn[v].tolist() for v in COG_VARS},
        "mean":       {v: float(np.nanmean(r_idrnn[v])) for v in COG_VARS},
        "sem":        {v: (float(np.nanstd(r_idrnn[v]) /
                             np.sqrt(np.sum(~np.isnan(r_idrnn[v]))))
                           if np.sum(~np.isnan(r_idrnn[v])) > 1 else 0.0)
                       for v in COG_VARS},
    },
    "vanilla": {
        "r_per_seed": {v: r_van[v].tolist() for v in COG_VARS},
        "mean":       {v: float(np.nanmean(r_van[v])) for v in COG_VARS},
        "sem":        {v: (float(np.nanstd(r_van[v]) /
                             np.sqrt(np.sum(~np.isnan(r_van[v]))))
                           if np.sum(~np.isnan(r_van[v])) > 1 else 0.0)
                       for v in COG_VARS},
    },
}
json_path = os.path.join(PLOT_DIR, f"{out_stem}.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved → {json_path}")
