#!/usr/bin/env python3
"""
Decode age-group labels from latents across all seeds independently.

Epoch selection uses leave-one-seed-out (LOSO) cross-validation:
  for each held-out seed s, the best epoch is chosen by minimising
  mean train-NLL across the other seeds — so the selection is
  completely independent of seed s.

For each held-out seed:
  1. Compute mean train NLL across other seeds for each epoch.
  2. Pick the epoch with lowest mean train NLL.
  3. Load seed s at that epoch, extract latents.
  4. Run LOOCV logistic regression to predict age group.

Then average balanced accuracy across seeds and plot with the same
format as plot_logistic (bars + significance line).
"""

import argparse
import os
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


from modelsandtraining import (
    compute_rnn_likelihoods_torch,
    test_latentrnn_secondstep_causal_posterior_weighting,
)
from compute_reconstruction_specificity import (
    load_model_config,
    create_model_from_config,
    load_model_checkpoint,
    compute_reconstruction_loss,
)

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Decode age group from latents across seeds")
parser.add_argument("--epoch", type=int, default=None,
                    help="Fixed epoch to use for all seeds. "
                         "If omitted, LOSO train-NLL selection is used.")
parser.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated list of seeds to use, e.g. '200,300,400'. "
                         "If omitted, all seeds found in the run directories are used.")
args = parser.parse_args()
FIXED_EPOCH   = args.epoch
FILTER_SEEDS  = (set(int(s) for s in args.seeds.split(",")) if args.seeds else None)

# ── configuration ─────────────────────────────────────────────────────────────
DGP        = "sloutsky"
DATA_DIR   = f"data_{DGP}"
IDRNN_DIR  = f"runs_{DGP}"
VAN_DIR    = f"runs_vanilla_{DGP}"
PLOT_DIR   = f"plots_{DGP}"
MIN_EPOCH  = 100
MAX_EPOCH  = 3000
INNER_CV   = 5          # inner folds for LogisticRegressionCV
os.makedirs(PLOT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── load data ─────────────────────────────────────────────────────────────────
xin_test  = torch.from_numpy(np.load(f"{DATA_DIR}/xin_test.npy")).float().to(device)
c_test    = torch.from_numpy(np.load(f"{DATA_DIR}/c_test.npy")).float().to(device)
xin_train = torch.from_numpy(np.load(f"{DATA_DIR}/xin_train.npy")).float().to(device)
c_train   = torch.from_numpy(np.load(f"{DATA_DIR}/c_train.npy")).float().to(device)

_enc_test_path  = f"{DATA_DIR}/xin_enc_test.npy"
xin_enc_test    = (torch.from_numpy(np.load(_enc_test_path)).float().to(device)
                   if os.path.exists(_enc_test_path) else None)
_enc_train_path = f"{DATA_DIR}/xin_enc_train.npy"
xin_enc_train   = (torch.from_numpy(np.load(_enc_train_path)).float().to(device)
                   if os.path.exists(_enc_train_path) else None)

B_test  = xin_test.shape[0]
B_train = xin_train.shape[0]

# ── age-group labels for test set ─────────────────────────────────────────────
df_test = pd.read_csv(f"{DATA_DIR}/df_test.csv")
df_unique = (df_test.drop_duplicates(subset="subid")
                    .sort_values("subid")
                    .reset_index(drop=True))
GROUP_MAP = {"young_child": 0, "old_child": 1, "adult": 2}
labels = np.array([GROUP_MAP[g] for g in df_unique["age"]])
print(f"Test labels: {dict(zip(*np.unique(labels, return_counts=True)))}")


# ── helpers ───────────────────────────────────────────────────────────────────

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
    """Return mean NLL for a single seed at a single epoch."""
    run_dir  = os.path.join(base_dir, f"seed_{seed}")
    ckpt_path = os.path.join(run_dir, "checkpoints", f"epoch{epoch:04d}.pt")
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
            x_enc_batched = x_enc.unsqueeze(1)
            with torch.no_grad():
                mu, _ = model.encoder(x_enc_batched, return_per_timestep=False)
            loss = compute_reconstruction_loss(
                model, xin_data, c_data, z_latent=mu, is_latent_model=True
            )
        else:
            loss = compute_reconstruction_loss(
                model, xin_data, c_data, is_latent_model=False
            )
        return float(loss.mean().item())
    except Exception as e:
        print(f"  NLL failed seed={seed} epoch={epoch}: {e}")
        return None


def extract_latents(base_dir, seed, epoch, is_latent,
                    xin_test, c_test, xin_train, c_train,
                    xin_enc_test, xin_enc_train, B):
    """Load model and return test latents."""
    run_dir     = os.path.join(base_dir, f"seed_{seed}")
    ckpt_path   = os.path.join(run_dir, "checkpoints", f"epoch{epoch:04d}.pt")
    frozen_path = os.path.join(run_dir, "frozen_decoder", "policy_model.pt")

    model_config = load_model_config(run_dir)
    cfg_path     = os.path.join(run_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    # If enc_in_dim == in_dim the model was trained with full input for encoder
    if is_latent:
        _enc_in_dim = cfg.get("model_config", {}).get("enc_in_dim", cfg["in_dim"])
        if _enc_in_dim == cfg["in_dim"]:
            xin_enc_test  = xin_test
            xin_enc_train = xin_train

    model = create_model_from_config(
        model_config, n_participants=B, device=device,
        frozen_decoder_path=frozen_path if is_latent else None
    )
    model = load_model_checkpoint(ckpt_path, model, device)

    with torch.no_grad():
        _, lat_tensor, _, _ = compute_rnn_likelihoods_torch(
            test_latentrnn_secondstep_causal_posterior_weighting,
            model, xin_test, c_test, xin_train,
            latent=is_latent, choice_train=c_train,
            id=True,
            test_xin_enc=xin_enc_test,
            train_xin_enc=xin_enc_train,
        )
    return lat_tensor  # (B, T, z_dim) for IDRNN; (B, T, hid) for Vanilla


def logistic_loocv(X, y, inner_cv=INNER_CV):
    """LOOCV logistic regression; returns (preds, bal_acc)."""
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    loo = LeaveOneOut()
    preds = np.empty(len(y), dtype=classes.dtype)

    for tr, te in tqdm(loo.split(X), total=len(y), desc="  LOOCV", leave=False):
        y_tr = y[tr]
        min_class = np.min(np.bincount(y_tr))
        n_splits  = min(inner_cv, min_class) if min_class >= 2 else 2
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        base = LogisticRegressionCV(
            penalty="l2", solver="lbfgs", max_iter=5000, cv=cv,
        )
        clf = make_pipeline(StandardScaler(), base)
        #clf = base
        clf.fit(X[tr], y_tr)
        preds[te] = clf.predict(X[te])

    return preds, balanced_accuracy_score(y, preds)


# ── per-seed best-epoch selection + decoding ──────────────────────────────────

def run_model_type(base_dir, is_latent, label, fixed_epoch=None):
    seeds = sorted([
        int(d.split("_")[1])
        for d in os.listdir(base_dir)
        if d.startswith("seed_")
        and (FILTER_SEEDS is None or int(d.split("_")[1]) in FILTER_SEEDS)
    ])
    print(f"\n{'='*70}")
    print(f"{label}  |  seeds: {seeds}")
    if fixed_epoch is not None:
        print(f"  Using fixed epoch: {fixed_epoch}")
    print(f"{'='*70}")

    seed_bal_accs = []
    seed_best_epochs = {}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        if fixed_epoch is not None:
            ckpt_path = os.path.join(base_dir, f"seed_{seed}", "checkpoints",
                                     f"epoch{fixed_epoch:04d}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  Checkpoint for epoch {fixed_epoch} not found. Skipping.")
                continue
            best_epoch = fixed_epoch
            print(f"  Using fixed epoch: {best_epoch}")
            seed_best_epochs[seed] = {"epoch": best_epoch}
        else:
            # LOSO epoch selection: use mean train NLL from all other seeds
            other_seeds = [s for s in seeds if s != seed]
            epochs_sets = {
                s: set(e for e in list_checkpoints(base_dir, s)
                       if MIN_EPOCH <= e <= MAX_EPOCH)
                for s in other_seeds
            }
            if not epochs_sets:
                print(f"  No other seeds available. Skipping.")
                continue
            common_epochs = sorted(set.intersection(*epochs_sets.values()))
            if not common_epochs:
                print(f"  No common epochs across other seeds. Skipping.")
                continue

            print(f"  Scanning {len(common_epochs)} common epochs across "
                  f"{len(other_seeds)} other seeds (train NLL) …")
            best_epoch, best_mean_nll = None, float("inf")
            for epoch in common_epochs:
                nlls = []
                for s in other_seeds:
                    nll = compute_nll_for_seed_epoch(
                        base_dir, s, epoch, is_latent,
                        xin_train, c_train, xin_enc_train, B_train
                    )
                    if nll is not None:
                        nlls.append(nll)
                if nlls:
                    mean_nll = float(np.mean(nlls))
                    if mean_nll < best_mean_nll:
                        best_mean_nll = mean_nll
                        best_epoch    = epoch

            if best_epoch is None:
                print(f"  Could not compute train NLL for any epoch. Skipping.")
                continue

            print(f"  LOSO best epoch: {best_epoch}  "
                  f"(mean train NLL across other seeds = {best_mean_nll:.4f})")
            seed_best_epochs[seed] = {"epoch": best_epoch,
                                      "loso_mean_train_nll": best_mean_nll}

        # Extract latents at best epoch
        lat = extract_latents(
            base_dir, seed, best_epoch, is_latent,
            xin_test, c_test, xin_train, c_train,
            xin_enc_test, None, B_test
        )  # lat: (B, T, d)

        # Aggregate across time
        if is_latent:
            X = lat[:, -1, :].cpu().numpy()   # last timestep (best posterior)
        else:
            X = lat.mean(dim=1).cpu().numpy()  # time-average for vanilla

        _, bal_acc = logistic_loocv(X, labels)
        print(f"  Balanced accuracy: {bal_acc:.4f}")
        seed_bal_accs.append(bal_acc)

    bal_accs = np.array(seed_bal_accs)
    print(f"\n{label} — bal_acc per seed: {bal_accs}")
    print(f"  mean={bal_accs.mean():.4f}  std={bal_accs.std():.4f}  n={len(bal_accs)}")
    return bal_accs, seed_best_epochs


bal_accs_idrnn, best_epochs_idrnn = run_model_type(IDRNN_DIR, is_latent=True,  label="IDRNN",   fixed_epoch=FIXED_EPOCH)
bal_accs_van,   best_epochs_van   = run_model_type(VAN_DIR,   is_latent=False, label="Vanilla", fixed_epoch=FIXED_EPOCH)

# ── plot ───────────────────────────────────────────────────────────────────────

def plot_across_seeds(bal_accs_idrnn, bal_accs_van, out_name, fixed_epoch=None):
    mean_i  = bal_accs_idrnn.mean()
    sem_i   = bal_accs_idrnn.std() / np.sqrt(len(bal_accs_idrnn))
    mean_v  = bal_accs_van.mean()
    sem_v   = bal_accs_van.std() / np.sqrt(len(bal_accs_van))

    fig, ax = plt.subplots()
    colors  = ["#4C72B0", "#DD8452"]
    xs      = [0, 1]
    means   = [mean_i, mean_v]
    sems    = [sem_i, sem_v]
    labels_ = ["IDRNN", "Vanilla"]

    bars = ax.bar(xs, means, color=colors, edgecolor="k", width=0.5)
    ax.errorbar(xs, means, yerr=sems, fmt="none", color="k", capsize=5, lw=1.5)

    # overlay individual seed points
    jitter = 0.05
    rng = np.random.default_rng(0)
    for xi, accs in zip(xs, [bal_accs_idrnn, bal_accs_van]):
        jit = rng.uniform(-jitter, jitter, size=len(accs))
        ax.scatter(xi + jit, accs, color="k", s=20, zorder=5, alpha=0.7)

    # significance annotation: paired t-test across seeds
    from scipy.stats import ttest_rel
    if len(bal_accs_idrnn) == len(bal_accs_van) and len(bal_accs_idrnn) > 1:
        _, p_val = ttest_rel(bal_accs_idrnn, bal_accs_van)
    else:
        p_val = np.nan

    max_bar = max(mean_i + sem_i, mean_v + sem_v)
    y_line  = max_bar + 0.04
    y_text  = y_line + 0.01
    ax.plot([bars[0].get_x() + bars[0].get_width()/2,
             bars[1].get_x() + bars[1].get_width()/2],
            [y_line, y_line], color="k", lw=0.8)
    star = ("***" if p_val < 0.001 else "**" if p_val < 0.01
            else "*" if p_val < 0.05 else "n.s.")
    p_str = f"p={p_val:.3f} {star}" if not np.isnan(p_val) else "n.s."
    ax.text(0.5, y_text, p_str, ha="center", fontsize=9)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels_)
    ax.set_ylabel("Balanced Accuracy", fontsize=11)
    epoch_desc = (f"fixed epoch {fixed_epoch}" if fixed_epoch is not None
                  else "LOSO train-NLL epoch selection")
    ax.set_title(f"LOOCV Decoding Accuracy\n({epoch_desc}, mean ± SEM)", fontsize=11)
    ax.set_ylim(0, min(y_text + 0.08, 1.05))
    ax.axhline(1/3, color="k", ls=":", lw=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(PLOT_DIR, f"{out_name}.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved plot → {path}")
    plt.close(fig)


_seed_tag  = ("_seeds" + "-".join(str(s) for s in sorted(FILTER_SEEDS))
              if FILTER_SEEDS else "")
out_stem = (f"logistic_across_seeds_epoch{FIXED_EPOCH}{_seed_tag}" if FIXED_EPOCH is not None
            else f"logistic_across_seeds_loso{_seed_tag}")
plot_across_seeds(bal_accs_idrnn, bal_accs_van, out_stem, fixed_epoch=FIXED_EPOCH)

# ── save summary ──────────────────────────────────────────────────────────────
summary = {
    "idrnn": {
        "bal_accs_per_seed": bal_accs_idrnn.tolist(),
        "mean": float(bal_accs_idrnn.mean()),
        "sem":  float(bal_accs_idrnn.std() / np.sqrt(len(bal_accs_idrnn))),
        "best_epochs": best_epochs_idrnn,
    },
    "vanilla": {
        "bal_accs_per_seed": bal_accs_van.tolist(),
        "mean": float(bal_accs_van.mean()),
        "sem":  float(bal_accs_van.std() / np.sqrt(len(bal_accs_van))),
        "best_epochs": best_epochs_van,
    },
}
out_json = os.path.join(PLOT_DIR, f"decode_across_seeds_summary{_seed_tag}.json")
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved → {out_json}")
