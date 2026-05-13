#!/usr/bin/env python3
"""
rate_distortion_thalmann.py — Rate-distortion analysis for the Thalmann
two-task IDRNN pipeline.

The step-2 loss is:
    L = lambda * KL(enc_z || lookup_z) + (1 - lambda) * NLL

  "Rate"       = KL(enc_z || lookup_z)   … how far is encoder from step-1 z
  "Distortion" = NLL on training choices … how well does decoder predict

This script:
  1. For each fold × seed at the CV-selected epoch, computes KL and NLL
     separately (post-hoc forward pass on training data) and plots
     (KL, NLL) as a scatter.
  2. For one representative seed, loads ALL checkpoint epochs and plots
     the (KL, NLL) training trajectory to visualise the trade-off path.
  3. When runs with multiple lambda values exist (read from config.json),
     groups by lambda and plots the Pareto frontier with error bars.

Usage
-----
    # Single-lambda summary (current runs):
    python rate_distortion_thalmann.py

    # Restrict to specific seeds:
    python rate_distortion_thalmann.py --seeds 200,300

    # Also compute trajectories (slow — loads every checkpoint):
    python rate_distortion_thalmann.py --trajectory
"""
import argparse, os, json
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dgp",        type=str, default="thalmann")
parser.add_argument("--folds",      type=int, default=3)
parser.add_argument("--seeds",      type=str, default=None,
                    help="Comma-separated seeds to include, e.g. '200,300'")
parser.add_argument("--trajectory", action="store_true",
                    help="Also compute (KL, NLL) at every checkpoint epoch "
                         "for one representative seed per fold (slow)")
args = parser.parse_args()

DGP        = args.dgp
N_FOLDS    = args.folds
IDRNN_BASE = f"runs_{DGP}"
DATA_DIR   = f"data_{DGP}"
PLOT_DIR   = f"plots_{DGP}"
FILTER_SEEDS = (set(int(s) for s in args.seeds.split(",")) if args.seeds else None)

os.makedirs(PLOT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Import model classes ──────────────────────────────────────────────────────
from modelsandtraining import (
    IDRNN, Decoder, LatentRNN_secondstep,
    LookupEncoderZ, LatentRNNz,
    gaussian_nll_point, elbo_lossZ
)

# ── Discover (lambda, seed, run_dir) triples ─────────────────────────────────
def discover_runs(fold):
    """
    Returns list of (lmbd, seed, run_dir) tuples for the given fold.
    Looks in two places:
      - runs_{DGP}/fold{k}/seed_{S}/            ← standard runs (lambda in config)
      - runs_{DGP}/fold{k}/lmbd{L}/seed_{S}/   ← lambda-sweep runs
    """
    fold_dir = os.path.join(IDRNN_BASE, f"fold{fold}")
    if not os.path.isdir(fold_dir):
        return []
    runs = []
    for d in os.listdir(fold_dir):
        sub = os.path.join(fold_dir, d)
        if not os.path.isdir(sub):
            continue
        if d.startswith("seed_"):
            # Standard run: read lambda from config
            seed = int(d.split("_")[1])
            if FILTER_SEEDS is not None and seed not in FILTER_SEEDS:
                continue
            cfg_path = os.path.join(sub, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    lmbd = json.load(f).get("lmbd", 0.1)
                runs.append((lmbd, seed, sub))
        elif d.startswith("lmbd"):
            # Lambda-sweep subdir: iterate seed dirs inside
            for sd in os.listdir(sub):
                if not sd.startswith("seed_"):
                    continue
                seed = int(sd.split("_")[1])
                if FILTER_SEEDS is not None and seed not in FILTER_SEEDS:
                    continue
                run_dir  = os.path.join(sub, sd)
                cfg_path = os.path.join(run_dir, "config.json")
                if os.path.exists(cfg_path):
                    with open(cfg_path) as f:
                        lmbd = json.load(f).get("lmbd", float(d.replace("lmbd","").replace("p",".")))
                    runs.append((lmbd, seed, run_dir))
    return sorted(runs)



# ── Load training data for a fold ────────────────────────────────────────────
def load_fold_train_data(fold):
    """
    Returns (xin_enc, xin_blocks, c_train) tensors for the training split.
    xin_enc   : encoder input (N, B_blk, T, enc_in_dim)
    xin_blocks: decoder input (N, B_blk, T, dec_in_dim)  [same as xin_enc if same_enc_dec]
    c_train   : choice labels (N, B_blk, T) long
    """
    fold_dir = os.path.join(DATA_DIR, f"fold{fold}")
    xin  = torch.from_numpy(np.load(os.path.join(fold_dir, "xin_train.npy"))).float()
    c    = torch.from_numpy(np.load(os.path.join(fold_dir, "c_train.npy"))).float()

    enc_path = os.path.join(fold_dir, "xin_enc_train.npy")
    xin_enc  = torch.from_numpy(np.load(enc_path)).float() if os.path.exists(enc_path) else xin

    # Both must be block-structured (N, B_blk, T, ...)
    if xin.dim() == 3:           # (N, T, d) → unsqueeze block dim
        xin     = xin.unsqueeze(1)
        xin_enc = xin_enc.unsqueeze(1)
        c       = c.unsqueeze(1)

    # c may be one-hot; convert to int labels
    if c.dim() == 4 and c.shape[-1] > 1:
        c = c.argmax(dim=-1)     # (N, B_blk, T)
    elif c.dim() == 4 and c.shape[-1] == 1:
        c = c.squeeze(-1).long()
    else:
        c = c.long()

    return xin_enc.to(device), xin.to(device), c.to(device)


# ── Compute KL and NLL for one model at one checkpoint ────────────────────────
@torch.no_grad()
def compute_kl_nll(model, xin_enc, xin_blocks, c_train, lookup_z):
    """
    Returns (kl_scalar, nll_scalar).
    kl  = gaussian_nll_point(mu_enc, logvar_enc, lookup_z)   [rate]
    nll = cross_entropy(logits, choices)                       [distortion]
    """
    # Full forward with encoder
    logits, mu, lv, z, _ = model(xin_enc, xin_blocks)
    # logits: (N, B_blk, T, A)  mu/lv: (N, z_dim)

    kl  = gaussian_nll_point(mu, lv, lookup_z).item()

    # NLL: mask out padding (-100 in choice labels)
    A      = logits.shape[-1]
    N, Bk, T = c_train.shape
    logits_flat = logits.reshape(-1, A)
    c_flat      = c_train.reshape(-1)
    mask        = (c_flat >= 0)           # -100 → padding → ignore
    if mask.sum() == 0:
        return kl, float("nan")
    nll = F.cross_entropy(logits_flat[mask], c_flat[mask]).item()
    return kl, nll


# ── Main collection loop ──────────────────────────────────────────────────────
print("Collecting (KL, NLL) at best epoch for each fold × seed ...")

# keyed by lambda value
by_lambda = defaultdict(lambda: {"kl": [], "nll": [], "total_loss": []})
trajectory_data = []   # list of (fold, seed, lambda, epochs, kls, nlls)

for fold in range(N_FOLDS):
    xin_enc, xin_blocks, c_train = load_fold_train_data(fold)
    print(f"\n  Fold {fold}: data loaded. N_train={xin_enc.shape[0]}, "
          f"B_blk={xin_enc.shape[1]}, T={xin_enc.shape[2]}")

    fold_runs = discover_runs(fold)
    traj_done_for_fold = False
    for lmbd, seed, run_dir in fold_runs:
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        best_ep = cfg.get("cv_selected_epoch", None)
        if best_ep is None:
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            if not os.path.isdir(ckpt_dir):
                continue
            eps = sorted([int(fn.replace("epoch","").replace(".pt",""))
                          for fn in os.listdir(ckpt_dir)
                          if fn.startswith("epoch") and fn.endswith(".pt")])
            if not eps:
                continue
            best_ep = eps[-1]

        # Load lookup_z: standard runs store it in run_dir/frozen_decoder/;
        # lambda-sweep runs have the same structure.
        policy_path = os.path.join(run_dir, "frozen_decoder", "policy_model.pt")
        if not os.path.exists(policy_path):
            continue
        sd       = torch.load(policy_path, map_location="cpu")
        lookup_z = sd["encoder.embed.weight"].to(device)

        # Build model using the run_dir-specific config
        enc_in  = cfg.get("model_config", {}).get("enc_in_dim", cfg["in_dim"])
        enc_hid = cfg.get("model_config", {}).get("enc_hidden", cfg.get("hidden", 8))
        dec_in  = cfg.get("model_config", {}).get("dec_in_dim", cfg["in_dim"])
        n_tasks = cfg.get("model_config", {}).get("n_tasks", None)
        t_emb   = cfg.get("model_config", {}).get("task_emb_dim", 0)
        reinit  = cfg.get("model_config", {}).get("reinit_decoder_per_block", False)
        ckpt_path = os.path.join(run_dir, "checkpoints", f"epoch{best_ep:04d}.pt")
        if not os.path.exists(ckpt_path):
            continue

        enc = IDRNN(in_dim=enc_in, z_dim=cfg["z_dim"], hid=enc_hid,
                    n_tasks=n_tasks, task_emb_dim=t_emb)
        dec = Decoder(in_dim=dec_in, z_dim=cfg["z_dim"], hid=cfg["hidden"], A=cfg["A"])
        for p in dec.parameters(): p.requires_grad = False
        model = LatentRNN_secondstep(encoder=enc, hid=cfg["hidden"], z_dim=cfg["z_dim"],
                                     in_dim=dec_in, A=cfg["A"], decoder=dec,
                                     n_tasks=n_tasks, task_emb_dim=t_emb,
                                     reinit_decoder_per_block=reinit)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()
        if n_tasks is not None and t_emb > 0 and DGP == "thalmann":
            tids = torch.from_numpy(
                np.load(os.path.join("data_thalmann", "task_ids_per_block.npy"))
            ).long().to(device)
            model.set_task_ids(tids)

        kl, nll = compute_kl_nll(model, xin_enc, xin_blocks, c_train, lookup_z)
        total   = lmbd * kl + (1 - lmbd) * nll
        by_lambda[lmbd]["kl"].append(kl)
        by_lambda[lmbd]["nll"].append(nll)
        by_lambda[lmbd]["total_loss"].append(total)
        print(f"    seed={seed}  lambda={lmbd}  best_ep={best_ep}"
              f"  KL={kl:.4f}  NLL={nll:.4f}  total={total:.4f}")

        # ── Training trajectory (optional, expensive) ───────────────────────
        if args.trajectory and not traj_done_for_fold:
            print(f"    [trajectory] loading all checkpoints for seed={seed} ...")
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            eps_all  = sorted([int(fn.replace("epoch","").replace(".pt",""))
                               for fn in os.listdir(ckpt_dir)
                               if fn.startswith("epoch") and fn.endswith(".pt")])
            traj_kls, traj_nlls = [], []
            for ep in eps_all:
                ep_ckpt = os.path.join(ckpt_dir, f"epoch{ep:04d}.pt")
                m_state = torch.load(ep_ckpt, map_location=device)
                model.load_state_dict(m_state)
                model.eval()
                k, n = compute_kl_nll(model, xin_enc, xin_blocks, c_train, lookup_z)
                traj_kls.append(k); traj_nlls.append(n)
            trajectory_data.append((fold, seed, lmbd, eps_all, traj_kls, traj_nlls))
            traj_done_for_fold = True

# ── Plot 1: (KL, NLL) scatter coloured by lambda ─────────────────────────────
lambdas_found = sorted(by_lambda.keys())
print(f"\nLambda values found: {lambdas_found}")

cmap   = plt.cm.viridis
lmin   = min(lambdas_found); lmax = max(lambdas_found)
norm   = plt.Normalize(vmin=lmin, vmax=lmax)

fig, ax = plt.subplots(figsize=(7, 5))
for lmbd in lambdas_found:
    kls  = np.array(by_lambda[lmbd]["kl"])
    nlls = np.array(by_lambda[lmbd]["nll"])
    col  = cmap(norm(lmbd))
    ax.scatter(kls, nlls, color=col, s=60, alpha=0.8, zorder=3,
               label=f"λ={lmbd:.2f}")
    # Mean marker
    ax.scatter(kls.mean(), nlls.mean(), color=col, s=160,
               edgecolors="black", linewidths=1.2, zorder=4, marker="D")
    # Error ellipse (±1 std)
    ax.errorbar(kls.mean(), nlls.mean(),
                xerr=kls.std(), yerr=nlls.std(),
                fmt="none", color=col, lw=1.5, capsize=4, zorder=3)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="λ (rate weight)")
ax.set_xlabel("KL( enc_z ‖ lookup_z )   [rate]")
ax.set_ylabel("NLL on training choices  [distortion]")
ax.set_title(f"Rate–distortion plot — {DGP.capitalize()} IDRNN\n"
             f"(post-hoc at CV-selected epoch; N_seeds={sum(len(v['kl']) for v in by_lambda.values())})")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
if len(lambdas_found) > 1:
    ax.legend(fontsize=8)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "rate_distortion.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nRate-distortion scatter → {out}")

# ── Plot 2: KL and NLL over epochs (trajectory) ───────────────────────────────
if trajectory_data:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    cmap2 = plt.cm.tab10
    for idx, (fold, seed, lmbd, eps, kls, nlls) in enumerate(trajectory_data):
        col = cmap2(idx % 10)
        lbl = f"fold={fold} seed={seed}"
        ax1.plot(eps, kls,  color=col, lw=1.5, label=lbl)
        ax2.plot(eps, nlls, color=col, lw=1.5, label=lbl)
        ax3.plot(kls, nlls, color=col, lw=1.5, label=lbl, alpha=0.8)
        # Mark start and end
        ax3.scatter(kls[0],  nlls[0],  color=col, s=80, marker="o", zorder=4)
        ax3.scatter(kls[-1], nlls[-1], color=col, s=80, marker="*", zorder=4)

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("KL (rate)")
    ax1.set_title("Rate over training"); ax1.legend(fontsize=7)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("NLL (distortion)")
    ax2.set_title("Distortion over training")
    ax3.set_xlabel("KL (rate)"); ax3.set_ylabel("NLL (distortion)")
    ax3.set_title("Rate–distortion trajectory\n(○=start, ★=end)")

    for ax in (ax1, ax2, ax3):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "rate_distortion_trajectory.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Trajectory plot → {out}")

# ── Plot 3: Pareto frontier (if multiple lambda values) ───────────────────────
if len(lambdas_found) > 1:
    mean_kl  = [np.mean(by_lambda[l]["kl"])  for l in lambdas_found]
    mean_nll = [np.mean(by_lambda[l]["nll"]) for l in lambdas_found]
    std_kl   = [np.std(by_lambda[l]["kl"])   for l in lambdas_found]
    std_nll  = [np.std(by_lambda[l]["nll"])  for l in lambdas_found]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_kl, mean_nll, "k-", lw=1.5, zorder=2)
    sc = ax.scatter(mean_kl, mean_nll, c=lambdas_found, cmap="viridis", s=100, zorder=3)
    ax.errorbar(mean_kl, mean_nll, xerr=std_kl, yerr=std_nll,
                fmt="none", color="gray", lw=1, capsize=3, zorder=2)
    plt.colorbar(sc, ax=ax, label="λ")
    for lmbd, kl, nll in zip(lambdas_found, mean_kl, mean_nll):
        ax.annotate(f"λ={lmbd:.2f}", (kl, nll), textcoords="offset points",
                    xytext=(5, 3), fontsize=7)
    ax.set_xlabel("Mean KL (rate)")
    ax.set_ylabel("Mean NLL (distortion)")
    ax.set_title(f"Pareto frontier — {DGP.capitalize()} IDRNN\n(mean ± std across seeds)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "rate_distortion_pareto.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Pareto frontier plot → {out}")

# ── Save results ──────────────────────────────────────────────────────────────
summary = {
    str(lmbd): {
        "kl_mean":  float(np.mean(by_lambda[lmbd]["kl"])),
        "kl_std":   float(np.std(by_lambda[lmbd]["kl"])),
        "nll_mean": float(np.mean(by_lambda[lmbd]["nll"])),
        "nll_std":  float(np.std(by_lambda[lmbd]["nll"])),
        "n":        len(by_lambda[lmbd]["kl"]),
    }
    for lmbd in lambdas_found
}
out_json = os.path.join(PLOT_DIR, "rate_distortion_results.json")
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved → {out_json}")
print("Done.")
