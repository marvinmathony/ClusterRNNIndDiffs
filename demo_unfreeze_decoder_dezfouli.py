#!/usr/bin/env python3
"""
Demo: brief decoder unfreezing in step 2 to teach the policy time-varying z.

Loads existing v1 IDRNN checkpoints (3 seeds × fold 0), and for each seed:

  1. Computes the BASELINE per-subject test NLL on the fold's held-out subjects
     using the standard causal-posterior evaluator (encoder run causally per
     trial, frozen decoder trained with constant full-session μ).

  2. Briefly unfreezes the decoder, switches the encoder to
     return_per_timestep=True, and fine-tunes for K epochs with policy NLL
     using per-timestep μ at training time.  Hidden state is initialised from
     z2h0(μ(t=0)) per block (causal counterpart of the original z2h0(μ_full)).

  3. Re-runs the same causal-posterior evaluator on the test fold and reports
     the FINE-TUNED per-subject NLL.

  4. Pairs the two by subject (averaged across seeds), runs a paired t-test,
     and emits a CSV + PNG.

Outputs:
  plots_dezfouli/unfreeze_demo.csv
  plots_dezfouli/unfreeze_demo.png
"""
import os, sys, json, copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

sys.path.insert(0, ".")
from modelsandtraining import (
    Decoder, IDRNN, LatentRNN_secondstep,
    test_latentrnn_secondstep_causal_posterior_weighting,
    compute_rnn_likelihoods_torch,
)

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data_dezfouli/fold0"
RUN_BASE = "runs_dezfouli/fold0"        # v1 checkpoints (Mar 27)
SEEDS    = [200, 300, 400]
K_EPOCHS = 300                           # brief unfreezing budget (was 100)
LR       = 5e-4                          # was 1e-4
WD       = 1e-4
A        = 2
PLOT_DIR = "plots_dezfouli"
os.makedirs(PLOT_DIR, exist_ok=True)


def load_fold_arrays():
    xin_train = torch.from_numpy(np.load(f"{DATA_DIR}/xin_train.npy")).float().to(DEVICE)
    xin_test  = torch.from_numpy(np.load(f"{DATA_DIR}/xin_test.npy")).float().to(DEVICE)
    c_train   = torch.from_numpy(np.load(f"{DATA_DIR}/c_train.npy")).float().to(DEVICE)
    c_test    = torch.from_numpy(np.load(f"{DATA_DIR}/c_test.npy")).float().to(DEVICE)
    return xin_train, c_train, xin_test, c_test


def build_model_from_cfg(cfg):
    mc = cfg["model_config"]
    encoder = IDRNN(
        in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"], hid=mc["enc_hidden"],
        n_tasks=mc.get("n_tasks"), task_emb_dim=mc.get("task_emb_dim", 0),
        continuous_encoder=mc.get("continuous_encoder", False),
    )
    decoder = Decoder(in_dim=mc["dec_in_dim"], z_dim=mc["z_dim"],
                      hid=mc["hidden"], A=mc["A"])
    model = LatentRNN_secondstep(
        encoder=encoder, hid=mc["hidden"], z_dim=mc["z_dim"],
        in_dim=mc["dec_in_dim"], A=mc["A"], decoder=decoder,
        n_tasks=mc.get("n_tasks"), task_emb_dim=mc.get("task_emb_dim", 0),
        reinit_decoder_per_block=mc.get("reinit_decoder_per_block", False),
    )
    return model.to(DEVICE)


@torch.no_grad()
def causal_per_subject_nll(model, xin_test, c_test, xin_train, c_train):
    """Standard causal-posterior NLL per held-out subject (matches testing_script)."""
    model.eval()
    df, _, _, _ = compute_rnn_likelihoods_torch(
        test_latentrnn_secondstep_causal_posterior_weighting,
        model, xin_test, c_test, xin_train,
        latent=True, choice_train=c_train, id=True,
    )
    return df["normalized_likelihood"].values


def fine_tune_per_timestep_z(model, xin_train, c_train, k_epochs):
    """Unfreeze decoder, switch encoder to return_per_timestep=True, fine-tune
    for k_epochs with per-timestep μ as decoder input."""
    model.return_per_timestep = True
    for p in model.decoder.parameters(): p.requires_grad = True
    for p in model.encoder.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=WD)

    n_blocks = xin_train.shape[1]
    losses = []
    for ep in range(1, k_epochs + 1):
        model.train(); opt.zero_grad()
        # encoder produces (B, Bk, T, z_dim) when return_per_timestep=True
        mu, _ = model.encoder(xin_train, return_per_timestep=True)
        block_logits = []
        for b in range(n_blocks):
            seq = xin_train[:, b]                      # (B, T, in_dim)
            z_b = mu[:, b]                              # (B, T, z_dim)
            rnn_in = torch.cat([seq, z_b], dim=-1)     # (B, T, in+z)
            # causal h0: from μ at trial 0 of this block
            h0 = model.decoder.z2h0(z_b[:, 0]).unsqueeze(0).contiguous()
            out, _ = model.decoder.rnn(rnn_in, h0)
            block_logits.append(model.decoder.lin(out))
        logits = torch.stack(block_logits, dim=1)
        nll = F.cross_entropy(logits.reshape(-1, A),
                                c_train.reshape(-1).long(),
                                ignore_index=-100)
        nll.backward(); opt.step()
        losses.append(float(nll))
        if ep == 1 or ep % 50 == 0:
            print(f"    [per-t]   ep {ep:3d}  train_nll={nll.item():.4f}")

    # Reset to default for the standard causal-posterior eval below
    model.return_per_timestep = False
    return model, losses


def fine_tune_static_z(model, xin_train, c_train, k_epochs):
    """CONTROL: unfrozen decoder but z stays static (full-session μ broadcast
    across trials, same as original step 2).  Tests whether the test-NLL
    improvement comes specifically from per-timestep z or just from K more
    decoder updates."""
    model.return_per_timestep = False
    for p in model.decoder.parameters(): p.requires_grad = True
    for p in model.encoder.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=WD)
    losses = []
    for ep in range(1, k_epochs + 1):
        model.train(); opt.zero_grad()
        # standard step-2 forward: returns logits using last-trial μ broadcast
        logits, mu, lv, z, h0 = model(xin_train, xin_train, sample_z=False)
        nll = F.cross_entropy(logits.reshape(-1, A),
                                c_train.reshape(-1).long(),
                                ignore_index=-100)
        nll.backward(); opt.step()
        losses.append(float(nll))
        if ep == 1 or ep % 50 == 0:
            print(f"    [static]  ep {ep:3d}  train_nll={nll.item():.4f}")
    return model, losses


def main():
    xin_train, c_train, xin_test, c_test = load_fold_arrays()
    print(f"xin_train {tuple(xin_train.shape)}, xin_test {tuple(xin_test.shape)}")

    rows = []
    train_curves = {}
    for seed in SEEDS:
        run_dir = f"{RUN_BASE}/seed_{seed}"
        cfg_p   = f"{run_dir}/config.json"
        if not os.path.exists(cfg_p):
            print(f"[seed {seed}] SKIP — no config at {cfg_p}")
            continue
        cfg     = json.load(open(cfg_p))
        cv_ep   = cfg["cv_selected_epoch"]
        ckpt_p  = f"{run_dir}/checkpoints/epoch{cv_ep:04d}.pt"
        if not os.path.exists(ckpt_p):
            print(f"[seed {seed}] SKIP — no ckpt at {ckpt_p}")
            continue

        ckpt = torch.load(ckpt_p, map_location=DEVICE)

        print(f"\n══ seed {seed} ══")
        # ── Baseline ─────────────────────────────────────────────────────────
        m0 = build_model_from_cfg(cfg)
        m0.load_state_dict(ckpt)
        for p in m0.decoder.parameters(): p.requires_grad = False
        baseline = causal_per_subject_nll(m0, xin_test, c_test, xin_train, c_train)
        print(f"  BASELINE       mean NLL/trial = {baseline.mean():.4f}")

        # ── CONTROL: same K + lr but static z (decoder unfrozen) ─────────────
        m_ctrl = build_model_from_cfg(cfg)
        m_ctrl.load_state_dict(ckpt)
        m_ctrl, curve_ctrl = fine_tune_static_z(m_ctrl, xin_train, c_train, K_EPOCHS)
        ctrl = causal_per_subject_nll(m_ctrl, xin_test, c_test, xin_train, c_train)
        print(f"  CTRL static-z  mean NLL/trial = {ctrl.mean():.4f}  "
              f"Δ vs base = {ctrl.mean() - baseline.mean():+.4f}")

        # ── TREATMENT: per-timestep z (decoder unfrozen) ─────────────────────
        m1 = build_model_from_cfg(cfg)
        m1.load_state_dict(ckpt)
        m1, curve = fine_tune_per_timestep_z(m1, xin_train, c_train, K_EPOCHS)
        train_curves[seed] = curve
        finetuned = causal_per_subject_nll(m1, xin_test, c_test, xin_train, c_train)
        print(f"  TX  per-t-z    mean NLL/trial = {finetuned.mean():.4f}  "
              f"Δ vs base = {finetuned.mean() - baseline.mean():+.4f}  "
              f"Δ vs ctrl = {finetuned.mean() - ctrl.mean():+.4f}")

        for sid, (b, c, f) in enumerate(zip(baseline, ctrl, finetuned)):
            rows.append({"seed": seed, "subj": sid,
                          "baseline": float(b), "ctrl_static_z": float(c),
                          "finetuned": float(f)})

    if not rows:
        print("No checkpoints found — abort.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PLOT_DIR, "unfreeze_demo.csv"), index=False)

    # ── Pool across seeds (per-subject paired comparison) ────────────────────
    avg = (df.groupby("subj").agg(baseline=("baseline", "mean"),
                                    ctrl=("ctrl_static_z", "mean"),
                                    finetuned=("finetuned", "mean"))
              .reset_index())
    t_bf, p_bf = ttest_rel(avg["baseline"],  avg["finetuned"])
    t_bc, p_bc = ttest_rel(avg["baseline"],  avg["ctrl"])
    t_cf, p_cf = ttest_rel(avg["ctrl"],      avg["finetuned"])
    print("\n=== Paired comparison (per subject, mean across seeds; n="
          f"{len(avg)}) ===")
    print(f"  baseline       mean = {avg['baseline'].mean():.4f}")
    print(f"  ctrl static z  mean = {avg['ctrl'].mean():.4f}   "
          f"Δ = {(avg['baseline']-avg['ctrl']).mean():+.4f}  "
          f"t={t_bc:.3f}  p={p_bc:.3g}")
    print(f"  finetuned per-t mean = {avg['finetuned'].mean():.4f}   "
          f"Δ = {(avg['baseline']-avg['finetuned']).mean():+.4f}  "
          f"t={t_bf:.3f}  p={p_bf:.3g}")
    print(f"  per-t vs ctrl  Δ    = {(avg['ctrl']-avg['finetuned']).mean():+.4f}  "
          f"t={t_cf:.3f}  p={p_cf:.3g}  ← attribution test")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for _, row in avg.iterrows():
        ax.plot([0, 1, 2],
                [row["baseline"], row["ctrl"], row["finetuned"]],
                color="gray", alpha=0.4, linewidth=0.6)
    ax.scatter([0]*len(avg), avg["baseline"],  color="#4C72B0", s=44,
                edgecolor="k", linewidth=0.4, label="baseline (frozen, static z)")
    ax.scatter([1]*len(avg), avg["ctrl"],      color="#7B848F", s=44,
                edgecolor="k", linewidth=0.4, label="ctrl: unfrozen, static z")
    ax.scatter([2]*len(avg), avg["finetuned"], color="#DD8452", s=44,
                edgecolor="k", linewidth=0.4, label="tx: unfrozen, per-timestep z")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["baseline", "ctrl static-z", "tx per-t z"])
    ax.set_ylabel("Test NLL / trial (lower = better)")
    ax.set_title(f"Per-subject paired comparison (fold 0)\n"
                  f"tx vs ctrl: paired t={t_cf:.2f}, p={p_cf:.2g}, n={len(avg)}")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    for seed, curve in train_curves.items():
        ax.plot(np.arange(1, len(curve) + 1), curve, label=f"seed {seed}")
    ax.set_xlabel("Fine-tune epoch")
    ax.set_ylabel("Train NLL (per-trial)")
    ax.set_title("Fine-tune training curve")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend()

    fig.suptitle(f"Brief decoder unfreezing demo on dezfouli (K={K_EPOCHS} epochs, lr={LR})",
                  fontweight="bold")
    fig.tight_layout()
    out_png = os.path.join(PLOT_DIR, "unfreeze_demo.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved → {os.path.join(PLOT_DIR, 'unfreeze_demo.csv')}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
