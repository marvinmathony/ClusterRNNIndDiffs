#!/usr/bin/env python3
"""
Evaluate group decoding (children vs adults) for all IDRNN hyperparameter combos.

Two-stage evaluation matching the real pipeline:
  1. Select best epoch per combo using NLL on val data (falls back to test)
  2. Evaluate group decoding at that NLL-selected epoch

Also reports the "ceiling" (best decoding across sampled epochs) to show the gap
between what NLL selects and what's theoretically achievable.

Usage:
    # After all hyperparam_search_sloutsky.sbatch jobs finish:
    python hp_eval_decoding_sloutsky.py

    # Evaluate a single combo:
    python hp_eval_decoding_sloutsky.py --lmbd 0.01 --z 3 --hidden 5 --enc_hidden 10
"""

import os
import sys
import json
import copy
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

from modelsandtraining import (
    IDRNN, Decoder, LatentRNN_secondstep, LatentRNNz, LookupEncoderZ, AblatedRNN
)
from compute_reconstruction_specificity import (
    load_model_config, create_model_from_config, load_model_checkpoint,
    compute_reconstruction_loss
)

# ── Configuration ───────────────────────────────────────────────────────
HP_RUNS_BASE = "hp_search_runs_sloutsky"
DATA_DIR = "data_sloutsky"
VANILLA_BASE = "runs_vanilla_sloutsky"
NLL_EPOCH_WINDOW = (100, 3000)


# ── NLL helper (same logic as select_best_epoch_by_nll.py) ──────────────

def compute_nll(model, x_input, targets, is_latent_model, device, x_enc_input=None):
    """
    Compute mean NLL per participant.
    For IDRNN: encode → z (mu) → decode.
    For Vanilla: full forward pass.
    """
    if is_latent_model:
        x_for_enc = x_enc_input if x_enc_input is not None else x_input
        x_enc = x_for_enc.unsqueeze(1)  # (B, 1, T, enc_in_dim)
        with torch.no_grad():
            mu, _ = model.encoder(x_enc, return_per_timestep=False)
        loss_per_participant = compute_reconstruction_loss(
            model, x_input, targets, z_latent=mu, is_latent_model=True
        )
    else:
        loss_per_participant = compute_reconstruction_loss(
            model, x_input, targets, is_latent_model=False
        )
    return loss_per_participant.mean().item(), loss_per_participant.cpu().numpy()


# ── Decoding helpers ─────────────────────────────────────────────────────

def loocv_balanced_accuracy(X, y):
    """Fast LOOCV logistic regression. Returns balanced accuracy."""
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(X):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000)
        clf.fit(sc.transform(X[tr]), y[tr])
        preds[te] = clf.predict(sc.transform(X[te]))
    return balanced_accuracy_score(y, preds), preds


def logistic_regression_multiclass(X, y, inner_cv=5):
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    K = len(classes)
    loo = LeaveOneOut()
    preds = np.empty(len(y), dtype=classes.dtype)
    probs = np.zeros((len(y), K), dtype=float)
    for tr, te in tqdm(loo.split(X), total=len(y), desc="LOOCV"):
        y_tr = y[tr]
        min_class = np.min(np.bincount(y_tr))
        if min_class < 2:
            raise ValueError("Not enough samples in the smallest class for inner CV.")
        n_splits = min(inner_cv, min_class)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        clf = LogisticRegressionCV(
            penalty="l2", solver="lbfgs", multi_class="multinomial",
            max_iter=2000, cv=cv,
        )
        clf.fit(X[tr], y_tr)
        preds[te] = clf.predict(X[te])
        probs[te, :] = clf.predict_proba(X[te])[0]
    bal_acc = balanced_accuracy_score(y, preds)
    auc = roc_auc_score(y, probs, multi_class="ovr", average="macro")
    return bal_acc, preds


# ── Data loading ─────────────────────────────────────────────────────────

def load_group_labels():
    """Load children/adults group labels from test data."""
    df_test = pd.read_csv(f"{DATA_DIR}/df_test.csv")
    df_unique = df_test.drop_duplicates(subset="subid").sort_values("subid").reset_index(drop=True)
    group_map = {"young_child": 0, "old_child": 1, "adult": 2}
    labels = np.array([group_map[g] for g in df_unique["age"].values])
    return labels


def load_eval_data(device="cpu"):
    """
    Load data for NLL-based epoch selection.
    Uses val split if available, falls back to test.
    Returns (xin, c, xin_enc_or_None, split_name).
    """
    for split in ("val", "test"):
        xin_path = os.path.join(DATA_DIR, f"xin_{split}.npy")
        c_path = os.path.join(DATA_DIR, f"c_{split}.npy")
        if os.path.exists(xin_path) and os.path.exists(c_path):
            xin = torch.from_numpy(np.load(xin_path, allow_pickle=True)).float().to(device)
            c = torch.from_numpy(np.load(c_path, allow_pickle=True)).float().to(device)
            enc_path = os.path.join(DATA_DIR, f"xin_enc_{split}.npy")
            xin_enc = (torch.from_numpy(np.load(enc_path)).float().to(device)
                       if os.path.exists(enc_path) else None)
            return xin, c, xin_enc, split
    raise RuntimeError(f"No val or test data found in {DATA_DIR}")


def load_test_data(device="cpu"):
    """Load test inputs/targets for decoding evaluation."""
    xin_test = np.load(f"{DATA_DIR}/xin_test.npy", allow_pickle=True)
    c_test = np.load(f"{DATA_DIR}/c_test.npy", allow_pickle=True)
    return (torch.from_numpy(xin_test).float().to(device),
            torch.from_numpy(c_test).float().to(device))


# ── Model helpers ────────────────────────────────────────────────────────

def extract_idrnn_latents(model, xin_test):
    """Extract IDRNN latents via encoder forward pass (last timestep mu)."""
    model.eval()
    with torch.no_grad():
        xenc = xin_test.unsqueeze(1)                       # (B, 1, T, in_dim)
        mu, _ = model.encoder(xenc, return_per_timestep=True)  # (B, 1, T, z_dim)
        lat = mu.squeeze(1)[:, -1, :].cpu().numpy()        # (B, z_dim)
    return lat


def extract_vanilla_latents(model, xin_test):
    """Extract Vanilla RNN latents (time-averaged hidden states)."""
    model.eval()
    with torch.no_grad():
        _, _, hidden_states = model(xin_test)              # (B, T, hidden)
        lat = hidden_states.mean(dim=1).cpu().numpy()      # (B, hidden)
    return lat


def load_idrnn_model(run_dir, epoch, xin_test, device="cpu", combo=None):
    """Load an IDRNN model from a run directory at a specific epoch."""
    B = xin_test.shape[0]
    in_dim = xin_test.shape[2]

    frozen_decoder_path = os.path.join(run_dir, "frozen_decoder", "policy_model.pt")
    ckpt_path = os.path.join(run_dir, "checkpoints", f"epoch{epoch:04d}.pt")

    if not os.path.exists(ckpt_path) or not os.path.exists(frozen_decoder_path):
        return None

    config = load_model_config(run_dir)

    if config is not None:
        model = create_model_from_config(
            config, n_participants=B, device=device,
            frozen_decoder_path=frozen_decoder_path
        )
    else:
        # Fallback for runs without config.json — use combo params if available
        hidden = combo["hidden"] if combo is not None else 8
        enc_hidden = combo["enc_hidden"] if combo is not None else hidden
        z_dim = combo["z_dim"] if combo is not None else 3
        A = 4

        temp_encoder = LookupEncoderZ(n_participants=B, z_dim=z_dim)
        temp_decoder = Decoder(in_dim=in_dim, z_dim=z_dim, hid=hidden, A=A)
        temp_model = LatentRNNz(
            encoder=temp_encoder, decoder=temp_decoder,
            hid=hidden, z_dim=z_dim, in_dim=in_dim, A=A, block_structure=False
        )
        temp_model.load_state_dict(
            torch.load(frozen_decoder_path, map_location=device), strict=False
        )
        encoder_IDRNN = IDRNN(in_dim=in_dim, z_dim=z_dim, hid=enc_hidden)
        frozen_decoder = copy.deepcopy(temp_model.decoder)
        for p in frozen_decoder.parameters():
            p.requires_grad = False
        model = LatentRNN_secondstep(
            encoder=encoder_IDRNN, hid=hidden, z_dim=z_dim,
            in_dim=in_dim, A=A, decoder=frozen_decoder
        ).to(device)

    model = load_model_checkpoint(ckpt_path, model, device)
    return model


def load_vanilla_baseline(xin_test, labels, device="cpu"):
    """Load vanilla baseline and compute its decoding accuracy."""
    loss_path = os.path.join(VANILLA_BASE, "best_epoch_by_nll.json")
    if not os.path.exists(loss_path):
        tensor_path = f"{DATA_DIR}/latents_tensorvanilla.pt"
        if os.path.exists(tensor_path):
            lat_raw = torch.load(tensor_path, map_location="cpu").numpy()
            lat_vanilla = lat_raw.mean(axis=1)
            ba, _ = logistic_regression_multiclass(lat_vanilla, labels)
            return ba, lat_vanilla
        return None, None

    with open(loss_path) as f:
        meta = json.load(f)
    best_epoch = meta["best_epoch"]
    best_seed = meta["best_seed"]

    cfg_path = os.path.join(VANILLA_BASE, f"seed_{best_seed}", "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    model = AblatedRNN(
        hid=cfg["hidden"], in_dim=cfg["in_dim"], A=cfg["A"], block_structure=False
    ).to(device)
    ckpt_path = os.path.join(
        VANILLA_BASE, f"seed_{best_seed}", "checkpoints", f"epoch{best_epoch:04d}.pt"
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    lat_vanilla = extract_vanilla_latents(model, xin_test)
    ba, _ = logistic_regression_multiclass(lat_vanilla, labels)
    print(f"  Vanilla: seed={best_seed}, epoch={best_epoch}, bal-acc={ba:.3f}")
    return ba, lat_vanilla


# ── Discovery ────────────────────────────────────────────────────────────

def discover_combos():
    """
    Auto-discover HP combos from hp_search_runs_sloutsky directory.
    Directory format: lmbd_X_z_Y_h_H_eh_EH
    """
    combos = []
    if not os.path.exists(HP_RUNS_BASE):
        return combos
    for d in sorted(os.listdir(HP_RUNS_BASE)):
        m = re.match(r"lmbd_([^_]+)_z_(\d+)_h_(\d+)_eh_(\d+)$", d)
        if m:
            combos.append({
                "lmbd": float(m.group(1)),
                "z_dim": int(m.group(2)),
                "hidden": int(m.group(3)),
                "enc_hidden": int(m.group(4)),
                "dir": d,
            })
    return combos


def discover_seeds(combo_dir):
    """Auto-detect available seeds from a combo run directory."""
    seeds = []
    if not os.path.exists(combo_dir):
        return seeds
    for d in os.listdir(combo_dir):
        if d.startswith("seed_"):
            try:
                seeds.append(int(d.split("_")[1]))
            except ValueError:
                pass
    return sorted(seeds)


def list_checkpoint_epochs(run_dir):
    """List available checkpoint epochs for a seed run."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return []
    epochs = []
    for f in os.listdir(ckpt_dir):
        if f.startswith("epoch") and f.endswith(".pt"):
            try:
                epochs.append(int(f.replace("epoch", "").replace(".pt", "")))
            except ValueError:
                pass
    return sorted(epochs)


# ── Main evaluation ──────────────────────────────────────────────────────

def evaluate_combo(combo, xin_eval, c_eval, xin_enc_eval, xin_test, labels,
                   nll_window, device="cpu"):
    """
    Evaluate one HP combo:
    1. Compute NLL at each epoch (within nll_window) to select best epoch + seed
    2. Compute decoding balanced accuracy at NLL-selected epoch
    3. Report ceiling — best decoding across sampled epochs
    """
    combo_dir = os.path.join(HP_RUNS_BASE, combo["dir"])
    seeds = discover_seeds(combo_dir)
    if not seeds:
        return None

    min_ep, max_ep = nll_window

    results = {
        "lmbd": combo["lmbd"],
        "z_dim": combo["z_dim"],
        "hidden": combo["hidden"],
        "enc_hidden": combo["enc_hidden"],
        "seeds": seeds,
    }

    # ── Phase 1: NLL-based epoch selection ──────────────────────────────
    nll_by_epoch = {}  # epoch -> {seed: nll}

    for seed in seeds:
        run_dir = os.path.join(combo_dir, f"seed_{seed}")
        available_epochs = list_checkpoint_epochs(run_dir)
        epochs_in_window = [e for e in available_epochs if min_ep <= e <= max_ep]

        frozen_decoder_path = os.path.join(run_dir, "frozen_decoder", "policy_model.pt")
        if not os.path.exists(frozen_decoder_path):
            print(f"    Warning: no frozen decoder for seed {seed}, skipping")
            continue

        for epoch in epochs_in_window:
            model = load_idrnn_model(run_dir, epoch, xin_eval, device, combo=combo)
            if model is None:
                continue
            try:
                nll, _ = compute_nll(
                    model, xin_eval, c_eval,
                    is_latent_model=True, device=device, x_enc_input=xin_enc_eval
                )
                if epoch not in nll_by_epoch:
                    nll_by_epoch[epoch] = {}
                nll_by_epoch[epoch][seed] = nll
            except Exception as e:
                print(f"    Warning: NLL failed epoch {epoch} seed {seed}: {e}")

    if not nll_by_epoch:
        print("    No valid epochs in NLL window")
        return None

    # Mean NLL across seeds per epoch → best = lowest
    mean_nll = {
        epoch: np.mean(list(seed_nlls.values()))
        for epoch, seed_nlls in nll_by_epoch.items()
        if seed_nlls
    }
    best_nll_epoch = min(mean_nll, key=mean_nll.get)
    best_nll_value = mean_nll[best_nll_epoch]

    # Best seed at that epoch = lowest NLL
    best_nll_seed = min(nll_by_epoch[best_nll_epoch],
                        key=nll_by_epoch[best_nll_epoch].get)

    results["nll_best_epoch"] = best_nll_epoch
    results["nll_best_seed"] = best_nll_seed
    results["nll_best_value"] = best_nll_value
    results["nll_by_epoch"] = {str(e): float(v) for e, v in mean_nll.items()}

    # ── Phase 2: decoding at NLL-selected epoch ──────────────────────────
    # Best seed at best epoch
    run_dir = os.path.join(combo_dir, f"seed_{best_nll_seed}")
    model = load_idrnn_model(run_dir, best_nll_epoch, xin_test, device, combo=combo)
    if model is not None:
        lat = extract_idrnn_latents(model, xin_test)
        ba_best, _ = logistic_regression_multiclass(lat, labels)
        results["ba_at_nll_epoch"] = ba_best
    else:
        results["ba_at_nll_epoch"] = 0.0

    # Mean decoding across seeds at the NLL-best epoch
    ba_seeds = []
    for seed in seeds:
        run_dir = os.path.join(combo_dir, f"seed_{seed}")
        model = load_idrnn_model(run_dir, best_nll_epoch, xin_test, device, combo=combo)
        if model is not None:
            lat = extract_idrnn_latents(model, xin_test)
            ba, _ = logistic_regression_multiclass(lat, labels)
            ba_seeds.append(ba)
    results["ba_mean_at_nll_epoch"] = float(np.mean(ba_seeds)) if ba_seeds else 0.0

    # ── Phase 3: ceiling — best decoding across sampled epochs ───────────
    best_ba_any = 0.0
    best_ba_epoch = None
    best_ba_seed = None
    decoding_by_epoch = {}

    for seed in seeds:
        run_dir = os.path.join(combo_dir, f"seed_{seed}")
        available_epochs = list_checkpoint_epochs(run_dir)
        # Sample every 500 epochs + always include the NLL-selected epoch
        sample_epochs = [e for e in available_epochs
                         if e % 500 == 0 or e == best_nll_epoch]

        for epoch in sample_epochs:
            model = load_idrnn_model(run_dir, epoch, xin_test, device, combo=combo)
            if model is None:
                continue
            lat = extract_idrnn_latents(model, xin_test)
            ba, _ = logistic_regression_multiclass(lat, labels)

            key = str(epoch)
            if key not in decoding_by_epoch:
                decoding_by_epoch[key] = {}
            decoding_by_epoch[key][seed] = ba

            if ba > best_ba_any:
                best_ba_any = ba
                best_ba_epoch = epoch
                best_ba_seed = seed

    results["ceiling_ba"] = best_ba_any
    results["ceiling_epoch"] = best_ba_epoch
    results["ceiling_seed"] = best_ba_seed
    results["decoding_by_epoch"] = decoding_by_epoch

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate group decoding for IDRNN HP search (NLL-aligned)"
    )
    parser.add_argument("--lmbd", type=float, default=None,
                        help="Filter to single lambda value")
    parser.add_argument("--z", type=int, default=None,
                        help="Filter to single z_dim")
    parser.add_argument("--hidden", type=int, default=None,
                        help="Filter to single common-process hidden size")
    parser.add_argument("--enc_hidden", type=int, default=None,
                        help="Filter to single IDRNN encoder hidden size")
    parser.add_argument("--nll_min", type=int, default=NLL_EPOCH_WINDOW[0],
                        help="Min epoch for NLL-based selection")
    parser.add_argument("--nll_max", type=int, default=NLL_EPOCH_WINDOW[1],
                        help="Max epoch for NLL-based selection")
    parser.add_argument("--output", type=str, default="hp_decoding_results_sloutsky.json",
                        help="Output JSON file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nll_window = (args.nll_min, args.nll_max)

    print("Loading data...")
    labels = load_group_labels()
    # eval data for NLL-based selection (val preferred, test fallback)
    xin_eval, c_eval, xin_enc_eval, eval_split = load_eval_data(device)
    # test data for decoding
    xin_test, c_test = load_test_data(device)
    print(f"  Eval split for NLL: {eval_split} ({xin_eval.shape[0]} sequences)")
    print(f"  Test split for decoding: {xin_test.shape[0]} sequences, {len(labels)} labels")

    # Vanilla baseline
    print("\nLoading Vanilla baseline...")
    ba_vanilla, _ = load_vanilla_baseline(xin_test, labels, device)
    if ba_vanilla is None:
        print("  WARNING: Could not load Vanilla baseline")
        ba_vanilla = float("nan")

    # Discover and optionally filter combos
    all_combos = discover_combos()
    if not all_combos:
        print(f"ERROR: No HP combos found in {HP_RUNS_BASE}/")
        sys.exit(1)

    if args.lmbd is not None:
        all_combos = [c for c in all_combos if abs(c["lmbd"] - args.lmbd) < 1e-6]
    if args.z is not None:
        all_combos = [c for c in all_combos if c["z_dim"] == args.z]
    if args.hidden is not None:
        all_combos = [c for c in all_combos if c["hidden"] == args.hidden]
    if args.enc_hidden is not None:
        all_combos = [c for c in all_combos if c["enc_hidden"] == args.enc_hidden]

    print(f"\nEvaluating {len(all_combos)} HP combos ...")
    print(f"NLL epoch window: [{nll_window[0]}, {nll_window[1]}]")

    all_results = []
    for i, combo in enumerate(all_combos):
        tag = (f"z={combo['z_dim']} λ={combo['lmbd']} "
               f"h={combo['hidden']} eh={combo['enc_hidden']}")
        print(f"\n[{i+1}/{len(all_combos)}] {tag}")
        res = evaluate_combo(
            combo, xin_eval, c_eval, xin_enc_eval,
            xin_test, labels, nll_window, device
        )
        if res is None:
            continue

        res["vanilla_ba"] = ba_vanilla
        res["delta_ba"] = res["ba_at_nll_epoch"] - ba_vanilla
        res["delta_ba_mean"] = res["ba_mean_at_nll_epoch"] - ba_vanilla
        res["delta_ceiling"] = res["ceiling_ba"] - ba_vanilla
        all_results.append(res)

        print(f"  NLL -> epoch={res['nll_best_epoch']}, seed={res['nll_best_seed']}, "
              f"NLL={res['nll_best_value']:.4f}")
        print(f"  Decoding at NLL-epoch: ba={res['ba_at_nll_epoch']:.3f} "
              f"(mean={res['ba_mean_at_nll_epoch']:.3f})  "
              f"delta={res['delta_ba']:+.3f}")
        print(f"  Ceiling: ba={res['ceiling_ba']:.3f} at epoch={res['ceiling_epoch']} "
              f"seed={res['ceiling_seed']}  delta={res['delta_ceiling']:+.3f}")

    # Sort by decoding at NLL-selected epoch
    all_results.sort(key=lambda r: r["ba_at_nll_epoch"], reverse=True)

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("HYPERPARAMETER SEARCH – GROUP DECODING (NLL-aligned epoch selection)")
    if not np.isnan(ba_vanilla):
        print(f"Vanilla baseline bal-acc = {ba_vanilla:.3f}")
    print(f"NLL epoch window: [{nll_window[0]}, {nll_window[1]}]  |  eval split: {eval_split}")
    print("=" * 120)
    header = (f"{'Rank':>4}  {'z':>2}  {'lmbd':>6}  {'hid':>4}  {'ehid':>4}  "
              f"{'nll_ep':>6}  {'nll':>7}  "
              f"{'ba_nll':>6}  {'ba_mn':>6}  {'d_van':>6}  "
              f"{'ceil':>6}  {'c_ep':>5}  {'d_ceil':>6}")
    print(header)
    print("-" * 120)

    for rank, res in enumerate(all_results, 1):
        wins = " <<<" if res["delta_ba"] > 0 else ""
        print(f"{rank:4d}  {res['z_dim']:2d}  {res['lmbd']:6.3f}  "
              f"{res['hidden']:4d}  {res['enc_hidden']:4d}  "
              f"{res['nll_best_epoch']:6d}  {res['nll_best_value']:7.4f}  "
              f"{res['ba_at_nll_epoch']:6.3f}  {res['ba_mean_at_nll_epoch']:6.3f}  "
              f"{res['delta_ba']:+6.3f}{wins}  "
              f"{res['ceiling_ba']:6.3f}  {str(res['ceiling_epoch']):>5}  "
              f"{res['delta_ceiling']:+6.3f}")

    print("=" * 120)

    n_wins_nll  = sum(1 for r in all_results if r["delta_ba"] > 0)
    n_wins_mean = sum(1 for r in all_results if r["delta_ba_mean"] > 0)
    n_wins_ceil = sum(1 for r in all_results if r["delta_ceiling"] > 0)
    print(f"\nIDRNN > Vanilla at NLL-epoch (best seed): {n_wins_nll}/{len(all_results)}")
    print(f"IDRNN > Vanilla at NLL-epoch (mean seeds): {n_wins_mean}/{len(all_results)}")
    print(f"IDRNN > Vanilla ceiling (any epoch/seed):  {n_wins_ceil}/{len(all_results)}")

    if all_results:
        best = all_results[0]
        print(f"\nBest combo: z={best['z_dim']}, lmbd={best['lmbd']}, "
              f"hidden={best['hidden']}, enc_hidden={best['enc_hidden']}")
        print(f"  NLL selected: epoch={best['nll_best_epoch']}, seed={best['nll_best_seed']}")
        print(f"  IDRNN ba={best['ba_at_nll_epoch']:.3f} vs Vanilla ba={ba_vanilla:.3f} "
              f"(delta={best['delta_ba']:+.3f})")
        if best["delta_ba"] <= 0 and best["delta_ceiling"] > 0:
            print(f"\n  NOTE: NLL selects a suboptimal epoch for decoding!")
            print(f"  Ceiling ba={best['ceiling_ba']:.3f} at epoch={best['ceiling_epoch']} "
                  f"would beat Vanilla (delta={best['delta_ceiling']:+.3f})")

    # Save full results
    output = {
        "vanilla_ba": ba_vanilla,
        "eval_split": eval_split,
        "nll_window": list(nll_window),
        "n_combos": len(all_results),
        "results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
