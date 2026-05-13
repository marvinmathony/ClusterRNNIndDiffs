"""
Decode environment condition (low / normal / high) from per-session latents,
comparing vanilla RNN hidden states vs IDRNN test-time encoder mu.

Hypothesis: vanilla latents track environmental variance (since they must
encode reward statistics to predict choice); IDRNN latents (mu) should NOT,
because the encoder is regularised to capture stable inter-individual
differences rather than transient task-state variance.

Per dataset i in {0..N-1} (seed_base = 1 + i*1000):
  - reproduce per-session env labels for the test rollouts (seed_base + 1)
  - load latents_tensorlatentmodel.pt  -> (200, 200, z_dim)
  - load latents_tensorvanilla.pt      -> (200, 200, hid)
  - reduce each session to: (a) trial-mean, (b) last-trial features
  - 5-fold stratified CV multinomial logistic regression -> session-level acc
"""

import argparse
import os
import random as pyrandom
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def reproduce_env_labels(seed, T=200, N=200):
    """Mirror the RNG sequence in sim_Q_data.gen_reward_seq to recover env labels."""
    np.random.seed(seed)
    pyrandom.seed(seed)
    np.random.dirichlet([1, 1, 1])
    envs = ["low", "normal", "high"]
    labels = []
    for _ in range(N):
        sel = np.random.choice(envs)
        labels.append(sel)
        if sel == "low":
            pyrandom.uniform(0.1, 0.3)
            pyrandom.uniform(0.05, 0.15)
        elif sel == "high":
            pyrandom.uniform(0.7, 0.95)
            pyrandom.uniform(0.2, 0.4)
        np.random.rand(2 * T)
    return np.array(labels)


def session_features(latent_tensor, mode):
    """
    latent_tensor: (B, T, D) torch or np
    mode: 'mean' | 'last' | 'meanstd'
    Returns (B, F) np.float32
    """
    x = latent_tensor.detach().cpu().numpy() if torch.is_tensor(latent_tensor) else latent_tensor
    if mode == "mean":
        return x.mean(axis=1)
    if mode == "last":
        return x[:, -1, :]
    if mode == "meanstd":
        return np.concatenate([x.mean(axis=1), x.std(axis=1)], axis=1)
    raise ValueError(mode)


def cv_decode(X, y, n_splits=5, seed=0, C=1.0, pca_dim=None):
    """If pca_dim is given, fit PCA on the train fold and reduce to that many dims."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, all_pred, all_true = [], [], []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])
        if pca_dim is not None and pca_dim < Xtr.shape[1]:
            pca = PCA(n_components=pca_dim, random_state=seed).fit(Xtr)
            Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        accs.append(accuracy_score(y[te], pred))
        all_pred.append(pred)
        all_true.append(y[te])
    return np.array(accs), np.concatenate(all_true), np.concatenate(all_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", type=int, default=list(range(10)))
    ap.add_argument("--feature", choices=["mean", "last", "meanstd"], default="mean")
    ap.add_argument("--match_dim", action="store_true",
                    help="Also run vanilla restricted to z_dim PCs (matched dimensionality control)")
    ap.add_argument("--cv_seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="env_decoding_results.csv")
    ap.add_argument("--data_prefix", type=str, default="data_dataset",
                    help="Prefix for dataset folder; full path = {prefix}{id}")
    args = ap.parse_args()

    rows = []
    cm_records = {}

    for did in args.datasets:
        ddir = Path(f"{args.data_prefix}{did}")
        if not ddir.exists():
            print(f"[skip] {ddir} not found")
            continue
        seed_base = 1 + did * 1000
        # latents are computed on TEST rollouts (rewards seed = seed_base + 1)
        env_test = reproduce_env_labels(seed_base + 1)
        env_train = reproduce_env_labels(seed_base)

        idrnn_p   = ddir / "latents_tensorlatentmodel.pt"
        vanilla_p = ddir / "latents_tensorvanilla.pt"
        if not (idrnn_p.exists() and vanilla_p.exists()):
            print(f"[skip] missing latents in {ddir}")
            continue

        idrnn_lat   = torch.load(idrnn_p,   map_location="cpu", weights_only=False)
        vanilla_lat = torch.load(vanilla_p, map_location="cpu", weights_only=False)

        # Reshape if 4-D block format leaks through
        def ensure_3d(t):
            if t.dim() == 4:
                B, Bk, T, D = t.shape
                t = t.view(B, Bk * T, D)
            return t
        idrnn_lat   = ensure_3d(idrnn_lat)
        vanilla_lat = ensure_3d(vanilla_lat)

        z_dim = idrnn_lat.shape[-1]
        runs = [("IDRNN_mu", idrnn_lat, None), ("vanilla_h", vanilla_lat, None)]
        if args.match_dim:
            runs.append(("vanilla_h_PCA{}".format(z_dim), vanilla_lat, z_dim))
        for name, lat, pca_dim in runs:
            X = session_features(lat, args.feature)
            y = env_test
            accs, yt, yp = cv_decode(X, y, seed=args.cv_seed, pca_dim=pca_dim)
            rows.append({
                "dataset_id": did, "model": name, "feature": args.feature,
                "n_features": X.shape[1] if pca_dim is None else pca_dim,
                "z_dim": lat.shape[-1],
                "acc_mean": accs.mean(), "acc_std": accs.std(),
                "acc_folds": ";".join(f"{a:.3f}" for a in accs),
                "n_low": int((y == "low").sum()),
                "n_normal": int((y == "normal").sum()),
                "n_high": int((y == "high").sum()),
            })
            cm_records[(did, name)] = confusion_matrix(yt, yp, labels=["low", "normal", "high"])
            print(f"  ds{did:2d} {name:18s}  acc = {accs.mean():.3f} ± {accs.std():.3f}  (folds: {accs})")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\n=== Summary across datasets ===")
    print(df.to_string(index=False))

    print("\n=== Mean across datasets ===")
    summ = df.groupby("model")["acc_mean"].agg(["mean", "std", "count"])
    chance = 1 / 3
    print(summ)
    print(f"(chance = {chance:.3f})")

    # Optional: aggregate confusion matrices
    print("\n=== Aggregated confusion matrices (rows = true) ===")
    for model in df["model"].unique():
        cms = [cm_records[k] for k in cm_records if k[1] == model]
        agg = np.sum(cms, axis=0)
        norm = agg / agg.sum(axis=1, keepdims=True)
        print(f"\n[{model}]  counts:")
        print(pd.DataFrame(agg, index=["low", "normal", "high"], columns=["low", "normal", "high"]))
        print(f"[{model}]  row-normalised:")
        print(pd.DataFrame(norm, index=["low", "normal", "high"], columns=["low", "normal", "high"]).round(3))


if __name__ == "__main__":
    main()
