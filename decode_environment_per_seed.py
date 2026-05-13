"""
Per-seed environment-decoding analysis.

For each (dataset, seed):
  - Load that seed's checkpoint at the dataset's selected best epoch.
  - Forward train and test inputs through the encoder/RNN to get latents:
      * IDRNN  -> encoder mu, per-trial  (B, T, z_dim)
      * Vanilla-> hidden states          (B, T, hid)
  - 5-fold stratified CV multinomial logistic regression on env labels
    (env labels reproduced from seed_base RNG).
  - Aggregate per-(dataset, seed) accuracies → per-model summary.

Latents are NOT averaged across seeds; each seed produces its own decoding score.
"""

import argparse
import json
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

from modelsandtraining import IDRNN, Decoder, LatentRNN_secondstep, AblatedRNN

SEEDS = [12, 50, 76, 100, 142]


def reproduce_env_labels(seed, T=200, N=200):
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


def cv_decode(X, y, n_splits=5, seed=0, pca_dim=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, all_pred, all_true = [], [], []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])
        if pca_dim is not None and pca_dim < Xtr.shape[1]:
            pca = PCA(n_components=pca_dim, random_state=seed).fit(Xtr)
            Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        accs.append(accuracy_score(y[te], pred))
        all_pred.append(pred)
        all_true.append(y[te])
    return np.array(accs), np.concatenate(all_true), np.concatenate(all_pred)


def load_idrnn(cfg, ckpt_path, device):
    enc_in_dim = cfg.get("model_config", {}).get("enc_in_dim", cfg["in_dim"])
    enc_hidden = cfg.get("model_config", {}).get("enc_hidden", cfg.get("hidden", 8))
    encoder = IDRNN(in_dim=enc_in_dim, z_dim=cfg["z_dim"], hid=enc_hidden)
    decoder = Decoder(in_dim=cfg["in_dim"], z_dim=cfg["z_dim"], hid=cfg["hidden"], A=cfg["A"])
    for p in decoder.parameters():
        p.requires_grad = False
    model = LatentRNN_secondstep(
        encoder=encoder, hid=cfg["hidden"], z_dim=cfg["z_dim"],
        in_dim=cfg["in_dim"], A=cfg["A"], decoder=decoder,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def load_vanilla(cfg, ckpt_path, device, block_structure=False):
    model = AblatedRNN(
        hid=cfg["hidden"], in_dim=cfg["in_dim"], A=cfg["A"],
        block_structure=block_structure,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def idrnn_latents(model, xin):
    """xin: (B, T, in_dim) torch.  Returns (B, T, z_dim)."""
    blocks = xin.unsqueeze(1)  # (B, 1, T, in_dim)
    mu, _ = model.encoder(blocks, return_per_timestep=True)  # (B, 1, T, z_dim)
    return mu.squeeze(1).cpu()


@torch.no_grad()
def vanilla_latents(model, xin):
    """xin: (B, T, in_dim).  AblatedRNN(block_structure=False) returns (logits, final_hid, hidden_traj)."""
    logits, _, hidden_tr = model(xin)
    # Could be (B, T, hid) directly, or (B, 1, T, hid) if block-structured. Squeeze.
    if hidden_tr.dim() == 4:
        hidden_tr = hidden_tr.squeeze(1)
    return hidden_tr.cpu()


@torch.no_grad()
def common_process_latents(model, xin):
    """
    IDRNN decoder run with z=0 → per-trial GRU hidden trajectory.
    This is the 'common process' representation: what the decoder retains about
    the task when stripped of individual identity.
    Returns (B, T, hid_dec).
    """
    B, T, _ = xin.shape
    z_dim = model.decoder.z2h0.in_features
    z = torch.zeros(B, z_dim, device=xin.device)
    zexp = z.unsqueeze(1).expand(-1, T, -1)
    h0 = model.decoder.z2h0(z).unsqueeze(0).contiguous()
    rnn_input = torch.cat([xin, zexp], dim=-1)
    out, _ = model.decoder.rnn(rnn_input, h0)
    return out.cpu()


@torch.no_grad()
def informed_decoder_latents(model, xin):
    """
    IDRNN decoder run with z = encoder μ (per-trial) → GRU hidden trajectory.
    This is the *informed* decoder: it sees both the task input and the
    individual-difference latent. Tests whether adding μ buys env-decoding
    above the z=0 case.
    Returns (B, T, hid_dec).
    """
    blocks = xin.unsqueeze(1)                                     # (B, 1, T, in_dim)
    mu, _ = model.encoder(blocks, return_per_timestep=True)       # (B, 1, T, z_dim)
    mu = mu.squeeze(1)                                            # (B, T, z_dim)
    h0 = model.decoder.z2h0(mu[:, 0]).unsqueeze(0).contiguous()   # match Decoder.forward
    rnn_input = torch.cat([xin, mu], dim=-1)
    out, _ = model.decoder.rnn(rnn_input, h0)
    return out.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", type=int, default=list(range(10)))
    ap.add_argument("--feature", choices=["mean", "last", "meanstd"], default="mean")
    ap.add_argument("--match_dim", action="store_true",
                    help="Also run vanilla restricted to z_dim PCs (matched control)")
    ap.add_argument("--cv_seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="env_decoding_per_seed.csv")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    rows = []

    for did in args.datasets:
        ddir = Path(f"data_dataset{did}")
        idrnn_runs = Path(f"runs_dataset{did}")
        van_runs = Path(f"runs_vanilla_dataset{did}")
        if not (ddir.exists() and idrnn_runs.exists() and van_runs.exists()):
            print(f"[skip] dataset {did} missing dirs")
            continue

        # Selected best epoch (one per dataset & model type)
        idrnn_meta_path = idrnn_runs / "best_epoch_by_rsa.json"
        van_meta_path = van_runs / "best_epoch_by_nll.json"
        if not van_meta_path.exists():
            van_meta_path = van_runs / "best_epoch_by_rsa.json"
        idrnn_best_ep = json.loads(idrnn_meta_path.read_text())["best_epoch"]
        van_best_ep = json.loads(van_meta_path.read_text())["best_epoch"]

        # Inputs
        xin_train = torch.from_numpy(np.load(ddir / "xin_train.npy")).float().to(device)
        xin_test  = torch.from_numpy(np.load(ddir / "xin_test.npy")).float().to(device)

        env_train = reproduce_env_labels(1 + did * 1000)
        env_test  = reproduce_env_labels(1 + did * 1000 + 1)

        for seed in SEEDS:
            # IDRNN
            cfg_p = idrnn_runs / f"seed_{seed}" / "config.json"
            ckpt_p = idrnn_runs / f"seed_{seed}" / "checkpoints" / f"epoch{idrnn_best_ep:04d}.pt"
            if not (cfg_p.exists() and ckpt_p.exists()):
                print(f"[skip] ds{did} seed{seed} IDRNN ckpt missing")
                continue
            cfg = json.loads(cfg_p.read_text())
            try:
                m = load_idrnn(cfg, ckpt_p, device)
                idrnn_train = idrnn_latents(m, xin_train)
                idrnn_test  = idrnn_latents(m, xin_test)
                # decoder hidden trajectory with z=0 (common-process mode)
                cp_train = common_process_latents(m, xin_train)
                cp_test  = common_process_latents(m, xin_test)
                # decoder hidden trajectory with z=μ (informed)
                inf_train = informed_decoder_latents(m, xin_train)
                inf_test  = informed_decoder_latents(m, xin_test)
                del m
            except Exception as e:
                print(f"[err ] ds{did} seed{seed} IDRNN: {e}")
                continue

            # Vanilla
            v_cfg_p = van_runs / f"seed_{seed}" / "config.json"
            v_ckpt_p = van_runs / f"seed_{seed}" / "checkpoints" / f"epoch{van_best_ep:04d}.pt"
            if not (v_cfg_p.exists() and v_ckpt_p.exists()):
                print(f"[skip] ds{did} seed{seed} vanilla ckpt missing")
                continue
            v_cfg = json.loads(v_cfg_p.read_text())
            try:
                vm = load_vanilla(v_cfg, v_ckpt_p, device, block_structure=False)
                van_train = vanilla_latents(vm, xin_train)
                van_test  = vanilla_latents(vm, xin_test)
                del vm
            except Exception as e:
                print(f"[err ] ds{did} seed{seed} vanilla: {e}")
                continue

            # Build features per split (train rollouts vs test rollouts)
            for split, (idr, van, cp, inf, env) in {
                "train_data": (idrnn_train, van_train, cp_train, inf_train, env_train),
                "test_data":  (idrnn_test,  van_test,  cp_test,  inf_test,  env_test),
            }.items():
                z_dim = idr.shape[-1]

                def feat(t):
                    x = t.numpy()
                    if args.feature == "mean":
                        return x.mean(axis=1)
                    if args.feature == "last":
                        return x[:, -1, :]
                    if args.feature == "meanstd":
                        return np.concatenate([x.mean(axis=1), x.std(axis=1)], axis=1)

                runs = [
                    ("IDRNN_mu", feat(idr), None, z_dim),
                    ("vanilla_h", feat(van), None, van.shape[-1]),
                    ("common_process_h", feat(cp), None, cp.shape[-1]),
                    ("informed_decoder_h", feat(inf), None, inf.shape[-1]),
                ]
                if args.match_dim:
                    runs.append(("vanilla_h_PCA{}".format(z_dim),
                                 feat(van), z_dim, van.shape[-1]))
                    runs.append(("common_process_h_PCA{}".format(z_dim),
                                 feat(cp), z_dim, cp.shape[-1]))
                    runs.append(("informed_decoder_h_PCA{}".format(z_dim),
                                 feat(inf), z_dim, inf.shape[-1]))

                for name, X, pca_dim, dim_full in runs:
                    accs, _, _ = cv_decode(X, env, seed=args.cv_seed, pca_dim=pca_dim)
                    rows.append({
                        "dataset_id": did,
                        "seed": seed,
                        "split": split,
                        "model": name,
                        "feature": args.feature,
                        "n_features": X.shape[1] if pca_dim is None else pca_dim,
                        "latent_dim": dim_full,
                        "acc_mean": accs.mean(),
                        "acc_std": accs.std(),
                    })
            print(f"  ds{did:2d} seed{seed:3d} done.")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\n=== Per-seed results saved to", args.out, "===")

    # Aggregate: avg over seeds within (dataset, split, model), then over datasets
    print("\n=== Per-seed accuracies (model × split) ===")
    summ_seed = (df.groupby(["model", "split"])["acc_mean"]
                 .agg(["mean", "std", "count"]).round(4))
    print(summ_seed)
    print("(chance = 1/3)\n")

    print("\n=== Per-dataset means (averaging over seeds) ===")
    per_ds = (df.groupby(["dataset_id", "split", "model"])["acc_mean"]
              .mean().reset_index())
    pivot = per_ds.pivot_table(index=["dataset_id", "split"], columns="model", values="acc_mean")
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
