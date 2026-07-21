"""
train_synthetic_step1.py — Step-1 IDRNN (LatentRNNz with lookup embeddings)
+ Vanilla baseline (AblatedRNN) on a synthetic uniform-α dataset.

Saves output in the same format as plots_thalmann/step1_vs_vanilla/ so the
downstream cross-regression analysis can ingest it directly:
  plots_dataset{id}/step1_vs_vanilla/latents_idrnn_step1_bestseed{S}.pt
  plots_dataset{id}/step1_vs_vanilla/latents_vanilla_bestseed{S}.pt

Best seed picked by reconstruction specificity (IDRNN) and lowest training
NLL (Vanilla), mirroring train_and_decode_thalmann_step1.py's selection.

Single-task, single-block setting (block_structure=False), no task embedding.
"""
import os, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelsandtraining import (
    LookupEncoderZ, LatentRNNz, AblatedRNN, Decoder,
)


# Config — matches existing dataset0 step-2 setup (z_dim=3, hidden=10, in_dim=4)
Z_DIM       = 3
HIDDEN      = 10
A           = 2
BASE_IN_DIM = 4
SEEDS       = [12, 50, 76, 100, 142]
LR          = 1e-3
WD          = 1e-4
N_MISMATCH  = 5


def compute_specificity(model, ids, xin, c, device, n_mismatch=N_MISMATCH, rng_seed=0):
    """Reconstruction specificity for single-task no-blocks data.
    matched   = NLL with each session's own lookup z
    mismatched = NLL with a derangement of z (no fixed points)
    Higher mismatched - matched ⇒ more participant-specific encoding."""
    model.eval()
    n_subj = xin.shape[0]
    mask   = (c.reshape(-1) >= 0).float()
    denom  = mask.sum().clamp(min=1)
    with torch.no_grad():
        logits_m, _, _, _ = model(ids, xin)
        log_p = F.log_softmax(logits_m.reshape(-1, A), dim=-1)
        chosen = log_p.gather(
            -1, c.reshape(-1).clamp(min=0).long().unsqueeze(-1)
        ).squeeze(-1)
        matched_nll = -(chosen * mask).sum() / denom

        rng = np.random.default_rng(rng_seed)
        mis_vals = []
        for _ in range(n_mismatch):
            perm = rng.permutation(n_subj)
            same = perm == np.arange(n_subj)
            while same.any():
                perm[same] = (perm[same] + 1) % n_subj
                same = perm == np.arange(n_subj)
            perm_t = torch.as_tensor(perm, device=device, dtype=torch.long)
            z_mis  = model.encoder(perm_t)
            logits_mis, _ = model.decoder(xin, z_mis)
            log_p_mis = F.log_softmax(logits_mis.reshape(-1, A), dim=-1)
            chosen_mis = log_p_mis.gather(
                -1, c.reshape(-1).clamp(min=0).long().unsqueeze(-1)
            ).squeeze(-1)
            mis_vals.append(float(-(chosen_mis * mask).sum() / denom))

    return float(np.mean(mis_vals)) - float(matched_nll), \
           float(matched_nll), float(np.mean(mis_vals))


def train_idrnn_one_seed(seed, ids, xin, c, device, n_epochs):
    torch.manual_seed(seed); np.random.seed(seed)
    encoder = LookupEncoderZ(n_participants=xin.shape[0], z_dim=Z_DIM)
    decoder = Decoder(in_dim=BASE_IN_DIM, z_dim=Z_DIM, hid=HIDDEN, A=A)
    model   = LatentRNNz(
        encoder=encoder, decoder=decoder, hid=HIDDEN,
        z_dim=Z_DIM, in_dim=BASE_IN_DIM, A=A, block_structure=False,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    for ep in range(1, n_epochs + 1):
        model.train(); opt.zero_grad()
        logits, _, _, _ = model(ids, xin)
        loss = F.cross_entropy(
            logits.reshape(-1, A), c.reshape(-1), ignore_index=-100,
        )
        loss.backward(); opt.step()
        if ep == 1 or ep % 500 == 0 or ep == n_epochs:
            print(f"  IDRNN seed {seed} ep {ep:>5d}: loss={loss.item():.4f}")
    return model


def train_vanilla_one_seed(seed, xin, c, device, n_epochs):
    torch.manual_seed(seed); np.random.seed(seed)
    model = AblatedRNN(
        hid=HIDDEN, in_dim=BASE_IN_DIM, A=A,
        block_structure=False, n_tasks=None, task_emb_dim=0,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    for ep in range(1, n_epochs + 1):
        model.train(); opt.zero_grad()
        logits, _, _ = model(xin)
        loss = F.cross_entropy(
            logits.reshape(-1, A), c.reshape(-1), ignore_index=-100,
        )
        loss.backward(); opt.step()
        if ep == 1 or ep % 500 == 0 or ep == n_epochs:
            print(f"  Vanilla seed {seed} ep {ep:>5d}: loss={loss.item():.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--dgp", type=str, default=None,
                    help="DGP subdir prefix, e.g. 'norev'. None → "
                         "data_dataset{id}/. Otherwise data_{dgp}_dataset{id}/.")
    args = ap.parse_args()

    if args.dgp:
        DATA_DIR = f"data_{args.dgp}_dataset{args.dataset_id}"
        OUT_DIR  = f"plots_{args.dgp}_dataset{args.dataset_id}/step1_vs_vanilla"
    else:
        DATA_DIR = f"data_dataset{args.dataset_id}"
        OUT_DIR  = f"plots_dataset{args.dataset_id}/step1_vs_vanilla"
    os.makedirs(OUT_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATA_DIR}  →  Output: {OUT_DIR}")

    # ── Load data ─────────────────────────────────────────────────────────────
    xin = torch.from_numpy(
        np.load(f"{DATA_DIR}/xin_train.npy")).float().to(DEVICE)
    c   = torch.from_numpy(
        np.load(f"{DATA_DIR}/c_train.npy")).long().to(DEVICE)
    n_sessions, n_trials, in_dim = xin.shape
    assert in_dim == BASE_IN_DIM
    print(f"Loaded: {n_sessions} sessions × {n_trials} trials × {in_dim} feats")

    ids = torch.arange(n_sessions, device=DEVICE)

    # ── IDRNN: train each seed, pick best by specificity ──────────────────────
    print(f"\n{'='*60}\nTraining IDRNN (LatentRNNz) over {len(SEEDS)} seeds\n{'='*60}")
    best = {"spec": -np.inf, "seed": None, "state": None, "z": None}
    seed_records = {}
    for seed in SEEDS:
        model = train_idrnn_one_seed(seed, ids, xin, c, DEVICE, args.epochs)
        spec, matched, mis = compute_specificity(model, ids, xin, c, DEVICE)
        print(f"IDRNN seed {seed}: spec={spec:.4f}  "
              f"(matched={matched:.4f}, mismatched={mis:.4f})")
        seed_records[str(seed)] = {
            "specificity": spec, "matched_nll": matched, "mismatched_nll": mis,
        }
        if spec > best["spec"]:
            best["spec"]  = spec
            best["seed"]  = seed
            best["state"] = {k: v.detach().cpu()
                             for k, v in model.state_dict().items()}
            best["z"]     = model.encoder.embed.weight.detach().cpu().numpy()

    print(f"\nBest IDRNN seed: {best['seed']}  (spec={best['spec']:.4f})")
    out_path = f"{OUT_DIR}/latents_idrnn_step1_bestseed{best['seed']}.pt"
    torch.save({
        "z":             best["z"],
        "subids":        np.arange(n_sessions),
        "seed":          int(best["seed"]),
        "z_dim":         Z_DIM,
        "hidden":        HIDDEN,
        "task_emb_dim":  0,
        "A":             A,
        "base_in_dim":   BASE_IN_DIM,
        "model_state":   best["state"],
        "dataset_id":    args.dataset_id,
        "specificity":   best["spec"],
        "n_epochs":      args.epochs,
        "seed_records":  seed_records,
    }, out_path)
    print(f"Saved IDRNN → {out_path}")

    # ── Vanilla: train each seed, pick best by training NLL ───────────────────
    print(f"\n{'='*60}\nTraining Vanilla (AblatedRNN) over {len(SEEDS)} seeds\n{'='*60}")
    best_v = {"nll": np.inf, "seed": None, "state": None, "h": None}
    seed_records_v = {}
    for seed in SEEDS:
        model = train_vanilla_one_seed(seed, xin, c, DEVICE, args.epochs)
        model.eval()
        with torch.no_grad():
            logits, _, hidden_tr = model(xin)
            nll = F.cross_entropy(
                logits.reshape(-1, A), c.reshape(-1), ignore_index=-100,
            ).item()
            valid = (c >= 0).float()
            n_valid = valid.sum(dim=1).clamp(min=1)
            h_avg = (hidden_tr * valid.unsqueeze(-1)).sum(dim=1) / n_valid.unsqueeze(-1)
        print(f"Vanilla seed {seed}: nll={nll:.4f}")
        seed_records_v[str(seed)] = {"final_nll": nll}
        if nll < best_v["nll"]:
            best_v["nll"]   = nll
            best_v["seed"]  = seed
            best_v["state"] = {k: v.detach().cpu()
                               for k, v in model.state_dict().items()}
            best_v["h"]     = h_avg.detach().cpu().numpy()

    print(f"\nBest Vanilla seed: {best_v['seed']}  (nll={best_v['nll']:.4f})")
    out_path = f"{OUT_DIR}/latents_vanilla_bestseed{best_v['seed']}.pt"
    torch.save({
        "h":             best_v["h"],
        "subids":        np.arange(n_sessions),
        "seed":          int(best_v["seed"]),
        "hidden":        HIDDEN,
        "task_emb_dim":  0,
        "A":             A,
        "base_in_dim":   BASE_IN_DIM,
        "model_state":   best_v["state"],
        "dataset_id":    args.dataset_id,
        "final_nll":     best_v["nll"],
        "n_epochs":      args.epochs,
        "seed_records":  seed_records_v,
    }, out_path)
    print(f"Saved Vanilla → {out_path}")

    # Write a tiny summary JSON for quick inspection
    with open(f"{OUT_DIR}/training_summary.json", "w") as f:
        json.dump({
            "dataset_id": args.dataset_id,
            "idrnn":   {"best_seed": int(best["seed"]),
                        "best_specificity": float(best["spec"]),
                        "seed_records": seed_records},
            "vanilla": {"best_seed": int(best_v["seed"]),
                        "best_nll": float(best_v["nll"]),
                        "seed_records": seed_records_v},
            "config": {"z_dim": Z_DIM, "hidden": HIDDEN, "A": A,
                       "base_in_dim": BASE_IN_DIM,
                       "n_epochs": args.epochs},
        }, f, indent=2)
    print(f"Saved summary → {OUT_DIR}/training_summary.json")


if __name__ == "__main__":
    main()
