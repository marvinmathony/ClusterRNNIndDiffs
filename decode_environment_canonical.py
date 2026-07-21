"""Environment-decoding on the canonical (full-data) nested-CV models.

Port of decode_environment_per_seed.py to the canonical z=1 IDRNN + canonical
vanilla models (one model per dataset, seed chosen by stage_g).  Reuses that
script's proven helpers (model loaders, latent extractors, CV decoder, env-label
reproduction) — only the model-path resolution changes.

Dimension-matched by default: every feature is PCA-reduced to z_dim (the IDRNN
bottleneck) before decoding, so all four feature families are compared at equal
capacity.

Output: env_decoding_canonical.csv  (cols mirror env_decoding_per_seed.csv,
with 'seed' = the canonical seed).
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, ".")
from decode_environment_per_seed import (
    reproduce_env_labels, cv_decode, load_idrnn, load_vanilla,
    idrnn_latents, vanilla_latents, common_process_latents, informed_decoder_latents,
)
from nested_cv.config import path_tag


def _canonical_choice(dgp, dataset_id, arch, hypothesis_z, metric):
    cdir = f"final_plots/{dgp}{path_tag(hypothesis_z, metric)}/dataset{dataset_id}/canonical/{arch}"
    # vanilla canonical lives under the plain free-z tag (no z anchor)
    if arch == "vanilla":
        cdir = f"final_plots/{dgp}/dataset{dataset_id}/canonical/{arch}"
    for fn in (f"{arch}_choice.json", f"{arch}_spec.json"):
        p = os.path.join(os.path.dirname(cdir), os.path.basename(cdir), "..", fn)
        p = os.path.normpath(p)
        if os.path.exists(p):
            obj = json.load(open(p))
            seed = obj.get("selected_seed") or (obj.get("seeds") or [None])[0]
            run = obj.get("selected_run_dir") or (
                os.path.join(obj["canonical_runs_dir"], f"seed_{seed}") if seed else None)
            return run, seed
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", type=int, default=[0,1,2,3,5,7,10,12,15,17])
    ap.add_argument("--dgp", default="synthetic")
    ap.add_argument("--hypothesis_z", type=int, default=1)
    ap.add_argument("--metric", default="cv_val_loss")
    ap.add_argument("--feature", choices=["mean","last","meanstd"], default="mean")
    ap.add_argument("--cv_seed", type=int, default=0)
    ap.add_argument("--out", default="env_decoding_canonical.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    rows = []

    for did in args.datasets:
        ddir = Path(f"data_dataset{did}")
        idr_run, idr_seed = _canonical_choice(args.dgp, did, "idrnn",
                                                args.hypothesis_z, args.metric)
        van_run, van_seed = _canonical_choice(args.dgp, did, "vanilla", None, "cv_val_loss")
        if idr_run is None or van_run is None:
            print(f"[skip] ds{did}: canonical models not ready (idr={idr_run}, van={van_run})")
            continue
        idr_cfg = json.load(open(os.path.join(idr_run, "config.json")))
        van_cfg = json.load(open(os.path.join(van_run, "config.json")))

        def _ckpt(run, cfg):
            ep = cfg.get("cv_selected_epoch")
            cdir = os.path.join(run, "checkpoints")
            if ep and os.path.exists(os.path.join(cdir, f"epoch{ep:04d}.pt")):
                return os.path.join(cdir, f"epoch{ep:04d}.pt")
            avail = sorted(int(f[5:-3]) for f in os.listdir(cdir)
                           if f.startswith("epoch") and f.endswith(".pt"))
            return os.path.join(cdir, f"epoch{avail[-1]:04d}.pt")

        xin_train = torch.from_numpy(np.load(ddir/"xin_train.npy")).float().to(device)
        xin_test  = torch.from_numpy(np.load(ddir/"xin_test.npy")).float().to(device)
        env_train = reproduce_env_labels(1 + did*1000)
        env_test  = reproduce_env_labels(1 + did*1000 + 1)

        m  = load_idrnn(idr_cfg, _ckpt(idr_run, idr_cfg), device)
        vm = load_vanilla(van_cfg, _ckpt(van_run, van_cfg), device, block_structure=False)

        feats = {
            "train_data": dict(
                idr=idrnn_latents(m, xin_train), van=vanilla_latents(vm, xin_train),
                cp=common_process_latents(m, xin_train), inf=informed_decoder_latents(m, xin_train),
                env=env_train),
            "test_data": dict(
                idr=idrnn_latents(m, xin_test), van=vanilla_latents(vm, xin_test),
                cp=common_process_latents(m, xin_test), inf=informed_decoder_latents(m, xin_test),
                env=env_test),
        }

        def feat(t):
            x = t.numpy()
            return x.mean(axis=1) if args.feature=="mean" else (
                   x[:,-1,:] if args.feature=="last" else
                   np.concatenate([x.mean(axis=1), x.std(axis=1)], axis=1))

        for split, d in feats.items():
            z_dim = d["idr"].shape[-1]
            # dim-matched: PCA every feature to z_dim (IDRNN already z_dim → no-op)
            runs = [
                ("IDRNN_mu",            feat(d["idr"]), None,  z_dim),
                ("vanilla_h",           feat(d["van"]), z_dim, d["van"].shape[-1]),
                ("common_process_h",    feat(d["cp"]),  z_dim, d["cp"].shape[-1]),
                ("informed_decoder_h",  feat(d["inf"]), z_dim, d["inf"].shape[-1]),
            ]
            for name, X, pca_dim, dim_full in runs:
                accs,_,_ = cv_decode(X, d["env"], seed=args.cv_seed, pca_dim=pca_dim)
                rows.append(dict(dataset_id=did, seed=idr_seed, split=split,
                                 model=name, feature=args.feature,
                                 n_features=(pca_dim or X.shape[1]), latent_dim=dim_full,
                                 acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs))))
        del m, vm
        print(f"  ds{did}: done (idr seed {idr_seed}, van seed {van_seed}, z_dim={z_dim})")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nSaved -> {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
