"""analyze_dezfouli_nested_cv.py — panel a/b/c data from the fresh nested CV.

Reads:
  runs_dezfouli_nested_cv_z2/foldF/seed_S/{config.json, checkpoints/}
  runs_vanilla_dezfouli_nested_cv_z2/foldF/seed_S/{config.json, checkpoints/}
  data_dezfouli/foldF/{xin,c,choice_one_hot}_{train,test}.npy + df_*.csv
  data_dezfouli/foldF/{cog_model,cog_model_cp,ill_specified_cp,
                       ill_specified_map}_results.csv
  final_plots/dezfouli_z2_spec/canonical/idrnn/latents_train.npy
  final_plots/dezfouli/canonical/vanilla/latents_train.npy

Produces:
  final_plots/dezfouli_z2_spec/nested_cv_summary.json — per-participant
  NLL arrays per model + diag labels + RSA / LOO-AUC summary stats.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, ".")
from modelsandtraining import IDRNN, Decoder, LatentRNN_secondstep, AblatedRNN
from nested_cv.config import path_tag, FINAL_SEEDS, N_OUTER_FOLDS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DGP = "dezfouli"
DATA_DIR = "data_dezfouli"


# ── Model builders mirroring extract_canonical_latents_dezfouli.py ─────────
def _build_idrnn(cfg):
    mc = cfg["model_config"]
    enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"], hid=mc["enc_hidden"],
                n_tasks=mc.get("n_tasks"), task_emb_dim=mc.get("task_emb_dim", 0))
    dec = Decoder(in_dim=mc["dec_in_dim"], z_dim=mc["z_dim"], hid=mc["hidden"], A=mc["A"])
    m = LatentRNN_secondstep(encoder=enc, hid=mc["hidden"], z_dim=mc["z_dim"],
                              in_dim=mc["dec_in_dim"], A=mc["A"], decoder=dec,
                              n_tasks=mc.get("n_tasks"),
                              task_emb_dim=mc.get("task_emb_dim", 0),
                              reinit_decoder_per_block=mc.get("reinit_decoder_per_block",
                                                              True))
    return m.to(DEVICE)


def _build_vanilla(cfg):
    mc = cfg["model_config"]
    return AblatedRNN(hid=mc["hidden"], in_dim=mc["dec_in_dim"], A=mc["A"],
                       block_structure=True,
                       n_tasks=mc.get("n_tasks"),
                       task_emb_dim=mc.get("task_emb_dim", 0)).to(DEVICE)


def _checkpoint_path(run_dir, cfg):
    """cv_selected_epoch if present; else last epoch checkpoint."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    epoch = cfg.get("cv_selected_epoch")
    if epoch is not None:
        cand = os.path.join(ckpt_dir, f"epoch{epoch:04d}.pt")
        if os.path.exists(cand):
            return cand
    avail = sorted(int(f.replace("epoch", "").replace(".pt", ""))
                   for f in os.listdir(ckpt_dir)
                   if f.startswith("epoch") and f.endswith(".pt")
                   and "pre_step25" not in f)
    if not avail:
        return None
    return os.path.join(ckpt_dir, f"epoch{avail[-1]:04d}.pt")


# ── Per-subject NLL from a forward pass ────────────────────────────────────
@torch.no_grad()
def per_subject_nll(model, xin_t, c_t, arch):
    """xin_t: (N, Bk, T, in_dim), c_t: (N, Bk, T) on DEVICE. Returns (N,) mean
    NLL per valid trial."""
    if arch == "idrnn":
        # LatentRNN_secondstep.forward(xenc, blocks) returns
        # (logits, mu, lv, z, h0_)
        logits, _, _, _, _ = model(xin_t, xin_t)
    else:
        logits, _, _ = model(xin_t)
    probs = F.softmax(logits, dim=-1)
    valid = c_t >= 0
    c_safe = c_t.clone()
    c_safe[~valid] = 0
    chosen = probs.gather(-1, c_safe.long().unsqueeze(-1)).squeeze(-1)
    chosen[~valid] = 1.0
    log_ll = chosen.clamp(min=1e-8).log()
    N = c_t.shape[0]
    log_ll_per_sub = log_ll.reshape(N, -1).sum(dim=1)
    n_valid_sub    = valid.reshape(N, -1).sum(dim=1).clamp(min=1).float()
    return (-log_ll_per_sub / n_valid_sub).cpu().numpy()


# ── Aggregate per-fold per-seed NLL across the 30 final seeds ──────────────
def aggregate_rnn_nll(arch, hypothesis_z, metric, n_seeds, seeds_override=None):
    tag = path_tag(hypothesis_z, metric if arch == "idrnn" else None)
    # Stage_c writes with metric=cv_val_loss (NLL-selected HPs)
    cv_tag = path_tag(hypothesis_z, "cv_val_loss")
    base = f"runs_dezfouli_nested_cv{cv_tag}" if arch == "idrnn" else \
           f"runs_vanilla_dezfouli_nested_cv{cv_tag}"

    seeds = seeds_override if seeds_override is not None else FINAL_SEEDS[:n_seeds]
    # subid -> list of per-fold mean-across-seed NLLs
    per_sub_nll = {}   # keyed by subid (int)

    for fold in range(N_OUTER_FOLDS):
        fold_dir   = f"data_dezfouli/fold{fold}"
        df_test    = pd.read_csv(f"{fold_dir}/df_test.csv")
        test_subids = df_test["subid"].values

        xin_test = torch.from_numpy(
            np.load(f"{fold_dir}/xin_test.npy")
        ).float().to(DEVICE)
        c_test = torch.from_numpy(
            np.load(f"{fold_dir}/c_test.npy")
        ).float().to(DEVICE)

        # Stack per-seed NLLs for this fold's test subjects
        seed_nlls = []
        for s in seeds:
            run_dir = f"{base}/fold{fold}/seed_{s}"
            cfg_p = os.path.join(run_dir, "config.json")
            if not os.path.exists(cfg_p):
                continue
            cfg = json.load(open(cfg_p))
            ckpt = _checkpoint_path(run_dir, cfg)
            if ckpt is None:
                continue
            model = _build_idrnn(cfg) if arch == "idrnn" else _build_vanilla(cfg)
            try:
                model.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
            except Exception as e:
                print(f"  [skip] {run_dir}: load failed ({type(e).__name__})")
                continue
            model.eval()
            try:
                nll = per_subject_nll(model, xin_test, c_test, arch)
            except Exception as e:
                print(f"  [skip] {run_dir}: forward failed ({type(e).__name__}: {e})")
                continue
            seed_nlls.append(nll)
        if not seed_nlls:
            print(f"  [warn] fold {fold} {arch}: no usable seeds")
            continue
        mean_nll = np.mean(np.stack(seed_nlls, axis=0), axis=0)  # (N_fold_test,)
        for sid, v in zip(test_subids, mean_nll):
            per_sub_nll[int(sid)] = float(v)
        print(f"  [{arch} fold{fold}] aggregated {len(seed_nlls)} seeds × "
              f"{len(test_subids)} subjects")
    return per_sub_nll


# ── Cog NLLs from existing per-fold CSVs ───────────────────────────────────
def aggregate_cog_nll(csv_basename):
    """Read per-fold {csv_basename}.csv; key 'normalized_likelihood'. Returns
    {subid -> nll}."""
    out = {}
    for fold in range(N_OUTER_FOLDS):
        p = f"data_dezfouli/fold{fold}/{csv_basename}.csv"
        if not os.path.exists(p):
            print(f"  [warn] missing {p}"); continue
        df = pd.read_csv(p)
        for _, row in df.iterrows():
            out[int(row["subid"])] = float(row["normalized_likelihood"])
    return out


# ── RSA + LOO logistic AUC on canonical latents ────────────────────────────
def diag_decoding_loo_auc(latents, diag_labels):
    """One-vs-rest LOO logistic regression; returns mean per-class AUC."""
    from sklearn.metrics import roc_auc_score
    le = LabelEncoder()
    y = le.fit_transform(diag_labels)
    n_class = len(le.classes_)
    aucs_per_class = []
    # one-vs-rest LOO probabilities
    loo = LeaveOneOut()
    probs = np.zeros((len(y), n_class), dtype=float)
    for train_idx, test_idx in loo.split(latents):
        clf = LogisticRegression(max_iter=2000, multi_class="auto",
                                  class_weight="balanced")
        clf.fit(latents[train_idx], y[train_idx])
        probs[test_idx] = clf.predict_proba(latents[test_idx])
    # ROC-AUC per class
    for c in range(n_class):
        try:
            aucs_per_class.append(float(roc_auc_score((y == c).astype(int),
                                                       probs[:, c])))
        except Exception:
            aucs_per_class.append(np.nan)
    return {
        "classes":   list(map(str, le.classes_)),
        "auc_per_class": aucs_per_class,
        "auc_macro": float(np.nanmean(aucs_per_class)),
    }


def rsa_against_diag(latents, diag_labels):
    """Spearman r between (Euclidean dist between latents) and
    (0/1 categorical mismatch) RDMs."""
    le = LabelEncoder(); y = le.fit_transform(diag_labels)
    N = len(y)
    iu = np.triu_indices(N, k=1)
    lat_d = np.sqrt(((latents[None, :, :] - latents[:, None, :])**2).sum(axis=-1))
    cat_d = (y[None, :] != y[:, None]).astype(float)
    rho, p = spearmanr(lat_d[iu], cat_d[iu])
    return {"spearman_r": float(rho), "spearman_p": float(p)}


def pc_match(h_raw, z_dim):
    """Project to z_dim PC scores (strict dim match for downstream LR /
    distance-based RSA).  Mirrors the synthetic-pipeline convention in
    analyze_synthetic_nested_cv.py where both archs are projected to the
    SAME PCA_N_COMPONENTS so the LR sees identical feature counts.
    If h_raw already has ≤ z_dim columns, return centred unchanged."""
    if h_raw.shape[1] <= z_dim:
        return (h_raw - h_raw.mean(0)).astype(np.float32)
    p = PCA(n_components=z_dim)
    return p.fit_transform(h_raw).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis_z", type=int, default=2)
    ap.add_argument("--metric", default="step1_specificity")
    ap.add_argument("--n_seeds", type=int, default=30)
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds override.")
    args = ap.parse_args()
    seeds_override = ([int(s) for s in args.seeds.split(",")]
                      if args.seeds else None)

    out_dir = f"final_plots/{DGP}{path_tag(args.hypothesis_z, args.metric)}"
    os.makedirs(out_dir, exist_ok=True)

    # ── Map subid → diag from the full df_train (canonical cohort) ────────
    df_train_all = pd.read_csv(f"{DATA_DIR}/df_train.csv")
    sub_to_diag  = dict(zip(df_train_all["subid"].astype(int),
                             df_train_all["diag"].astype(str)))

    # ── Per-participant NLL across all 3 folds × n_seeds ──────────────────
    print("== IDRNN NLL ==")
    idr_nll = aggregate_rnn_nll("idrnn",   args.hypothesis_z,
                                  args.metric, args.n_seeds, seeds_override)
    print("== Vanilla NLL ==")
    van_nll = aggregate_rnn_nll("vanilla", args.hypothesis_z,
                                  args.metric, args.n_seeds, seeds_override)

    # Cog NLLs from existing per-fold CSVs (RNN-independent)
    print("== Cog NLLs ==")
    cog_em   = aggregate_cog_nll("cog_model_results")
    cog_cp   = aggregate_cog_nll("cog_model_cp_results")
    ill_cp   = aggregate_cog_nll("ill_specified_cp_results")
    ill_em   = aggregate_cog_nll("ill_specified_map_results")

    # ── Build long-form per-participant DataFrame ────────────────────────
    all_subs = sorted(set(idr_nll.keys()) | set(van_nll.keys()))
    df_out = []
    for sid in all_subs:
        df_out.append({
            "subid":         sid,
            "diag":          sub_to_diag.get(sid, "Unknown"),
            "nll_idrnn":     idr_nll.get(sid, np.nan),
            "nll_vanilla":   van_nll.get(sid, np.nan),
            "nll_cog_em":    cog_em.get(sid, np.nan),
            "nll_cog_cp":    cog_cp.get(sid, np.nan),
            "nll_ill_cp":    ill_cp.get(sid, np.nan),
            "nll_ill_em":    ill_em.get(sid, np.nan),
        })
    df_out = pd.DataFrame(df_out)
    per_sub_csv = os.path.join(out_dir, "per_participant_nll.csv")
    df_out.to_csv(per_sub_csv, index=False)
    print(f"  Per-participant NLL saved -> {per_sub_csv}  (n={len(df_out)})")

    # ── Canonical-latents-based RSA + diag-decoding (panel b/c) ───────────
    rsa_block, dec_block = {}, {}
    # Both IDRNN and Vanilla canonical now live under the {z,metric}-tagged
    # path (Dezfouli; verified on the live run).
    idr_lat_p = f"{out_dir}/canonical/idrnn/latents_train.npy"
    van_lat_p = f"{out_dir}/canonical/vanilla/latents_train.npy"
    if os.path.exists(idr_lat_p):
        idr_lat = np.load(idr_lat_p)
        diag_arr = np.array([sub_to_diag.get(sid, "Unknown")
                              for sid in df_train_all["subid"].astype(int)])
        rsa_block["idrnn"]    = rsa_against_diag(idr_lat, diag_arr)
        dec_block["idrnn"]    = diag_decoding_loo_auc(idr_lat, diag_arr)
        if os.path.exists(van_lat_p):
            van_lat = np.load(van_lat_p)
            van_lat_pc = pc_match(van_lat, idr_lat.shape[1])
            rsa_block["vanilla_h_pc"] = rsa_against_diag(van_lat_pc, diag_arr)
            dec_block["vanilla_h_pc"] = diag_decoding_loo_auc(van_lat_pc, diag_arr)
        # Save per-participant latents alongside for the panel-c scatter
        np.save(os.path.join(out_dir, "latents_idrnn_train.npy"), idr_lat)
    else:
        print(f"  [warn] canonical IDRNN latents not found at {idr_lat_p} — "
              "panel b/c will fall back when missing")

    # ── Summary JSON ──────────────────────────────────────────────────────
    summary = {
        "n_participants":   int(len(df_out)),
        "n_folds":          N_OUTER_FOLDS,
        "n_seeds":          args.n_seeds,
        "metric_for_seed_select": args.metric,
        "hypothesis_z":     args.hypothesis_z,
        "per_participant_csv": per_sub_csv,
        "rsa":              rsa_block,
        "diag_decoding":    dec_block,
        "nll_mean_by_model": {
            "Ill-spec. CP":  float(df_out["nll_ill_cp"].mean()),
            "Ill-spec. EM":  float(df_out["nll_ill_em"].mean()),
            "Cog model CP":  float(df_out["nll_cog_cp"].mean()),
            "Cog model EM":  float(df_out["nll_cog_em"].mean()),
            "IDRNN":         float(df_out["nll_idrnn"].mean()),
            "Vanilla RNN":   float(df_out["nll_vanilla"].mean()),
        },
    }
    json_path = os.path.join(out_dir, "nested_cv_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary -> {json_path}")


if __name__ == "__main__":
    main()
