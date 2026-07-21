#!/usr/bin/env python3
"""Data-scaling / sample-efficiency analysis for Dezfouli diagnosis structure.

Question: does the IDRNN need *less behaviour* to express diagnostic-group
structure than the vanilla RNN?

For the VANILLA RNN the per-subject latent is built online from the behavioural
sequence. With block_structure=True the GRU hidden state RESETS every block, so
"seeing more trials" accumulates evidence by averaging across (within-block)
hidden states. We sweep a trial budget T and form the vanilla representation two
ways from the first T valid trials of each subject's session (blocks concatenated
in order):
   • mean : running average of the per-trial hidden state   (h_avg up to T)
   • last : the hidden state at the T-th valid trial         (h_last at T)

For the IDRNN the subject code is the amortised step-1 lookup embedding z, which
does NOT depend on test-time input — so it is a flat, trial-independent
reference. (Fairness note: z was trained on the subject's *full* behaviour, so
this contrasts vanilla's online code-from-T-trials against IDRNN's
fully-amortised code, not a both-models-retrained-on-T scaling law.)

Both representations are dim-matched + whitened to k = z_dim (identical to
publication panel b'), then scored by:
   • PERMANOVA R^2  (variance in latent distances explained by diagnosis)
   • LOO macro-AUC  (leak-free per-fold PCA -> balanced multinomial LR)
with subject-bootstrap 95% CIs.

Outputs (final_plots/dezfouli_z{Z}_full/publication_figure/):
   datascaling.csv             — per-(metric,model,T) value + CI
   panel_datascaling.{png,pdf} — two-panel curve figure
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from modelsandtraining import AblatedRNN, IDRNN

DATA = "data_dezfouli"
DIAG_ORDER = ["Healthy", "Depression", "Bipolar"]


# ── cohort reconstruction (mirrors train_and_decode_dezfouli_step1.load_all_subjects) ──
def load_cohort():
    tr_df = pd.read_csv(f"{DATA}/fold0/df_train.csv"); te_df = pd.read_csv(f"{DATA}/fold0/df_test.csv")
    tr_x = np.load(f"{DATA}/fold0/xin_train.npy");     te_x = np.load(f"{DATA}/fold0/xin_test.npy")
    tr_c = np.load(f"{DATA}/fold0/c_train.npy");       te_c = np.load(f"{DATA}/fold0/c_test.npy")
    xin = np.concatenate([tr_x, te_x], 0); c = np.concatenate([tr_c, te_c], 0)
    sids = np.concatenate([tr_df["subid"].values, te_df["subid"].values])
    diag = np.concatenate([tr_df["diag"].values, te_df["diag"].values])
    order = np.argsort(sids)
    return xin[order], c[order], sids[order], diag[order]


# ── scoring helpers (identical conventions to the publication notebook) ──
def dim_match_whiten(X_raw, k):
    X = np.asarray(X_raw, float); D = X.shape[1]
    Z = PCA(n_components=k, whiten=True).fit_transform(X) if D > k else X.copy()
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-12)


def _sqdist(X):
    X = np.asarray(X, float)
    return ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)


def permanova_r2(X, g):
    """PERMANOVA R^2 = SS_between / SS_total on squared Euclidean distances."""
    sq = _sqdist(X); n = len(g)
    SS_T = sq[np.triu_indices(n, 1)].sum() / n
    SS_W = 0.0
    for lev in np.unique(g):
        idx = np.where(g == lev)[0]; ng = len(idx)
        if ng > 1:
            SS_W += sq[np.ix_(idx, idx)][np.triu_indices(ng, 1)].sum() / ng
    return float((SS_T - SS_W) / SS_T) if SS_T > 0 else np.nan


def loo_macro_auc(X_raw, k, y, n_class=3):
    """Leak-free LOO macro-AUC: per-fold PCA->k (if D>k), balanced multinomial LR."""
    X = np.asarray(X_raw, float); n, D = X.shape
    P = np.zeros((n, n_class))
    for tr, te in LeaveOneOut().split(X):
        if D > k:
            pca = PCA(n_components=k).fit(X[tr])
            Ztr, Zte = pca.transform(X[tr]), pca.transform(X[te])
        else:
            mu = X[tr].mean(0, keepdims=True); Ztr, Zte = X[tr] - mu, X[te] - mu
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Ztr, y[tr])
        P[te] = clf.predict_proba(Zte)
    aucs = [roc_auc_score((y == c).astype(int), P[:, c]) for c in range(n_class)]
    return float(np.mean(aucs)), P


def boot_ci_permanova(X, y, n_boot, seed):
    rng = np.random.default_rng(seed); n = len(y); vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = permanova_r2(X[idx], y[idx])
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def boot_ci_auc(P, y, n_boot, seed, n_class=3):
    rng = np.random.default_rng(seed); n = len(y); vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals[b] = np.mean([roc_auc_score((y[idx] == c).astype(int), P[idx, c])
                               for c in range(n_class)])
        except ValueError:
            vals[b] = np.nan
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z_dim", type=int, default=1, help="IDRNN z anchor / dim-match k")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--grid", type=str, default="5,10,15,20,30,40,60,80,120,160,220,300,398")
    args = ap.parse_args()
    Z = args.z_dim; k = Z
    base = f"final_plots/dezfouli_z{Z}_full"
    outdir = os.path.join(base, "publication_figure"); os.makedirs(outdir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # labels
    xin, c, sids, diag = load_cohort()
    le = LabelEncoder(); le.classes_ = np.array(DIAG_ORDER); y = le.transform(diag)
    N = len(y)

    valid = (c != -100)
    vf = valid.reshape(N, -1)

    def prep(per_trial):
        """(N, Bk, T, D) -> (per-subject valid per-trial seq in session order, cumsum)."""
        D = per_trial.shape[-1]
        flat = per_trial.reshape(N, -1, D)
        seq = [flat[i, vf[i]] for i in range(N)]
        return seq, [np.cumsum(s, 0) for s in seq]

    # ── vanilla per-timestep hidden (z-independent -> always from z2_full) ──
    vb = torch.load("final_plots/dezfouli_z2_full/canonical/vanilla/latents_vanilla_canonical_full.pt",
                    map_location=dev, weights_only=False)
    vm = AblatedRNN(hid=vb["hidden"], in_dim=vb["base_in_dim"], A=vb["A"], block_structure=True).to(dev)
    vm.load_state_dict(vb["model_state"], strict=False); vm.eval()
    with torch.no_grad():
        _, _, hid = vm(torch.as_tensor(xin, dtype=torch.float32, device=dev))   # (N,Bk,T,H)
    van_seq, van_cs = prep(hid.cpu().numpy())
    n_valid = np.array([s.shape[0] for s in van_seq])
    print(f"valid trials/subject: min={n_valid.min()} med={int(np.median(n_valid))} max={n_valid.max()}")

    # ── IDRNN step-2 ENCODER per-trial mu (test-time latent, distilled from behaviour) ──
    ib = torch.load(os.path.join(base, "canonical/idrnn/latents_idrnn_canonical_full.pt"),
                    map_location=dev, weights_only=False)
    enc_sd = {kk[len("encoder."):]: v for kk, v in ib["model_state_pre_step25"].items()
              if kk.startswith("encoder.")}
    enc_hid = enc_sd["block_encoder.weight_hh_l0"].shape[1]
    enc = IDRNN(in_dim=ib["base_in_dim"], z_dim=ib["z_dim"], hid=enc_hid).to(dev)
    enc.load_state_dict(enc_sd, strict=True); enc.eval()
    with torch.no_grad():
        mu, _ = enc(torch.as_tensor(xin, dtype=torch.float32, device=dev), return_per_timestep=True)
    idr_seq, idr_cs = prep(mu.cpu().numpy())                                    # per-trial mu

    # ── IDRNN step-1 LOOKUP z: amortised, trial-independent flat reference (panel-b' latent) ──
    z_idr = np.load(os.path.join(base, "canonical/idrnn/latents_train_step1.npy"))
    Zi = dim_match_whiten(z_idr, k)
    r2_idr = permanova_r2(Zi, y); r2_idr_ci = boot_ci_permanova(Zi, y, args.n_boot, 101)
    auc_idr, Pi = loo_macro_auc(z_idr, k, y); auc_idr_ci = boot_ci_auc(Pi, y, args.n_boot, 102)
    print(f"IDRNN step-1 lookup (amortised): R2={r2_idr:.3f}  AUC={auc_idr:.3f}")

    # sanity: vanilla full-session R^2 should reproduce panel b'
    h_full = np.stack([van_cs[i][-1] / n_valid[i] for i in range(N)])
    print(f"sanity: vanilla full-session PERMANOVA R^2 = {permanova_r2(dim_match_whiten(h_full, k), y):.3f} (panel b')")

    grid = [int(g) for g in args.grid.split(",")]
    grid = [t for t in grid if t <= int(n_valid.min())]           # common support
    # both models' test-time latents are accumulated by the SAME rule (mean / last over first T)
    SOURCES = [("vanilla", van_seq, van_cs), ("idrnn_step2", idr_seq, idr_cs)]
    rows = []
    for T_ in grid:
        for src, seq, cs in SOURCES:
            reps = {"mean": np.stack([cs[i][T_ - 1] / T_ for i in range(N)]),   # running mean over first T
                    "last": np.stack([seq[i][T_ - 1] for i in range(N)])}       # latent at trial T
            for acc, Hrep in reps.items():
                D = Hrep.shape[1]
                # score at the dim-matched k; for vanilla ALSO at full dimensionality
                score_dims = [("", k)] + ([("_full", D)] if (src == "vanilla" and D > k) else [])
                for suf, kk in score_dims:
                    Zr = dim_match_whiten(Hrep, kk)
                    r2 = permanova_r2(Zr, y); r2lo, r2hi = boot_ci_permanova(Zr, y, args.n_boot, 1000 + T_ + 7 * kk)
                    auc, Pr = loo_macro_auc(Hrep, kk, y); alo, ahi = boot_ci_auc(Pr, y, args.n_boot, 2000 + T_ + 7 * kk)
                    rows.append(dict(T=T_, model=f"{src}_{acc}{suf}", dim=kk, r2=r2, r2_lo=r2lo, r2_hi=r2hi,
                                     auc=auc, auc_lo=alo, auc_hi=ahi))
                    print(f"  T={T_:4d}  {src+'_'+acc+suf:<20} (dim={kk:>2}) R2={r2:.3f} [{r2lo:.3f},{r2hi:.3f}]  AUC={auc:.3f} [{alo:.3f},{ahi:.3f}]")
    df = pd.DataFrame(rows)
    # store the IDRNN step-1 amortised flat references as metadata rows (T = -1)
    df = pd.concat([df, pd.DataFrame([
        dict(T=-1, model="idrnn_step1_amortised", dim=k, r2=r2_idr, r2_lo=r2_idr_ci[0], r2_hi=r2_idr_ci[1],
             auc=auc_idr, auc_lo=auc_idr_ci[0], auc_hi=auc_idr_ci[1])])], ignore_index=True)
    df.to_csv(os.path.join(outdir, "datascaling.csv"), index=False)

    # ── preview figure (polished version lives in dezfouli_publication_panels.ipynb) ──
    from nature_plot_style import nature_colors
    COL_IDR1 = nature_colors["Blue"][3]; COL_IDR2 = nature_colors["Blue"][1]
    COL_VAN = nature_colors["Orange"][3]; COL_VAN_L = nature_colors["Skin tones"][3]
    COL_TRUE = nature_colors["Grey"][5]
    chance_r2 = (len(np.unique(y)) - 1) / (N - 1)
    COL_VAN_F = nature_colors["Skin tones"][1]
    curves = [("vanilla_mean",      COL_VAN,   "-",  "Vanilla h (k=1) — mean"),
              ("vanilla_mean_full", COL_VAN_F, "-",  "Vanilla h (full 15-D) — mean"),
              ("vanilla_last",      COL_VAN_L, "--", "Vanilla h (k=1) — last"),
              ("idrnn_step2_mean",  COL_IDR2,  "-",  "IDRNN enc (test-time) — mean"),
              ("idrnn_step2_last",  COL_IDR2,  "--", "IDRNN enc (test-time) — last")]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    panels = [("r2", "PERMANOVA R$^2$  (diag separation)", r2_idr, r2_idr_ci),
              ("auc", "LOO macro-AUC  (diag decoding)",    auc_idr, auc_idr_ci)]
    for ax, (metric, ylab, idr_val, idr_ci) in zip(axes, panels):
        ch = chance_r2 if metric == "r2" else 0.5
        for tag, col, ls, lab in curves:
            s = df[(df.model == tag) & (df["T"] >= 0)].sort_values("T")
            ax.plot(s["T"], s[metric], ls, marker="o", color=col, ms=3.0, lw=1.4, label=lab, zorder=3)
            ax.fill_between(s["T"], s[f"{metric}_lo"], s[f"{metric}_hi"], color=col, alpha=0.12, zorder=1)
        # IDRNN step-1 lookup amortised flat reference
        ax.axhline(idr_val, color=COL_IDR1, lw=1.6, label="IDRNN z (step-1, amortised)", zorder=3)
        ax.axhspan(idr_ci[0], idr_ci[1], color=COL_IDR1, alpha=0.12, zorder=1)
        ax.axhline(ch, ls=":", color=COL_TRUE, lw=1.0, zorder=1)
        ax.text(df[df["T"] >= 0]["T"].min(), ch, "chance ", ha="left", va="bottom", fontsize=7, color=COL_TRUE)
        ax.set_xlabel("Number of trials seen (T)"); ax.set_ylabel(ylab)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].legend(fontsize=6.5, loc="upper left", frameon=False)
    fig.suptitle(f"Dezfouli data-scaling: diagnosis structure vs trials seen (z={Z}, dim-matched k={k})",
                 fontsize=9, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "panel_datascaling.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "panel_datascaling.pdf"), bbox_inches="tight")
    print(f"saved -> {outdir}/panel_datascaling.png  +  datascaling.csv")


if __name__ == "__main__":
    main()
