#!/usr/bin/env python3
"""
Thalmann step-1 decodability grid search.

Training/evaluation logic follows train_and_decode_thalmann_step1.py:
  - Step-1 only: LookupEncoderZ + Decoder (no step-2 distillation)
  - Inner 3-fold CV over task-0 blocks (task-1 always in train)
  - Retrain from scratch on all 31 blocks for selected_epoch epochs
  - N_SEEDS seeds per combo; IDRNN best = argmax specificity,
    Vanilla best = argmin train NLL
  - Decoding: in-sample OLS on ALL 236 subjects per questionnaire scale
  - Significance: Steiger test (valid in-sample since it only needs r(y, preds_I),
    r(y, preds_V), r(preds_I, preds_V), n)

Grid — matched common-process capacity (Vanilla.hidden = IDRNN.decoder.hidden):
  - IDRNN: z_dim ∈ {3, 5, 10} × hidden ∈ {1, 2, 3, 5}   (12 combos)
  - Vanilla:            hidden ∈ {1, 2, 3, 5}           (4 combos)
  - Pairing: IDRNN(z, h) vs Vanilla(h) for Steiger at matched decoder size.
  - z_dim == hidden cells are the matched-PREDICTOR anchors (Steiger bias-free).
  - Primary interest: small hidden, large z_dim — does IDRNN's lookup z
    pick up the slack when the common process is too small?

Outputs (plots_thalmann/step1_decodability_search/):
  seed_summary.csv           — per-seed training stats
  decoding_results.csv       — per-combo × per-scale r, adj R², Steiger
  decodability_vs_hidden.png — mean |r| and mean adj R² vs hidden, lines per z_dim
  per_scale_decodability.png — 2×3 panel, |r| vs hidden per scale

Caches per-combo training in runs_thalmann_step1_decodability_search/ so the
script is resume-friendly.
"""
import os, sys, glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, norm, f as f_dist

sys.path.insert(0, ".")
from modelsandtraining import (
    Decoder, LookupEncoderZ, LatentRNNz, AblatedRNN,
    train_ablated_noblocks_palminteri_CV, train_final_ablated_model_after_cv,
)

# ── Config ─────────────────────────────────────────────────────────────────────
DGP               = "thalmann"
DATA_DIR          = "data_thalmann"
OUT_DIR           = "plots_thalmann/step1_decodability_search"
CKPT_DIR          = "runs_thalmann_step1_decodability_search"

Z_DIM_GRID        = [3, 5, 10]
HIDDEN_GRID       = [7, 10, 15, 20]
TASK_EMB_DIM      = 2
A                 = 4
STEP1_MAX_EPOCHS  = 5000
STEP1_PATIENCE    = 500
VANILLA_MAX_EPOCHS = 5000
N_INNER_FOLDS     = 3
N_SEEDS           = 3
SEEDS             = [42, 56, 85]
LR                = 1e-3
WEIGHT_DECAY      = 1e-4
N_MISMATCH_SAMPLES = 20

SCALES = {
    "PANAS_PA":  ([f"PANAS_{i}"  for i in [0,2,4,6,8,14,16,18]], "PANAS\nPos. Affect"),
    "PANAS_NA":  ([f"PANAS_{i}"  for i in [1,3,5,7,9,11,13,15]], "PANAS\nNeg. Affect"),
    "STICSA":    ([f"STICSA_{i}" for i in range(22)],             "STICSA\nAnxiety"),
    "PHQ":       ([f"PHQ_9_{i}"  for i in range(10)],             "PHQ-9\nDepression"),
    "CEI":       ([f"CEI_{i}"    for i in range(4)],              "CEI\nCuriosity"),
    "BIG5_open": ([f"BIG_5_{i}"  for i in range(6)],              "BIG5\nOpenness"),
    "MotivMem":  (["motiv_mem_0"],                                "Motivation\n(memory)"),
    # CFA compound factors (loaded from CFA_compound_questionnaire_factors_s1.csv)
    "AxDep":     (["AxDep"],                                      "CFA\nAxDep"),
    "posMood":   (["posMood"],                                    "CFA\nposMood"),
    "negMood":   (["negMood"],                                    "CFA\nnegMood"),
    "Exp":       (["Exp"],                                        "CFA\nExploration"),
}
SCALE_KEYS = list(SCALES.keys())

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Data ──────────────────────────────────────────────────────────────────────
def load_all_subjects():
    """Return xin (236,31,200,5), c (236,31,200), subids (236,), task_ids (31,)."""
    f0_tr_df = pd.read_csv(f"{DATA_DIR}/fold0/df_train.csv")
    f0_te_df = pd.read_csv(f"{DATA_DIR}/fold0/df_test.csv")
    tr_xin = np.load(f"{DATA_DIR}/fold0/xin_train.npy")
    te_xin = np.load(f"{DATA_DIR}/fold0/xin_test.npy")
    tr_c   = np.load(f"{DATA_DIR}/fold0/c_train.npy")
    te_c   = np.load(f"{DATA_DIR}/fold0/c_test.npy")
    all_xin  = np.concatenate([tr_xin, te_xin], axis=0)
    all_c    = np.concatenate([tr_c,   te_c],   axis=0)
    all_sids = np.concatenate([f0_tr_df["subid"].values, f0_te_df["subid"].values])
    order = np.argsort(all_sids)
    xin, c, subids = all_xin[order], all_c[order], all_sids[order]
    task_ids = np.load(f"{DATA_DIR}/task_ids_per_block.npy")
    assert xin.shape == (236, 31, 200, 5), f"Unexpected xin shape {xin.shape}"
    return xin, c, subids, task_ids


# ── IDRNN step-1 (parameterised by z_dim + hidden) ────────────────────────────
def _build_step1_model(n_participants, in_dim, z_dim, hidden):
    encoder = LookupEncoderZ(n_participants=n_participants, z_dim=z_dim)
    dec_in_dim = in_dim + TASK_EMB_DIM
    decoder = Decoder(in_dim=dec_in_dim, z_dim=z_dim, hid=hidden, A=A)
    model = LatentRNNz(
        encoder=encoder, decoder=decoder, hid=hidden, z_dim=z_dim,
        in_dim=dec_in_dim, A=A, block_structure=True,
        n_tasks=2, task_emb_dim=TASK_EMB_DIM,
    )
    return model


def _masked_block_nll(logits_all, c_all, block_mask):
    logits_sel = logits_all[:, block_mask]
    c_sel      = c_all[:,      block_mask]
    return F.cross_entropy(
        logits_sel.reshape(-1, A), c_sel.reshape(-1), ignore_index=-100
    )


def train_idrnn_step1_inner_cv(xin_t, c_t, task_ids_tensor, seed, z_dim, hidden):
    n_subj = xin_t.shape[0]
    task0_blocks = np.where(task_ids_tensor.cpu().numpy() == 0)[0]
    task1_blocks = np.where(task_ids_tensor.cpu().numpy() == 1)[0]
    kf = KFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)
    all_val_curves = []

    for fi, (tr_idx, va_idx) in enumerate(kf.split(task0_blocks), start=1):
        tr_blocks = np.concatenate([task0_blocks[tr_idx], task1_blocks])
        va_blocks = task0_blocks[va_idx]
        train_mask = np.zeros(31, dtype=bool); train_mask[tr_blocks] = True
        val_mask   = np.zeros(31, dtype=bool); val_mask[va_blocks]   = True

        torch.manual_seed(seed * 1000 + fi); np.random.seed(seed * 1000 + fi)
        model = _build_step1_model(
            n_subj, in_dim=xin_t.shape[-1], z_dim=z_dim, hidden=hidden
        ).to(device)
        model.set_task_ids(task_ids_tensor)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        ids_all = torch.arange(n_subj, device=device)
        tm_t = torch.tensor(train_mask, device=device)
        vm_t = torch.tensor(val_mask,   device=device)

        val_curve = []; best_val = float("inf"); epochs_no_improve = 0
        for ep in range(1, STEP1_MAX_EPOCHS + 1):
            model.train(); opt.zero_grad()
            logits, _, _ = model(ids_all, xin_t)
            train_nll = _masked_block_nll(logits, c_t, tm_t)
            train_nll.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                logits_ev, _, _ = model(ids_all, xin_t)
                val_nll = _masked_block_nll(logits_ev, c_t, vm_t).item()
            val_curve.append(val_nll)
            if val_nll < best_val - 1e-5:
                best_val = val_nll; epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= STEP1_PATIENCE:
                    break
            if ep % 500 == 0:
                print(f"    [z{z_dim} h{hidden} seed {seed} fold {fi}] ep {ep}  "
                      f"train={train_nll.item():.4f}  val={val_nll:.4f}")
        all_val_curves.append(np.array(val_curve))
        print(f"    [z{z_dim} h{hidden} seed {seed} fold {fi}] stopped ep "
              f"{len(val_curve)}, best val={best_val:.4f}")

    L = min(len(c) for c in all_val_curves)
    mat = np.stack([c[:L] for c in all_val_curves], axis=0)
    mean_val = mat.mean(axis=0)
    selected_epoch = int(np.argmin(mean_val)) + 1
    print(f"  [z{z_dim} h{hidden} seed {seed}] CV epoch = {selected_epoch}  "
          f"(val NLL = {mean_val[selected_epoch-1]:.4f})")
    return selected_epoch, mean_val


def train_idrnn_step1_final(xin_t, c_t, task_ids_tensor, seed, n_epochs, z_dim, hidden):
    n_subj = xin_t.shape[0]
    torch.manual_seed(seed); np.random.seed(seed)
    model = _build_step1_model(
        n_subj, in_dim=xin_t.shape[-1], z_dim=z_dim, hidden=hidden
    ).to(device)
    model.set_task_ids(task_ids_tensor)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ids_all = torch.arange(n_subj, device=device)
    for ep in range(1, n_epochs + 1):
        model.train(); opt.zero_grad()
        logits, _, _ = model(ids_all, xin_t)
        nll = F.cross_entropy(logits.reshape(-1, A), c_t.reshape(-1).long(),
                              ignore_index=-100)
        nll.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(ids_all, xin_t)
        final_nll = F.cross_entropy(logits.reshape(-1, A),
                                    c_t.reshape(-1).long(),
                                    ignore_index=-100).item()
    emb = model.encoder.embed.weight.detach().cpu().numpy()
    return model, emb, final_nll


def compute_specificity_step1(model, xin_t, c_t, n_mismatch=N_MISMATCH_SAMPLES,
                              rng_seed=0):
    model.eval()
    n_subj = xin_t.shape[0]
    ids_all = torch.arange(n_subj, device=device)
    mask = (c_t.reshape(-1) >= 0).float()
    denom = mask.sum().clamp(min=1)
    with torch.no_grad():
        logits_m, _, _ = model(ids_all, xin_t)
        log_p = F.log_softmax(logits_m.reshape(-1, A), dim=-1)
        chosen = log_p.gather(
            -1, c_t.reshape(-1).clamp(min=0).long().unsqueeze(-1)
        ).squeeze(-1)
        matched_nll = -(chosen * mask).sum() / denom

        rng = np.random.default_rng(rng_seed)
        mis_vals = []
        for _ in range(n_mismatch):
            perm_np = rng.permutation(n_subj)
            same = (perm_np == np.arange(n_subj))
            while same.any():
                perm_np[same] = (perm_np[same] + 1) % n_subj
                same = (perm_np == np.arange(n_subj))
            perm = torch.as_tensor(perm_np, device=device, dtype=torch.long)
            z_mis = model.encoder(perm)
            logits_list = []
            for b in range(xin_t.size(1)):
                block_in = model._append_task_emb(xin_t[:, b], b)
                out, _ = model.decoder(block_in, z_mis)
                logits_list.append(out)
            logits_mis = torch.stack(logits_list, dim=1)
            log_p_mis = F.log_softmax(logits_mis.reshape(-1, A), dim=-1)
            chosen_mis = log_p_mis.gather(
                -1, c_t.reshape(-1).clamp(min=0).long().unsqueeze(-1)
            ).squeeze(-1)
            mis_vals.append(float(-(chosen_mis * mask).sum() / denom))

    mismatched_nll = float(np.mean(mis_vals))
    matched_val    = float(matched_nll)
    return mismatched_nll - matched_val, matched_val, mismatched_nll


# ── Vanilla (parameterised by hidden) ────────────────────────────────────────
def train_vanilla_full(xin_t, c_t, task_ids_tensor, seed, hidden):
    torch.manual_seed(seed); np.random.seed(seed)
    in_dim = xin_t.shape[-1]
    dec_in_dim = in_dim + TASK_EMB_DIM

    cv = train_ablated_noblocks_palminteri_CV(
        hidden=hidden, in_dim=in_dim, A=A, X_train=xin_t, y_train=c_t.long(),
        device=device, epoch_nr=VANILLA_MAX_EPOCHS, lr=LR,
        nr_splits=N_INNER_FOLDS,
        dec_in_dim=dec_in_dim, n_tasks=2, task_emb_dim=TASK_EMB_DIM,
        task_ids=task_ids_tensor,
    )
    n_epochs = cv["selected_epoch"]

    torch.manual_seed(seed); np.random.seed(seed)
    model, _, _, _, _, _ = train_final_ablated_model_after_cv(
        hidden=hidden, in_dim=in_dim, A=A, X_train=xin_t, y_train=c_t.long(),
        device=device, n_epochs=n_epochs, lr=LR,
        checkpoint_dir=None, loss_dir=None,
        dec_in_dim=dec_in_dim, n_tasks=2, task_emb_dim=TASK_EMB_DIM,
        task_ids=task_ids_tensor,
    )
    model.eval()
    with torch.no_grad():
        logits, _, hidden_tr = model(xin_t)
        valid = (c_t >= 0).float()
        n_valid = valid.sum(dim=(1, 2)).clamp(min=1)
        h_avg = (hidden_tr * valid.unsqueeze(-1)).sum(dim=(1, 2)) / n_valid.unsqueeze(-1)
        final_nll = F.cross_entropy(
            logits.reshape(-1, A), c_t.reshape(-1).long(), ignore_index=-100
        ).item()
    return h_avg.cpu().numpy(), final_nll, n_epochs


# ── Decoding + stats ──────────────────────────────────────────────────────────
def insample_ols(Z, y):
    """In-sample OLS. Returns r, adjusted R², F-test p, predictions."""
    n, p = Z.shape
    clf   = LinearRegression().fit(Z, y)
    preds = clf.predict(Z)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - p - 1, 1)
    dfn, dfd = p, n - p - 1
    F      = (r2 / dfn) / ((1 - r2) / dfd) if r2 < 1 else np.inf
    p_val  = float(f_dist.sf(F, dfn, dfd))
    return float(np.sqrt(max(r2, 0.0))), float(r2_adj), p_val, preds


def steiger_test(r12, r13, r23, n):
    """Two dependent correlations r12=r(y,preds1), r13=r(y,preds2), r23=r(preds1,preds2)."""
    r12 = np.clip(r12, -0.9999, 0.9999)
    r13 = np.clip(r13, -0.9999, 0.9999)
    r23 = np.clip(r23, -0.9999, 0.9999)
    z12, z13 = np.arctanh(r12), np.arctanh(r13)
    rsq_bar = (r12**2 + r13**2) / 2
    f = (1 - r23) / (2 * (1 - rsq_bar))
    h = (1 - f * rsq_bar) / (1 - rsq_bar)
    var = (2 * (1 - r23) / (n - 1)) * h
    if var <= 0:
        return 0.0, 1.0
    z = (z12 - z13) / np.sqrt(var)
    return float(z), float(2 * norm.sf(abs(z)))


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


# ── Per-combo training with caching ───────────────────────────────────────────
def run_idrnn_combo(xin_t, c_t, task_ids_tensor, z_dim, hidden):
    tag = f"z{z_dim}_h{hidden}"
    cache = os.path.join(CKPT_DIR, f"idrnn_{tag}.pt")
    if os.path.exists(cache):
        print(f"[IDRNN {tag}] cached → loading")
        return torch.load(cache, map_location="cpu", weights_only=False)

    per_seed = {}
    for seed in SEEDS:
        print(f"\n══ IDRNN {tag} seed {seed} ══")
        sel_ep, _ = train_idrnn_step1_inner_cv(
            xin_t, c_t, task_ids_tensor, seed, z_dim=z_dim, hidden=hidden
        )
        model, emb, final_nll = train_idrnn_step1_final(
            xin_t, c_t, task_ids_tensor, seed, sel_ep, z_dim=z_dim, hidden=hidden
        )
        spec, m_nll, mis_nll = compute_specificity_step1(
            model, xin_t, c_t, rng_seed=seed
        )
        print(f"  [IDRNN {tag} seed {seed}] final_nll={final_nll:.4f}  "
              f"matched={m_nll:.4f}  mismatched={mis_nll:.4f}  "
              f"specificity={spec:+.4f}")
        per_seed[seed] = dict(
            emb=emb, selected_epoch=sel_ep, final_nll=final_nll,
            matched_nll=m_nll, mismatched_nll=mis_nll, specificity=spec,
        )
    best_seed = max(per_seed, key=lambda s: per_seed[s]["specificity"])
    out = dict(per_seed=per_seed, best_seed=best_seed,
               z_dim=z_dim, hidden=hidden)
    torch.save(out, cache)
    print(f"[IDRNN {tag}] best seed = {best_seed}  "
          f"spec={per_seed[best_seed]['specificity']:+.4f}  → cached")
    return out


def run_vanilla_combo(xin_t, c_t, task_ids_tensor, hidden):
    tag = f"h{hidden}"
    cache = os.path.join(CKPT_DIR, f"vanilla_{tag}.pt")
    if os.path.exists(cache):
        print(f"[Vanilla {tag}] cached → loading")
        return torch.load(cache, map_location="cpu", weights_only=False)

    per_seed = {}
    for seed in SEEDS:
        print(f"\n══ Vanilla {tag} seed {seed} ══")
        h_lat, final_nll, n_ep = train_vanilla_full(
            xin_t, c_t, task_ids_tensor, seed, hidden=hidden
        )
        print(f"  [Vanilla {tag} seed {seed}] final_nll={final_nll:.4f}  n_ep={n_ep}")
        per_seed[seed] = dict(h=h_lat, final_nll=final_nll, selected_epoch=n_ep)
    best_seed = min(per_seed, key=lambda s: per_seed[s]["final_nll"])
    out = dict(per_seed=per_seed, best_seed=best_seed, hidden=hidden)
    torch.save(out, cache)
    print(f"[Vanilla {tag}] best seed = {best_seed}  "
          f"nll={per_seed[best_seed]['final_nll']:.4f}  → cached")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    xin_np, c_np, subids, task_ids_np = load_all_subjects()
    xin_t = torch.as_tensor(xin_np, dtype=torch.float32, device=device)
    c_t   = torch.as_tensor(c_np,   dtype=torch.long,    device=device)
    task_ids_tensor = torch.as_tensor(task_ids_np, dtype=torch.long, device=device)
    print(f"  xin {tuple(xin_t.shape)}, c {tuple(c_t.shape)}, N={len(subids)}")

    quest = pd.read_csv("data/finalQuestionnaireDataSession1.csv").set_index("ID")
    factors = (pd.read_csv("data/CFA_compound_questionnaire_factors_s1.csv")
                 .set_index("ID")[["AxDep", "posMood", "negMood", "Exp"]])
    quest = quest.join(factors, how="left")
    for k, (items, _) in SCALES.items():
        quest[k] = quest[items].mean(axis=1)

    # ── Train / load all combos ───────────────────────────────────────────────
    idrnn_combos = {}
    for z_dim in Z_DIM_GRID:
        for hidden in HIDDEN_GRID:
            idrnn_combos[(z_dim, hidden)] = run_idrnn_combo(
                xin_t, c_t, task_ids_tensor, z_dim, hidden
            )

    vanilla_combos = {}
    for hidden in HIDDEN_GRID:
        vanilla_combos[hidden] = run_vanilla_combo(
            xin_t, c_t, task_ids_tensor, hidden
        )

    # ── Union in every cached combo so CSVs + plots accumulate across runs ───
    for path in sorted(glob.glob(os.path.join(CKPT_DIR, "idrnn_z*_h*.pt"))):
        name = os.path.splitext(os.path.basename(path))[0]   # idrnn_z{Z}_h{H}
        try:
            _, z_tok, h_tok = name.split("_", 2)
            key = (int(z_tok[1:]), int(h_tok[1:]))
        except Exception:
            continue
        if key in idrnn_combos:
            continue
        idrnn_combos[key] = torch.load(path, map_location="cpu", weights_only=False)

    for path in sorted(glob.glob(os.path.join(CKPT_DIR, "vanilla_h*.pt"))):
        name = os.path.splitext(os.path.basename(path))[0]   # vanilla_h{H}
        try:
            _, h_tok = name.split("_", 1)
            key = int(h_tok[1:])
        except Exception:
            continue
        if key in vanilla_combos:
            continue
        vanilla_combos[key] = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\nUsing {len(idrnn_combos)} IDRNN + {len(vanilla_combos)} Vanilla "
          f"cached combos (current grid ∪ prior runs)")

    # ── Seed summary CSV ──────────────────────────────────────────────────────
    rows = []
    for (z_dim, hidden), d in idrnn_combos.items():
        for s, r in d["per_seed"].items():
            rows.append(dict(
                model="IDRNN", z_dim=z_dim, hidden=hidden, seed=s,
                selected_epoch=r["selected_epoch"], final_nll=r["final_nll"],
                matched_nll=r["matched_nll"], mismatched_nll=r["mismatched_nll"],
                specificity=r["specificity"], is_best=(s == d["best_seed"]),
            ))
    for hidden, d in vanilla_combos.items():
        for s, r in d["per_seed"].items():
            rows.append(dict(
                model="Vanilla", z_dim=np.nan, hidden=hidden, seed=s,
                selected_epoch=r["selected_epoch"], final_nll=r["final_nll"],
                matched_nll=np.nan, mismatched_nll=np.nan, specificity=np.nan,
                is_best=(s == d["best_seed"]),
            ))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "seed_summary.csv"), index=False)
    print(f"\nSaved seed_summary.csv")

    # ── Decoding per combo × scale ────────────────────────────────────────────
    print("\n── In-sample OLS decoding ──")
    decode_rows = []
    for scale in SCALE_KEYS:
        y_all = quest.reindex(subids)[scale].values.astype(float)
        mask = ~np.isnan(y_all)
        yy = y_all[mask]
        n = int(mask.sum())
        if n < 30:
            print(f"  {scale}: n={n}, skipping"); continue

        # Vanilla preds per hidden (best seed)
        van_out = {}
        for hidden, d in vanilla_combos.items():
            Hv = d["per_seed"][d["best_seed"]]["h"][mask]
            van_out[hidden] = insample_ols(Hv, yy)

        # IDRNN per (z, h) + Steiger vs Vanilla @ matched hidden (if available)
        hiddens_with_idrnn = set()
        for (z_dim, hidden), d in idrnn_combos.items():
            hiddens_with_idrnn.add(hidden)
            Zi = d["per_seed"][d["best_seed"]]["emb"][mask]
            r_i, r2_adj_i, p_i, preds_i = insample_ols(Zi, yy)
            if hidden in van_out:
                r_v, r2_adj_v, p_v, preds_v = van_out[hidden]
                r_iv, _ = pearsonr(preds_i, preds_v)
                z_s, p_s = steiger_test(r_i, r_v, r_iv, n)
            else:
                r_v = r2_adj_v = p_v = r_iv = z_s = p_s = np.nan
            decode_rows.append(dict(
                scale=scale, z_dim=z_dim, hidden=hidden, n=n,
                r_idrnn=r_i, r2_adj_idrnn=r2_adj_i, p_idrnn=p_i,
                r_vanilla=r_v, r2_adj_vanilla=r2_adj_v, p_vanilla=p_v,
                r_iv=r_iv, z_steiger=z_s, p_steiger=p_s,
                matched_capacity=(z_dim == hidden),
            ))

        # Vanilla-only rows for hidden values without any IDRNN counterpart
        for hidden, (r_v, r2_adj_v, p_v, _) in van_out.items():
            if hidden in hiddens_with_idrnn:
                continue
            decode_rows.append(dict(
                scale=scale, z_dim=np.nan, hidden=hidden, n=n,
                r_idrnn=np.nan, r2_adj_idrnn=np.nan, p_idrnn=np.nan,
                r_vanilla=r_v, r2_adj_vanilla=r2_adj_v, p_vanilla=p_v,
                r_iv=np.nan, z_steiger=np.nan, p_steiger=np.nan,
                matched_capacity=False,
            ))
    decode_df = pd.DataFrame(decode_rows)
    decode_df.to_csv(os.path.join(OUT_DIR, "decoding_results.csv"), index=False)
    print(f"Saved decoding_results.csv ({len(decode_df)} rows)")

    # ── Discover z_dim / hidden values present in the CSV ────────────────────
    z_dims_plot = sorted(
        int(z) for z in decode_df["z_dim"].dropna().unique()
    )
    hiddens_plot = sorted(int(h) for h in decode_df["hidden"].unique())

    # ── Aggregate for summary plot ────────────────────────────────────────────
    idrnn_df = decode_df.dropna(subset=["z_dim"])
    agg = idrnn_df.groupby(["z_dim", "hidden"]).agg(
        mean_abs_r_idrnn=("r_idrnn",     lambda s: np.abs(s).mean()),
        mean_r2_adj_idrnn=("r2_adj_idrnn", "mean"),
    ).reset_index()
    van_agg = decode_df.dropna(subset=["r_vanilla"]).groupby(["hidden"]).agg(
        mean_abs_r_vanilla=("r_vanilla",     lambda s: np.abs(s).mean()),
        mean_r2_adj_vanilla=("r2_adj_vanilla", "mean"),
    ).reset_index()

    cmap = plt.get_cmap("viridis")
    z_colors = {z: cmap(i / max(len(z_dims_plot) - 1, 1))
                for i, z in enumerate(z_dims_plot)}

    # ── Plot 1: mean |r| and mean adj R² vs hidden ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, ycol_i, ycol_v, ylabel in [
        (axes[0], "mean_abs_r_idrnn",  "mean_abs_r_vanilla",  "mean |r| across 6 scales"),
        (axes[1], "mean_r2_adj_idrnn", "mean_r2_adj_vanilla", "mean adjusted R²"),
    ]:
        for z_dim in z_dims_plot:
            sub = agg[agg["z_dim"] == z_dim].sort_values("hidden")
            ax.plot(sub["hidden"], sub[ycol_i], "o-",
                    color=z_colors[z_dim], lw=2, ms=7,
                    label=f"IDRNN  z_dim={z_dim}")
        sub_v = van_agg.sort_values("hidden")
        ax.plot(sub_v["hidden"], sub_v[ycol_v], "s--",
                color="black", lw=2, ms=7, label="Vanilla")
        # matched-PREDICTOR anchors (z_dim == hidden) — Steiger bias-free
        matched = agg[agg["z_dim"] == agg["hidden"]]
        if not matched.empty:
            ax.scatter(matched["hidden"], matched[ycol_i],
                       facecolors="none", edgecolors="red",
                       s=180, lw=2, zorder=5,
                       label="z_dim = hidden (matched #predictors)")
        ax.set_xlabel("decoder hidden (common process)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(hiddens_plot)
        ax.legend(fontsize=8, loc="best")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Thalmann step-1 decodability vs common-process capacity",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "decodability_vs_hidden.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved decodability_vs_hidden.png")

    # ── Plot 2: per-scale |r| vs hidden ───────────────────────────────────────
    n_scales = len(SCALE_KEYS)
    n_cols = 3
    n_rows = (n_scales + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             sharex=True, squeeze=False)
    for ax, scale in zip(axes.flat, SCALE_KEYS):
        sub = decode_df[decode_df["scale"] == scale]
        sub_idrnn = sub.dropna(subset=["z_dim"])
        for z_dim in z_dims_plot:
            sub_z = sub_idrnn[sub_idrnn["z_dim"] == z_dim].sort_values("hidden")
            ax.plot(sub_z["hidden"], sub_z["r_idrnn"].abs(), "o-",
                    color=z_colors[z_dim], lw=1.5, ms=5,
                    label=f"IDRNN z={z_dim}")
        sub_v = (sub.dropna(subset=["r_vanilla"])
                    .drop_duplicates("hidden").sort_values("hidden"))
        ax.plot(sub_v["hidden"], sub_v["r_vanilla"].abs(), "s--",
                color="black", lw=1.5, ms=5, label="Vanilla")
        ax.set_title(SCALES[scale][1].replace("\n", " "), fontsize=10)
        ax.set_ylabel("|r|", fontsize=9)
        ax.set_xticks(hiddens_plot)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)
    # xlabel on bottom-most populated axis of each column; hide empty trailing axes
    for j in range(n_cols):
        for i in range(n_rows - 1, -1, -1):
            idx = i * n_cols + j
            if idx < n_scales:
                axes[i, j].set_xlabel("hidden", fontsize=10)
                break
    for idx in range(n_scales, n_rows * n_cols):
        axes.flat[idx].axis("off")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Per-scale in-sample |r| vs decoder hidden",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "per_scale_decodability.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved per_scale_decodability.png")

    # ── Matched-capacity Steiger table ────────────────────────────────────────
    matched = decode_df[decode_df["matched_capacity"]]
    if not matched.empty:
        print("\n── Steiger test at z_dim == hidden (matched-predictor, bias-free) ──")
        for _, row in matched.sort_values(["scale", "hidden"]).iterrows():
            print(f"  {row['scale']:<10} h=z={row['hidden']}  "
                  f"r_I={row['r_idrnn']:+.3f}  r_V={row['r_vanilla']:+.3f}  "
                  f"Δr={row['r_idrnn']-row['r_vanilla']:+.3f}  "
                  f"p_Steiger={row['p_steiger']:.3g} {sig_stars(row['p_steiger'])}")

    # ── Print best combos ─────────────────────────────────────────────────────
    print("\n── Top IDRNN combos by mean |r| ──")
    top = agg.sort_values("mean_abs_r_idrnn", ascending=False)
    for _, row in top.iterrows():
        print(f"  z={int(row.z_dim)} h={int(row.hidden)}  "
              f"mean|r|={row.mean_abs_r_idrnn:.3f}  "
              f"mean R²_adj={row.mean_r2_adj_idrnn:+.3f}")

    print(f"\nAll outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
