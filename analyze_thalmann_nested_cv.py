#!/usr/bin/env python3
"""Panel a (held-out NLL) for Thalmann z=3 — port of analyze_dezfouli_nested_cv.py.

Per participant (each subject is in exactly one outer fold's test set), computes
mean per-trial NLL — SPLIT BY TASK — for:
  • IDRNN        (LatentRNN_secondstep encoder->z->frozen decoder, task emb)
  • CP-RNN       (same IDRNN decoder forwarded with z = 0)  [ablation]
  • Vanilla RNN  (AblatedRNN)
averaged across the per-fold final seeds, pooled across the 3 folds.

Cog-model NLLs are read from the existing per-fold CSVs (per-trial
normalized_likelihood over ALL 500 trials).  IMPORTANT: the RNNs are trained
with the loss weighted to task-1 (restless) only, so compare like-with-like:
  - nll_*_task1  : restless trials  (where the RNN is optimised)  <- headline
  - nll_*_task0  : 2-armed trials   (RNN not optimised here)
  - nll_*_all    : all 500 trials   (matches cog CSV units)

Output: final_plots/thalmann_z3/per_participant_nll.csv
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
from modelsandtraining import IDRNN, Decoder, LatentRNN_secondstep, AblatedRNN
from nested_cv.config import FINAL_SEEDS, N_OUTER_FOLDS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAG = "_z3"
# Env-overridable so the same analyzer serves S1 nested-CV and the pooled S1+S2
# fold-CV (per-fold RNN runs in runs_thalmann_s2_foldcv/{arch}/fold{F}/seed_{S},
# cog CSVs + fold data in data_thalmann_s2/fold{F}).
DATA = os.environ.get("THAL_NLL_DATA", "data_thalmann")
OUT_DIR = os.environ.get("THAL_NLL_OUT", f"final_plots/thalmann{TAG}")
RUN_IDRNN = os.environ.get("THAL_NLL_RUN_IDRNN", f"runs_thalmann_nested_cv{TAG}/fold{{fold}}/seed_{{seed}}")
RUN_VANILLA = os.environ.get("THAL_NLL_RUN_VANILLA", f"runs_vanilla_thalmann_nested_cv{TAG}/fold{{fold}}/seed_{{seed}}")
N_SEEDS_NLL = int(os.environ.get("THAL_NLL_NSEEDS", "30"))
os.makedirs(OUT_DIR, exist_ok=True)
TASK_IDS = np.load(f"{DATA}/task_ids_per_block.npy")          # 31 (S1) or 62 (pooled)
A = 4


def _build_idrnn(cfg):
    mc = cfg["model_config"]
    enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"], hid=mc["enc_hidden"],
                n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"],
                continuous_encoder=mc.get("continuous_encoder", True))
    dec = Decoder(in_dim=mc["dec_in_dim"], z_dim=mc["z_dim"], hid=mc["hidden"], A=mc["A"])
    m = LatentRNN_secondstep(encoder=enc, hid=mc["hidden"], z_dim=mc["z_dim"],
                             in_dim=mc["dec_in_dim"], A=mc["A"], decoder=dec,
                             n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"],
                             reinit_decoder_per_block=mc.get("reinit_decoder_per_block", True))
    return m.to(DEVICE)


def _build_vanilla(cfg):
    mc = cfg["model_config"]
    return AblatedRNN(hid=mc["hidden"], in_dim=mc["dec_in_dim"], A=mc["A"],
                      block_structure=True, n_tasks=mc["n_tasks"],
                      task_emb_dim=mc["task_emb_dim"]).to(DEVICE)


def _ckpt(run_dir, cfg):
    cdir = os.path.join(run_dir, "checkpoints")
    ep = cfg.get("cv_selected_epoch")
    if ep is not None and os.path.exists(os.path.join(cdir, f"epoch{ep:04d}.pt")):
        return os.path.join(cdir, f"epoch{ep:04d}.pt")
    av = sorted(int(f[5:9]) for f in os.listdir(cdir)
                if f.startswith("epoch") and f.endswith(".pt") and "pre_step25" not in f)
    return os.path.join(cdir, f"epoch{av[-1]:04d}.pt") if av else None


def _load_state(p):
    sd = torch.load(p, map_location=DEVICE)
    return sd["model_state"] if isinstance(sd, dict) and "model_state" in sd else sd


@torch.no_grad()
def _per_subject_nll_bytask(logits, c_t):
    """logits (N,Bk,T,A), c_t (N,Bk,T). Returns dict task0/task1/all -> (N,) mean
    per-trial NLL using TASK_IDS per block."""
    logp = F.log_softmax(logits, dim=-1)
    valid = (c_t >= 0)
    c_safe = c_t.clone(); c_safe[~valid] = 0
    chosen = logp.gather(-1, c_safe.long().unsqueeze(-1)).squeeze(-1)   # (N,Bk,T)
    nll = -chosen
    nll = torch.where(valid, nll, torch.zeros_like(nll))
    vmask = valid.float()
    tid = torch.as_tensor(TASK_IDS, device=logits.device)
    out = {}
    for name, sel in [("task0", tid == 0), ("task1", tid == 1),
                      ("all", torch.ones_like(tid, dtype=torch.bool))]:
        bsel = sel.view(1, -1, 1).float()
        num = (nll * vmask * bsel).sum(dim=(1, 2))
        den = (vmask * bsel).sum(dim=(1, 2)).clamp(min=1)
        out[name] = (num / den).cpu().numpy()
    return out


@torch.no_grad()
def _cp_rnn_nll_bytask(model, xin, c_t):
    """IDRNN decoder forwarded with z=0 (uninformative latent ablation)."""
    N, Bk, T, _ = xin.shape
    z0 = torch.zeros(N, model.decoder.z2h0.in_features, device=DEVICE)
    blk_logits = []
    h = None
    for b in range(Bk):
        if model.reinit_decoder_per_block and b > 0:
            h = None
        inp = model._append_task_emb(xin[:, b], b)
        lb, h = model.decoder(inp, z0, hidden=h)
        blk_logits.append(lb)
    logits = torch.stack(blk_logits, dim=1)
    return _per_subject_nll_bytask(logits, c_t)


def aggregate_rnn(arch, seeds):
    tmpl = RUN_IDRNN if arch == "idrnn" else RUN_VANILLA
    per = {}        # subid -> {task0,task1,all}
    per_cp = {}     # CP-RNN (idrnn only)
    for fold in range(N_OUTER_FOLDS):
        fdir = f"{DATA}/fold{fold}"
        subids = pd.read_csv(f"{fdir}/df_test.csv")["subid"].values
        xin = torch.from_numpy(np.load(f"{fdir}/xin_test.npy")).float().to(DEVICE)
        c_t = torch.from_numpy(np.load(f"{fdir}/c_test.npy")).float().to(DEVICE)
        task_ids = torch.as_tensor(TASK_IDS, dtype=torch.long, device=DEVICE)
        acc = {k: [] for k in ("task0", "task1", "all")}
        acc_cp = {k: [] for k in ("task0", "task1", "all")}
        nseed = 0
        for s in seeds:
            rd = tmpl.format(fold=fold, seed=s)
            cp = os.path.join(rd, "config.json")
            if not os.path.exists(cp):
                continue
            cfg = json.load(open(cp))
            ck = _ckpt(rd, cfg)
            if ck is None:
                continue
            model = _build_idrnn(cfg) if arch == "idrnn" else _build_vanilla(cfg)
            model.load_state_dict(_load_state(ck), strict=False)
            model.set_task_ids(task_ids)
            model.eval()
            if arch == "idrnn":
                logits, _, _, _, _ = model(xin, xin)
            else:
                logits, _, _ = model(xin)
            d = _per_subject_nll_bytask(logits, c_t)
            for k in acc:
                acc[k].append(d[k])
            if arch == "idrnn":
                dc = _cp_rnn_nll_bytask(model, xin, c_t)
                for k in acc_cp:
                    acc_cp[k].append(dc[k])
            nseed += 1
        if nseed == 0:
            print(f"  [warn] {arch} fold{fold}: no seeds"); continue
        mean = {k: np.mean(np.stack(acc[k]), axis=0) for k in acc}
        for i, sid in enumerate(subids):
            per[int(sid)] = {k: float(mean[k][i]) for k in mean}
        if arch == "idrnn":
            meanc = {k: np.mean(np.stack(acc_cp[k]), axis=0) for k in acc_cp}
            for i, sid in enumerate(subids):
                per_cp[int(sid)] = {k: float(meanc[k][i]) for k in meanc}
        print(f"  [{arch} fold{fold}] {nseed} seeds x {len(subids)} subj")
    return per, per_cp


def aggregate_cog(basename):
    out = {}
    for fold in range(N_OUTER_FOLDS):
        p = f"{DATA}/fold{fold}/{basename}.csv"
        if not os.path.exists(p):
            print(f"  [warn] missing {p}"); continue
        for _, r in pd.read_csv(p).iterrows():
            out[int(r["subid"])] = float(r["normalized_likelihood"])
    return out


def main():
    seeds = FINAL_SEEDS[:N_SEEDS_NLL]
    print("== IDRNN + CP-RNN ==")
    idr, cp = aggregate_rnn("idrnn", seeds)
    print("== Vanilla ==")
    van, _ = aggregate_rnn("vanilla", seeds)
    print("== Cog NLLs (all-trial, from CSVs) ==")
    cog_em = aggregate_cog("cog_model_results")
    cog_cp = aggregate_cog("cog_model_cp_results")
    ill_em = aggregate_cog("illspec_em_results")
    ill_cp = aggregate_cog("illspec_cp_results")

    subs = sorted(set(idr) | set(van))
    rows = []
    for sid in subs:
        r = {"subid": sid}
        for arch, d in [("idrnn", idr), ("cp_rnn", cp), ("vanilla", van)]:
            for tk in ("task0", "task1", "all"):
                r[f"nll_{arch}_{tk}"] = d.get(sid, {}).get(tk, np.nan)
        r["nll_cog_em"] = cog_em.get(sid, np.nan)
        r["nll_cog_cp"] = cog_cp.get(sid, np.nan)
        r["nll_ill_em"] = ill_em.get(sid, np.nan)
        r["nll_ill_cp"] = ill_cp.get(sid, np.nan)
        rows.append(r)
    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "per_participant_nll.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {out}  (n={len(df)})")
    means = {c: float(df[c].mean()) for c in df.columns if c.startswith("nll_")}
    print("Mean NLL by model:")
    for k, v in means.items():
        print(f"  {k:<20} {v:.4f}")
    json.dump(means, open(os.path.join(OUT_DIR, "nll_means.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
