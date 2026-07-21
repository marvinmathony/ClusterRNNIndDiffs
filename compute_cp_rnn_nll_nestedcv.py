"""Nested-CV held-out CP-RNN NLL.

For each (dataset, fold, top-K-spec seeds): take the IDRNN trained on the
fold's 2/3 training subjects (stage_c, runs_dataset{D}_nested_cv_z1/foldF/seed_S),
forward its decoder with z=0 on the fold's test subjects (xin_test of that fold),
compute per-session summed NLL.  Pool across (fold, seed) → one row per
held-out subject.  Mean across seeds within each subject; subject appears
exactly once across the 3 folds.

Saves: cp_rnn_nll_nestedcv.csv  with columns dataset_id, fold, session, nll_summed.
"""
import os, json, sys, glob
import numpy as np, pandas as pd, torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from modelsandtraining import IDRNN, Decoder, LatentRNN_secondstep

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASETS = [0, 1, 2, 3, 5, 7, 10, 12, 15, 17]
TOP_K = 5


def _build_idrnn(cfg):
    mc = cfg["model_config"]
    enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"], hid=mc["enc_hidden"],
                n_tasks=mc.get("n_tasks"), task_emb_dim=mc.get("task_emb_dim", 0))
    dec = Decoder(in_dim=mc["dec_in_dim"], z_dim=mc["z_dim"], hid=mc["hidden"], A=mc["A"])
    for p in dec.parameters(): p.requires_grad = False
    m = LatentRNN_secondstep(encoder=enc, hid=mc["hidden"], z_dim=mc["z_dim"],
                              in_dim=mc["dec_in_dim"], A=mc["A"], decoder=dec,
                              reinit_decoder_per_block=mc.get("reinit_decoder_per_block", False))
    return m.to(DEVICE)


def cp_nll_for_fold(D, fold):
    fold_base = f"runs_dataset{D}_nested_cv_z1/fold{fold}"
    seed_dirs = sorted(glob.glob(f"{fold_base}/seed_*"))
    # Rank by step1_specificity (top-K)
    scored = []
    for sd in seed_dirs:
        cfg_p = os.path.join(sd, "config.json")
        if not os.path.exists(cfg_p): continue
        try:
            cfg = json.load(open(cfg_p))
            if cfg.get("step1_specificity") is not None:
                scored.append((float(cfg["step1_specificity"]), sd, cfg))
        except: pass
    scored.sort(key=lambda t: -t[0])
    top = scored[:TOP_K]
    if not top: return None

    # Load fold test data
    xin = torch.from_numpy(np.load(f"data_dataset{D}/fold{fold}/xin_test.npy")).float().to(DEVICE)
    c   = torch.from_numpy(np.load(f"data_dataset{D}/fold{fold}/c_test.npy")).long().to(DEVICE)
    N, T, _ = xin.shape

    per_seed = []
    for spec, sd, cfg in top:
        ep = cfg.get("cv_selected_epoch")
        ckpt = os.path.join(sd, "checkpoints", f"epoch{ep:04d}.pt")
        if not os.path.exists(ckpt): continue
        m = _build_idrnn(cfg)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
        m.eval()
        with torch.no_grad():
            z_zero = torch.zeros(N, cfg["model_config"]["z_dim"], device=DEVICE)
            logits, _ = m.decoder(xin, z_zero)
            logp = F.log_softmax(logits, dim=-1)
            chosen = logp.gather(-1, c.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            mask = (c >= 0).float()
            nll_sum = -(chosen * mask).sum(dim=1).cpu().numpy()
        per_seed.append(nll_sum)
        del m
    return np.mean(per_seed, axis=0) if per_seed else None  # (N_test,)


def main():
    rows = []
    for D in DATASETS:
        for fold in range(3):
            nll = cp_nll_for_fold(D, fold)
            if nll is None:
                print(f"  ds{D} fold{fold}: no top-K seeds — skip"); continue
            for s, v in enumerate(nll):
                rows.append(dict(dataset_id=D, fold=fold, session=s,
                                 nll_summed=float(v)))
            print(f"  ds{D} fold{fold}: mean summed NLL = {nll.mean():.2f} (N={len(nll)})")
    df = pd.DataFrame(rows)
    df.to_csv("cp_rnn_nll_nestedcv.csv", index=False)
    print(f"\nSaved -> cp_rnn_nll_nestedcv.csv ({len(df)} rows)")
    print("Per-dataset means:")
    print(df.groupby("dataset_id")["nll_summed"].mean().to_string())


if __name__ == "__main__":
    main()
