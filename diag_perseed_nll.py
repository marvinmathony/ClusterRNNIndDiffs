#!/usr/bin/env python3
"""Per-seed held-out NLL diagnostic on the PROPER per-fold pooled runs
(runs_pooled/s2_2task/{arch}/fold{F}/seed_{S}).  Answers:
  (1) single-seed vs seed-averaged — prints every seed + mean/SD,
  (2) is the held-out latent used — IDRNN vs CP-RNN (z=0) per seed, and the
      gap CP_RNN - IDRNN (positive = latent helps).
Each subject is held out in exactly one fold; per seed we pool the 3 folds.
Writes final_plots/thalmann_z3_s2_pooled/per_seed_nll.csv.
"""
import glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
from modelsandtraining import IDRNN, Decoder, LatentRNN_secondstep, AblatedRNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = "data_thalmann_s2"
RUNS = "runs_pooled/s2_2task"
TASK_IDS = np.load(f"{DATA}/task_ids_per_block.npy")


def _build_idrnn(cfg):
    mc = cfg["model_config"]
    enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"], hid=mc["enc_hidden"],
                n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"],
                continuous_encoder=mc.get("continuous_encoder", True))
    dec = Decoder(in_dim=mc["dec_in_dim"], z_dim=mc["z_dim"], hid=mc["hidden"], A=mc["A"])
    return LatentRNN_secondstep(encoder=enc, hid=mc["hidden"], z_dim=mc["z_dim"],
                                in_dim=mc["dec_in_dim"], A=mc["A"], decoder=dec,
                                n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"],
                                reinit_decoder_per_block=mc.get("reinit_decoder_per_block", True)).to(DEVICE)


def _build_vanilla(cfg):
    mc = cfg["model_config"]
    return AblatedRNN(hid=mc["hidden"], in_dim=mc["dec_in_dim"], A=mc["A"], block_structure=True,
                      n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"]).to(DEVICE)


def _ckpt(rd, cfg):
    cdir = os.path.join(rd, "checkpoints"); ep = cfg.get("cv_selected_epoch")
    if ep and os.path.exists(os.path.join(cdir, f"epoch{ep:04d}.pt")):
        return os.path.join(cdir, f"epoch{ep:04d}.pt")
    av = sorted(int(f[5:9]) for f in os.listdir(cdir) if f.startswith("epoch") and f.endswith(".pt") and "pre_step25" not in f)
    return os.path.join(cdir, f"epoch{av[-1]:04d}.pt") if av else None


def _load_state(p):
    sd = torch.load(p, map_location=DEVICE)
    return sd["model_state"] if isinstance(sd, dict) and "model_state" in sd else sd


@torch.no_grad()
def _nll(logits, c_t):
    logp = F.log_softmax(logits, -1); valid = (c_t >= 0)
    cs = c_t.clone(); cs[~valid] = 0
    nll = -logp.gather(-1, cs.long().unsqueeze(-1)).squeeze(-1)
    nll = torch.where(valid, nll, torch.zeros_like(nll))
    return (nll.sum(dim=(1, 2)) / valid.float().sum(dim=(1, 2)).clamp(min=1)).cpu().numpy()


@torch.no_grad()
def _cp_nll(model, xin, c_t):
    N, Bk, T, _ = xin.shape
    z0 = torch.zeros(N, model.decoder.z2h0.in_features, device=DEVICE)
    h = None; blk = []
    for b in range(Bk):
        if model.reinit_decoder_per_block and b > 0:
            h = None
        inp = model._append_task_emb(xin[:, b], b)
        lb, h = model.decoder(inp, z0, hidden=h); blk.append(lb)
    return _nll(torch.stack(blk, 1), c_t)


def seeds_avail():
    return sorted({int(d.split("seed_")[1]) for f in range(3)
                   for d in glob.glob(f"{RUNS}/idrnn/fold{f}/seed_*")})


def per_seed():
    tid = torch.as_tensor(TASK_IDS, dtype=torch.long, device=DEVICE)
    rows = []
    for s in seeds_avail():
        idr, cp, van = [], [], []
        for f in range(3):
            xin = torch.from_numpy(np.load(f"{DATA}/fold{f}/xin_test.npy")).float().to(DEVICE)
            c_t = torch.from_numpy(np.load(f"{DATA}/fold{f}/c_test.npy")).float().to(DEVICE)
            ri = f"{RUNS}/idrnn/fold{f}/seed_{s}"; rv = f"{RUNS}/vanilla/fold{f}/seed_{s}"
            if os.path.exists(f"{ri}/config.json"):
                cfg = json.load(open(f"{ri}/config.json")); ck = _ckpt(ri, cfg)
                m = _build_idrnn(cfg); m.load_state_dict(_load_state(ck), strict=False); m.set_task_ids(tid); m.eval()
                idr.append(_nll(m(xin, xin)[0], c_t)); cp.append(_cp_nll(m, xin, c_t))
            if os.path.exists(f"{rv}/config.json"):
                cfg = json.load(open(f"{rv}/config.json")); ck = _ckpt(rv, cfg)
                m = _build_vanilla(cfg); m.load_state_dict(_load_state(ck), strict=False); m.set_task_ids(tid); m.eval()
                van.append(_nll(m(xin)[0], c_t))
        rows.append(dict(seed=s, idrnn=np.concatenate(idr).mean(), cp_rnn=np.concatenate(cp).mean(),
                         vanilla=np.concatenate(van).mean() if van else np.nan))
    return rows


if __name__ == "__main__":
    rows = per_seed()
    import pandas as pd
    df = pd.DataFrame(rows); df["cp_minus_idrnn"] = df.cp_rnn - df.idrnn
    df.to_csv(f"final_plots/thalmann_z3_s2_pooled/per_seed_nll.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n=== across {len(df)} seeds (held-out NLL, all blocks) ===")
    for k in ("idrnn", "cp_rnn", "vanilla"):
        print(f"  {k:8s} mean={df[k].mean():.4f}  SD={df[k].std():.4f}  range=[{df[k].min():.4f},{df[k].max():.4f}]")
    g = df.cp_minus_idrnn
    print(f"\n  latent benefit (CP_RNN - IDRNN): mean={g.mean():+.4f} SD={g.std():.4f}; "
          f"seeds where latent helps (IDRNN<CP-RNN): {(g>0).sum()}/{len(df)}")
