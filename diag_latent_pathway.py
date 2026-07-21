#!/usr/bin/env python3
"""Diagnose the IDRNN latent pathway on full-cohort models (train==test, so the
in-sample lookup z is available).  For each model compute per-trial NLL with:
  - lookup z   (step-1 per-subject fitted z; in-sample 'best case')
  - encoder z  (mu from encoder(xin); what held-out eval uses)
  - z = 0      (CP-RNN)
and report z-std + corr(lookup, encoder).  If lookup << encoder ≈ z0, the encoder
fails to carry the latent (so held-out gap is an encoder-generalisation issue, not
'the latent is useless').  strict=True load to catch any missing weights.
"""
import glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
from modelsandtraining import IDRNN, Decoder, LatentRNN_secondstep

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CASES = [("S1   thalmann_z3_full",  "final_plots/thalmann_z3_full",     "data_thalmann_full"),
         ("pool thalmann_z3_s2_full","final_plots/thalmann_z3_s2_full",  "data_thalmann_s2_full")]


def build(cfg):
    mc = cfg["model_config"]
    enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"], hid=mc["enc_hidden"],
                n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"],
                continuous_encoder=mc.get("continuous_encoder", True))
    dec = Decoder(in_dim=mc["dec_in_dim"], z_dim=mc["z_dim"], hid=mc["hidden"], A=mc["A"])
    return LatentRNN_secondstep(encoder=enc, hid=mc["hidden"], z_dim=mc["z_dim"],
                                in_dim=mc["dec_in_dim"], A=mc["A"], decoder=dec,
                                n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"],
                                reinit_decoder_per_block=mc.get("reinit_decoder_per_block", True)).to(DEVICE)


def ckpt(rd, cfg):
    cdir = os.path.join(rd, "checkpoints"); ep = cfg.get("cv_selected_epoch")
    if ep and os.path.exists(os.path.join(cdir, f"epoch{ep:04d}.pt")):
        return os.path.join(cdir, f"epoch{ep:04d}.pt")
    av = sorted(int(f[5:9]) for f in os.listdir(cdir) if f.startswith("epoch") and f.endswith(".pt") and "pre_step25" not in f)
    return os.path.join(cdir, f"epoch{av[-1]:04d}.pt")


@torch.no_grad()
def nll_with_z(model, xin, c_t, z):
    N, Bk, T, _ = xin.shape
    h = None; blk = []
    for b in range(Bk):
        if model.reinit_decoder_per_block and b > 0:
            h = None
        inp = model._append_task_emb(xin[:, b], b)
        lb, h = model.decoder(inp, z, hidden=h); blk.append(lb)
    logits = torch.stack(blk, 1)
    logp = F.log_softmax(logits, -1); valid = (c_t >= 0); cs = c_t.clone(); cs[~valid] = 0
    nll = -logp.gather(-1, cs.long().unsqueeze(-1)).squeeze(-1)
    nll = torch.where(valid, nll, torch.zeros_like(nll))
    return (nll.sum(dim=(1, 2)) / valid.float().sum(dim=(1, 2)).clamp(min=1)).cpu().numpy()


for name, base, data in CASES:
    seed_dirs = sorted(glob.glob(f"{base}/canonical/idrnn/runs/seed_*"))
    if not seed_dirs:
        print(f"\n{name}: no seed dirs"); continue
    rd = seed_dirs[0]
    cfg = json.load(open(f"{rd}/config.json"))
    m = build(cfg)
    sd = torch.load(ckpt(rd, cfg), map_location=DEVICE)
    sd = sd["model_state"] if isinstance(sd, dict) and "model_state" in sd else sd
    miss, unexp = m.load_state_dict(sd, strict=False)
    m.set_task_ids(torch.from_numpy(np.load(f"{data}/task_ids_per_block.npy")).long().to(DEVICE))
    m.eval()
    enc_keys = [k for k in m.state_dict() if k.startswith("encoder.")]
    miss_enc = [k for k in miss if k.startswith("encoder.")]
    xin = torch.from_numpy(np.load(f"{data}/xin_train.npy")).float().to(DEVICE)
    c_t = torch.from_numpy(np.load(f"{data}/c_train.npy")).float().to(DEVICE)
    z_lookup = np.load(f"{rd}/step1_z_lookup.npy")
    with torch.no_grad():
        mu, lv = m.encoder(xin, return_per_timestep=False)
    z_enc = mu.cpu().numpy()
    print(f"\n===== {name}  (seed dir {os.path.basename(rd)}) =====")
    print(f"  load: {len(miss)} missing ({len(miss_enc)} encoder), {len(unexp)} unexpected; encoder has {len(enc_keys)} params")
    print(f"  z_lookup shape {z_lookup.shape}  std/dim {np.round(z_lookup.std(0),3)}")
    print(f"  z_enc    shape {z_enc.shape}  std/dim {np.round(z_enc.std(0),3)}")
    for d in range(z_lookup.shape[1]):
        c = np.corrcoef(z_lookup[:, d], z_enc[:, d])[0, 1]
        print(f"    dim{d}: corr(lookup, encoder) = {c:+.3f}")
    zl = torch.from_numpy(z_lookup).float().to(DEVICE)
    ze = mu
    z0 = torch.zeros_like(ze)
    nl, ne, n0 = (nll_with_z(m, xin, c_t, z).mean() for z in (zl, ze, z0))
    print(f"  in-sample NLL:  lookup={nl:.4f}   encoder={ne:.4f}   z0(CP)={n0:.4f}")
    print(f"  gap vs z0:      lookup {n0-nl:+.4f}   encoder {n0-ne:+.4f}   (encoder/lookup = {(n0-ne)/(n0-nl+1e-9):.0%})")
