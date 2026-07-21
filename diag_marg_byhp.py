"""Decisive recipe test: MARGINALIZED+CAUSAL held-out IDRNN vs CP-RNN gap, for
both HP selections (ELBO=cv_val_loss vs NLL=cv_val_nll), all 3 folds, using the
existing 1-seed HP-search runs.  Tells us the right recipe for panel a."""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, "."); torch.set_num_threads(4)
from modelsandtraining import (IDRNN, Decoder, LatentRNN_secondstep,
    test_latentrnn_secondstep_causal_posterior_weighting as TESTFN, compute_rnn_likelihoods_torch)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA, HP, NSUB, N = "data_thalmann_s2", "hp_pooled/s2_2task/idrnn", 30, 100
TID = torch.from_numpy(np.load(f"{DATA}/task_ids_per_block.npy")).long().to(DEV)


def build_load(rd, cfg):
    mc = cfg["model_config"]
    enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"], hid=mc["enc_hidden"], n_tasks=mc["n_tasks"],
                task_emb_dim=mc["task_emb_dim"], continuous_encoder=mc.get("continuous_encoder", True))
    dec = Decoder(in_dim=mc["dec_in_dim"], z_dim=mc["z_dim"], hid=mc["hidden"], A=mc["A"])
    m = LatentRNN_secondstep(encoder=enc, hid=mc["hidden"], z_dim=mc["z_dim"], in_dim=mc["dec_in_dim"], A=mc["A"],
            decoder=dec, n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"],
            reinit_decoder_per_block=mc.get("reinit_decoder_per_block", True)).to(DEV)
    cdir = f"{rd}/checkpoints"; ep = cfg.get("cv_selected_epoch")
    ckp = f"{cdir}/epoch{ep:04d}.pt" if ep and os.path.exists(f"{cdir}/epoch{ep:04d}.pt") else \
        f"{cdir}/" + sorted(f for f in os.listdir(cdir) if f.startswith("epoch") and "pre_step25" not in f)[-1]
    sd = torch.load(ckp, map_location=DEV); sd = sd.get("model_state", sd) if isinstance(sd, dict) else sd
    m.load_state_dict(sd, strict=False); m.set_task_ids(TID); m.eval()
    return m


def winner(f, metric):
    best = None
    for rd in glob.glob(f"{HP}/fold{f}/*"):
        try: c = json.load(open(f"{rd}/config.json"))
        except: continue
        if metric not in c: continue
        if best is None or c[metric] < best[0]: best = (c[metric], rd, c)
    return best


for label, metric in [("ELBO-HP (cv_val_loss)", "cv_val_loss"), ("NLL-HP (cv_val_nll)", "cv_val_nll")]:
    gaps, idrs, cps = [], [], []
    for f in range(3):
        _, rd, cfg = winner(f, metric)
        m = build_load(rd, cfg)
        xin = torch.from_numpy(np.load(f"{DATA}/fold{f}/xin_test.npy")[:NSUB]).float().to(DEV)
        c_t = torch.from_numpy(np.load(f"{DATA}/fold{f}/c_test.npy")[:NSUB]).float().to(DEV)
        di, *_ = compute_rnn_likelihoods_torch(TESTFN, m, xin, c_t, xin, latent=True, N=N, id=True)
        dc, *_ = compute_rnn_likelihoods_torch(TESTFN, m, xin, c_t, xin, latent=True, N=N, id=False)
        gi, gc = di["normalized_likelihood"].mean(), dc["normalized_likelihood"].mean()
        gaps.append(gc - gi); idrs.append(gi); cps.append(gc)
        print(f"  [{label}] fold{f}: IDRNN={gi:.4f} CP={gc:.4f} gap={gc-gi:+.4f}  (lmbd={cfg.get('lmbd')})")
    print(f"==> {label}: mean IDRNN={np.mean(idrs):.4f} CP={np.mean(cps):.4f} gap={np.mean(gaps):+.4f}\n")
