"""Confirm the fix: for each fold, compare the ELBO-winner vs the NLL-winner
HP-search run (trained on fold-train, 1 seed) on the fold's HELD-OUT test set —
encoder z_std + held-out IDRNN vs CP-RNN(z=0) NLL gap.  Leakage-free: selection
was by inner-CV cv_val_nll, the gap is on the untouched outer-test fold."""
import glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "."); torch.set_num_threads(4)
from modelsandtraining import IDRNN, Decoder, LatentRNN_secondstep
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = "data_thalmann_s2"; HP = "hp_pooled/s2_2task/idrnn"


def build(cfg):
    mc = cfg["model_config"]
    enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"], hid=mc["enc_hidden"], n_tasks=mc["n_tasks"],
                task_emb_dim=mc["task_emb_dim"], continuous_encoder=mc.get("continuous_encoder", True))
    dec = Decoder(in_dim=mc["dec_in_dim"], z_dim=mc["z_dim"], hid=mc["hidden"], A=mc["A"])
    return LatentRNN_secondstep(encoder=enc, hid=mc["hidden"], z_dim=mc["z_dim"], in_dim=mc["dec_in_dim"],
                                A=mc["A"], decoder=dec, n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"],
                                reinit_decoder_per_block=mc.get("reinit_decoder_per_block", True)).to(DEV)


def ck(rd, cfg):
    d = f"{rd}/checkpoints"; ep = cfg.get("cv_selected_epoch")
    if ep and os.path.exists(f"{d}/epoch{ep:04d}.pt"): return f"{d}/epoch{ep:04d}.pt"
    av = sorted(int(f[5:9]) for f in os.listdir(d) if f.startswith("epoch") and f.endswith(".pt") and "pre_step25" not in f)
    return f"{d}/epoch{av[-1]:04d}.pt"


@torch.no_grad()
def nll_with_z(m, xin, c_t, z):
    h = None; blk = []
    for b in range(xin.shape[1]):
        if m.reinit_decoder_per_block and b > 0: h = None
        lb, h = m.decoder(m._append_task_emb(xin[:, b], b), z, hidden=h); blk.append(lb)
    lp = F.log_softmax(torch.stack(blk, 1), -1); v = (c_t >= 0); cs = c_t.clone(); cs[~v] = 0
    nll = torch.where(v, -lp.gather(-1, cs.long().unsqueeze(-1)).squeeze(-1), torch.tensor(0., device=DEV))
    return (nll.sum((1, 2)) / v.float().sum((1, 2)).clamp(min=1)).cpu().numpy()


def winner(f, metric):
    best = None
    for rd in glob.glob(f"{HP}/fold{f}/*"):
        try: c = json.load(open(f"{rd}/config.json"))
        except: continue
        if metric not in c: continue
        if best is None or c[metric] < best[0]: best = (c[metric], rd, c)
    return best


tid = torch.from_numpy(np.load(f"{DATA}/task_ids_per_block.npy")).long().to(DEV)
for label, metric in [("ELBO-winner (cv_val_loss)", "cv_val_loss"), ("NLL-winner (cv_val_nll)", "cv_val_nll")]:
    gaps, zstds, idr, cps = [], [], [], []
    for f in range(3):
        _, rd, cfg = winner(f, metric)
        m = build(cfg); sd = torch.load(ck(rd, cfg), map_location=DEV)
        sd = sd.get("model_state", sd) if isinstance(sd, dict) else sd
        m.load_state_dict(sd, strict=False); m.set_task_ids(tid); m.eval()
        xin = torch.from_numpy(np.load(f"{DATA}/fold{f}/xin_test.npy")).float().to(DEV)
        c_t = torch.from_numpy(np.load(f"{DATA}/fold{f}/c_test.npy")).float().to(DEV)
        with torch.no_grad():
            mu, _ = m.encoder(xin, return_per_timestep=False)
        ni = nll_with_z(m, xin, c_t, mu).mean()
        n0 = nll_with_z(m, xin, c_t, torch.zeros_like(mu)).mean()
        gaps.append(n0 - ni); zstds.append(float(mu.std(0).mean())); idr.append(ni); cps.append(n0)
    print(f"{label:28s}  held-out IDRNN={np.mean(idr):.4f}  CP-RNN={np.mean(cps):.4f}  "
          f"gap(CP-IDRNN)={np.mean(gaps):+.4f}  z_enc_std={np.mean(zstds):.3f}")
