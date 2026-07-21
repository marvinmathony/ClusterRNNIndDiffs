"""Confirm: held-out IDRNN vs CP-RNN with the PROPER marginalized+causal eval
(compute_rnn_likelihoods_torch -> test_latentrnn_secondstep_causal_posterior_weighting,
id=True=IDRNN marginalised over q(z|x_<=t); id=False=CP-RNN z=0) vs the point-estimate
model(xin,xin) my analyzer used.  One fold/seed, subset of held-out subjects, for speed."""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "."); torch.set_num_threads(4)
from modelsandtraining import (IDRNN, Decoder, LatentRNN_secondstep,
    test_latentrnn_secondstep_causal_posterior_weighting as TESTFN, compute_rnn_likelihoods_torch)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA, RUNS, FOLD, SEED, NSUB, N = "data_thalmann_s2", "runs_pooled/s2_2task/idrnn", 0, 200, 40, 100

rd = f"{RUNS}/fold{FOLD}/seed_{SEED}"; cfg = json.load(open(f"{rd}/config.json")); mc = cfg["model_config"]
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
m.load_state_dict(sd, strict=False)
m.set_task_ids(torch.from_numpy(np.load(f"{DATA}/task_ids_per_block.npy")).long().to(DEV)); m.eval()

xin = torch.from_numpy(np.load(f"{DATA}/fold{FOLD}/xin_test.npy")[:NSUB]).float().to(DEV)
c_t = torch.from_numpy(np.load(f"{DATA}/fold{FOLD}/c_test.npy")[:NSUB]).float().to(DEV)
print(f"fold{FOLD} seed{SEED}, {NSUB} held-out subjects, N={N} importance samples\n")

# --- point estimate (what analyze_thalmann_nested_cv used): z=mu, whole-sequence ---
with torch.no_grad():
    logp = F.log_softmax(m(xin, xin)[0], -1); v = (c_t >= 0); cs = c_t.clone(); cs[~v] = 0
    nll = torch.where(v, -logp.gather(-1, cs.long().unsqueeze(-1)).squeeze(-1), torch.zeros((), device=DEV))
    pe_idrnn = float((nll.sum((1, 2)) / v.float().sum((1, 2)).clamp(min=1)).mean())
    z0 = torch.zeros(xin.shape[0], mc["z_dim"], device=DEV)
    # cp point estimate via decoder z=0 loop
    h = None; blk = []
    for b in range(xin.shape[1]):
        if m.reinit_decoder_per_block and b > 0: h = None
        lb, h = m.decoder(m._append_task_emb(xin[:, b], b), z0, hidden=h); blk.append(lb)
    lp = F.log_softmax(torch.stack(blk, 1), -1)
    nll0 = torch.where(v, -lp.gather(-1, cs.long().unsqueeze(-1)).squeeze(-1), torch.zeros((), device=DEV))
    pe_cp = float((nll0.sum((1, 2)) / v.float().sum((1, 2)).clamp(min=1)).mean())
print(f"POINT ESTIMATE (z=mu, non-causal):   IDRNN={pe_idrnn:.4f}  CP-RNN={pe_cp:.4f}  gap={pe_cp-pe_idrnn:+.4f}")

# --- proper marginalized + causal eval ---
df_id, *_ = compute_rnn_likelihoods_torch(TESTFN, m, xin, c_t, xin, latent=True, N=N, id=True)
df_cp, *_ = compute_rnn_likelihoods_torch(TESTFN, m, xin, c_t, xin, latent=True, N=N, id=False)
mi, mc_ = df_id["normalized_likelihood"].mean(), df_cp["normalized_likelihood"].mean()
print(f"MARGINALIZED+CAUSAL (proper):        IDRNN={mi:.4f}  CP-RNN={mc_:.4f}  gap={mc_-mi:+.4f}")
