"""Per-(fold,seed) MARGINALIZED+CAUSAL held-out NLL for IDRNN / CP-RNN / Vanilla,
the correct evaluation (compute_rnn_likelihoods_torch +
test_latentrnn_secondstep_causal_posterior_weighting).  Saves per-subject NLL so
panel a can aggregate over seeds.  Parameterised so it serves any run tag.

Usage:
  python eval_marginal_heldout.py --idrnn_dir runs_pooled/s2_2task/idrnn/fold{f}/seed_{s} \
      --vanilla_dir runs_pooled/s2_2task/vanilla/fold{f}/seed_{s} \
      --data data_thalmann_s2 --fold F --seed S --out OUT/fold{F}_seed{S}.csv [--N 200] [--n_subj -1]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "."); torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
from modelsandtraining import (IDRNN, Decoder, LatentRNN_secondstep, AblatedRNN,
    test_latentrnn_secondstep_causal_posterior_weighting as TESTFN, compute_rnn_likelihoods_torch)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ckpt(rd, cfg):
    cdir = f"{rd}/checkpoints"; ep = cfg.get("cv_selected_epoch")
    if ep and os.path.exists(f"{cdir}/epoch{ep:04d}.pt"): return f"{cdir}/epoch{ep:04d}.pt"
    av = sorted(f for f in os.listdir(cdir) if f.startswith("epoch") and f.endswith(".pt") and "pre_step25" not in f)
    return f"{cdir}/{av[-1]}"


def _load(rd, kind, tid):
    cfg = json.load(open(f"{rd}/config.json")); mc = cfg["model_config"]
    if kind == "idrnn":
        enc = IDRNN(in_dim=mc["enc_in_dim"], z_dim=mc["z_dim"], hid=mc["enc_hidden"], n_tasks=mc["n_tasks"],
                    task_emb_dim=mc["task_emb_dim"], continuous_encoder=mc.get("continuous_encoder", True))
        dec = Decoder(in_dim=mc["dec_in_dim"], z_dim=mc["z_dim"], hid=mc["hidden"], A=mc["A"])
        m = LatentRNN_secondstep(encoder=enc, hid=mc["hidden"], z_dim=mc["z_dim"], in_dim=mc["dec_in_dim"],
                A=mc["A"], decoder=dec, n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"],
                reinit_decoder_per_block=mc.get("reinit_decoder_per_block", True)).to(DEV)
    else:
        m = AblatedRNN(hid=mc["hidden"], in_dim=mc["dec_in_dim"], A=mc["A"], block_structure=True,
                       n_tasks=mc["n_tasks"], task_emb_dim=mc["task_emb_dim"]).to(DEV)
    sd = torch.load(_ckpt(rd, cfg), map_location=DEV); sd = sd.get("model_state", sd) if isinstance(sd, dict) else sd
    m.load_state_dict(sd, strict=False); m.set_task_ids(tid); m.eval()
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--idrnn_dir", required=True); ap.add_argument("--vanilla_dir", required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=200); ap.add_argument("--n_subj", type=int, default=-1)
    a = ap.parse_args()
    tid = torch.from_numpy(np.load(f"{a.data}/task_ids_per_block.npy")).long().to(DEV)
    sub = pd.read_csv(f"{a.data}/fold{a.fold}/df_test.csv")["subid"].values
    xin = torch.from_numpy(np.load(f"{a.data}/fold{a.fold}/xin_test.npy")).float().to(DEV)
    c_t = torch.from_numpy(np.load(f"{a.data}/fold{a.fold}/c_test.npy")).float().to(DEV)
    if a.n_subj > 0:
        xin, c_t, sub = xin[:a.n_subj], c_t[:a.n_subj], sub[:a.n_subj]
    mi = _load(a.idrnn_dir, "idrnn", tid)
    di, *_ = compute_rnn_likelihoods_torch(TESTFN, mi, xin, c_t, xin, latent=True, N=a.N, id=True)
    dc, *_ = compute_rnn_likelihoods_torch(TESTFN, mi, xin, c_t, xin, latent=True, N=a.N, id=False)
    mv = _load(a.vanilla_dir, "vanilla", tid)
    dv, *_ = compute_rnn_likelihoods_torch(TESTFN, mv, xin, c_t, xin, latent=False, N=a.N, id=False)
    out = pd.DataFrame({"subid": sub, "fold": a.fold, "seed": a.seed,
                        "idrnn": di["normalized_likelihood"].values,
                        "cp_rnn": dc["normalized_likelihood"].values,
                        "vanilla": dv["normalized_likelihood"].values})
    os.makedirs(os.path.dirname(a.out), exist_ok=True); out.to_csv(a.out, index=False)
    print(f"saved {a.out}: IDRNN={out.idrnn.mean():.4f} CP={out.cp_rnn.mean():.4f} "
          f"Vanilla={out.vanilla.mean():.4f} gap(CP-IDRNN)={out.cp_rnn.mean()-out.idrnn.mean():+.4f}")
