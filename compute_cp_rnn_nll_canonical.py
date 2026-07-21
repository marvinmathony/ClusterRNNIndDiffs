"""CP-RNN NLL = canonical IDRNN evaluated with z=0 (individual-difference ablation).

For each synthetic dataset: load the canonical z=1 IDRNN (chosen by stage_g),
forward its decoder with z forced to 0 on the full 200-session training cohort
(the same set the EM-fit cog models score on), compute summed NLL per session.

Saves: env_cp_rnn_nll_canonical.csv  with columns dataset_id, session, nll_summed.
This matches the per-participant NLL unit used by the original
plot_publication_figure.py panel a (mean ~100).
"""
import os, json, sys, glob
import numpy as np, pandas as pd, torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from modelsandtraining import IDRNN, Decoder, LatentRNN_secondstep
from nested_cv.config import path_tag

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASETS = [0, 1, 2, 3, 5, 7, 10, 12, 15, 17]


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


def cp_nll_per_session(D):
    """Per-session summed NLL under canonical IDRNN with z=0."""
    choice_p = f"final_plots/synthetic_z1/dataset{D}/canonical/idrnn_choice.json"
    if not os.path.exists(choice_p):
        choice_p = f"final_plots/synthetic_z1/dataset{D}/canonical/idrnn_spec.json"
    obj = json.load(open(choice_p))
    seed = obj.get("selected_seed") or obj.get("seeds", [None])[0]
    run_dir = obj.get("selected_run_dir") or os.path.join(obj["canonical_runs_dir"], f"seed_{seed}")
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    ep = cfg.get("cv_selected_epoch")
    cdir = os.path.join(run_dir, "checkpoints")
    ckpt = (os.path.join(cdir, f"epoch{ep:04d}.pt") if ep and
            os.path.exists(os.path.join(cdir, f"epoch{ep:04d}.pt"))
            else os.path.join(cdir, sorted(os.listdir(cdir))[-1]))

    # Load model
    m = _build_idrnn(cfg)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
    m.eval()

    # Inputs + targets
    xin   = torch.from_numpy(np.load(f"data_dataset{D}/xin_train.npy")).float().to(DEVICE)  # (N, T, in)
    c     = torch.from_numpy(np.load(f"data_dataset{D}/c_train.npy")).long().to(DEVICE)     # (N, T)
    N, T, _ = xin.shape

    # Forward decoder with z=0
    with torch.no_grad():
        z_zero = torch.zeros(N, cfg["model_config"]["z_dim"], device=DEVICE)
        logits, _ = m.decoder(xin, z_zero)   # (N, T, A)
        logp = F.log_softmax(logits, dim=-1)
        chosen = logp.gather(-1, c.clamp(min=0).unsqueeze(-1)).squeeze(-1)   # (N, T)
        mask = (c >= 0).float()
        nll_summed_per_session = -(chosen * mask).sum(dim=1).cpu().numpy()   # (N,)
    return nll_summed_per_session, N, int(mask.sum(dim=1).median().item())


def main():
    rows = []
    for D in DATASETS:
        nll_per_sess, N, T = cp_nll_per_session(D)
        for s, v in enumerate(nll_per_sess):
            rows.append(dict(dataset_id=D, session=s, nll_summed=float(v),
                             n_trials=T))
        print(f"  ds{D}: mean summed NLL={nll_per_sess.mean():.2f}  "
              f"per-trial={nll_per_sess.mean()/T:.4f}  (N={N}, T={T})")
    pd.DataFrame(rows).to_csv("cp_rnn_nll_canonical.csv", index=False)
    print(f"\nSaved -> cp_rnn_nll_canonical.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
