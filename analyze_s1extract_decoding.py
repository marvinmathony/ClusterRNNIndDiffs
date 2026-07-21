#!/usr/bin/env python3
"""S1-extraction control for the pooled panel-c decoding (retention confound).

Design: models = the pooled ds_t012 canonical runs (fully informed, trained on
S1+S2); vanilla representation = trial-averaged h over SESSION-1 BLOCKS ONLY,
so every analysed participant's summary covers the identical block set and the
has_s2 coverage axis cannot exist in the features. Cohort = has_s1 participants
(n=236). IDRNN z stays the trained lookup (note: informed by S2 behaviour for
the 175 returners — a mild asymmetry in IDRNN's favour, flagged in output).

t012 block layout (222): task0 0-29=S1, 30-59=S2; task1 60=S1, 61=S2;
horizon 62-141=S1, 142-221=S2  →  S1 mask = [0:30] + [60] + [62:142].

Outputs: final_plots/thalmann_z3_s2_3task_full/decoding/s1extract_decoding.csv
         (+ prints table incl. has_s2 decode from S1-extracted h)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import glob, json
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from pingouin import bayesfactor_pearson

from modelsandtraining import AblatedRNN

FULL = "final_plots/thalmann_z3_s2_3task_full"
DATA = "data_sub_t012_full"
ALPHAS = (0.001, 0.01, 0.1, 1, 10, 100, 1000); EPS = 1e-8; ZD = 3

S1_BLOCKS = np.zeros(222, bool)
S1_BLOCKS[0:30] = True; S1_BLOCKS[60] = True; S1_BLOCKS[62:142] = True
assert S1_BLOCKS.sum() == 111


@torch.no_grad()
def vanilla_h(run_dir, block_mask):
    cfg = json.load(open(f"{run_dir}/config.json"))["model_config"]
    ck = sorted(glob.glob(f"{run_dir}/checkpoints/epoch*.pt"), key=lambda p: int(p[-7:-3]))[-1]
    sd = torch.load(ck, map_location="cpu")
    sd = sd.get("model_state", sd)
    m = AblatedRNN(hid=cfg["hidden"], in_dim=cfg["dec_in_dim"], A=cfg["A"],
                   block_structure=True, n_tasks=cfg["n_tasks"], task_emb_dim=cfg["task_emb_dim"])
    m.set_task_ids(torch.from_numpy(np.load(f"{DATA}/task_ids_per_block.npy")).long())
    m.load_state_dict(sd); m.eval()
    xin = torch.from_numpy(np.load(f"{DATA}/xin_train.npy")).float()
    _, _, h = m(xin)
    h = h.numpy(); N, Bk, T, Hd = h.shape
    valid = (xin[..., 0].numpy() != -100.0) & block_mask[None, :, None]
    valid = valid.reshape(N, -1); hf = h.reshape(N, Bk * T, Hd)
    hav = np.zeros((N, Hd), np.float32)
    for i in range(N):
        mm = valid[i]
        if mm.sum():
            hav[i] = hf[i, mm].mean(0)
    return hav


def loo_yhat(Z, y):
    n = len(y); yhat = np.empty(n); idx = np.arange(n)
    for i in range(n):
        tr = idx[idx != i]; Ztr, ytr, zte = Z[tr], y[tr], Z[i]
        if Ztr.shape[1] > ZD:
            p = PCA(n_components=ZD).fit(Ztr); Ztr = p.transform(Ztr); zte = p.transform(zte[None])[0]
        mz, sz = Ztr.mean(0), Ztr.std(0) + EPS; my, sy = ytr.mean(), ytr.std() + EPS
        yhat[i] = RidgeCV(alphas=list(ALPHAS)).fit((Ztr - mz) / sz, (ytr - my) / sy
                                                   ).predict(((zte - mz) / sz)[None])[0] * sy + my
    return yhat


def r2(y, yh):
    return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    seed_dirs = sorted(glob.glob(f"{FULL}/canonical/vanilla/runs/seed_*"),
                       key=lambda d: int(d.split("seed_")[1]))[:10]
    print(f"extracting S1-only h from {len(seed_dirs)} pooled vanilla seeds ...", flush=True)
    H_s1 = np.stack([vanilla_h(d, S1_BLOCKS) for d in seed_dirs])
    np.save(f"{FULL}/decoding/_bootreps_vanilla_rawh_s1extract.npy", H_s1)

    Zi = np.load("final_plots/thalmann_z3_ds_t012/decoding/_bootreps_idrnn.npy")
    tg = pd.read_csv(f"{FULL}/decoding/trait_targets.csv")
    meta = pd.read_csv("data_thalmann_s2/df_all.csv").set_index("subid").loc[tg["subid"]]
    has_s1 = (meta["has_s1"] == 1).values
    s2 = meta["has_s2"].values.astype(float)
    print(f"cohort: has_s1 n={has_s1.sum()}  (of {len(has_s1)})")

    # coverage check: is has_s2 still decodable from S1-extracted h?
    yb = s2[has_s1]
    yh = np.mean([loo_yhat(H_s1[s][has_s1], yb) for s in range(len(H_s1))], axis=0)
    r_cov = float(pearsonr(yb, yh)[0])
    print(f"has_s2 decode from S1-extracted h (behavioural attrition signal): LOO r = {r_cov:+.3f}")

    rows = []
    print(f"\n{'trait':<14}{'arch':<9}{'n':>5}{'R2':>9}{'r':>8}{'BF_gt':>12}")
    for k in ("PANAS_PA", "PANAS_NA", "STICSA", "PHQ", "CEI", "BIG5_open",
              "AxDep", "posMood", "negMood", "Exp",
              "WM_composite", "WM_OS", "WM_SS", "WM_WMU"):
        y = tg[k].values.astype(float)
        sel = has_s1 & np.isfinite(y)
        yv = y[sel]
        for arch, R in (("idrnn", Zi), ("vanilla_s1x", H_s1)):
            yh = np.mean([loo_yhat(R[s][sel], yv) for s in range(R.shape[0])], axis=0)
            rr2 = r2(yv, yh); r = float(np.corrcoef(yh, yv)[0, 1])
            bf = float(bayesfactor_pearson(r, len(yv), alternative="greater"))
            rows.append(dict(trait=k, arch=arch, n=len(yv), r2=rr2, pearson_r=r, bf10_greater=bf))
            print(f"{k:<14}{arch:<9}{len(yv):>5}{rr2:>+9.4f}{r:>+8.3f}{bf:>12.3g}")
    out = f"{FULL}/decoding/s1extract_decoding.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nNote: IDRNN z is the trained lookup (S2-informed for returners); "
          f"vanilla h uses S1 blocks only — asymmetry favours IDRNN.")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
