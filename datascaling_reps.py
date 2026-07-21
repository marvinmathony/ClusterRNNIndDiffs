#!/usr/bin/env python3
"""Shared loaders for the LEAKAGE-FREE data-scaling decoding analyses
(compute_datascaling_traintest_bootstrap_perfold.py + compute_datascaling_canonical_bf.py),
which port the main panel's philosophy (compute_loo_r2_canonical_bf.py +
compute_loo_r2_traintest_bootstrap_perfold.py) to the 7 task-constellation models.

Representations per (constellation, arch): (10 seeds, 238, dim)
  idrnn   = step-1 lookup z (dim = Z_DIM = 3); the existing _bootreps_idrnn.npy cache
            is raw z and safe to reuse.
  vanilla = RAW trial-averaged h (dim = hidden, e.g. 15), cached to
            _bootreps_vanilla_rawh.npy. NB the older _bootreps_vanilla.npy is PCA'd on
            the FULL cohort — that leaks the held-out participant into the dim-match,
            so the per-fold scripts must load raw h and PCA inside each LOO fold.

Mask convention (carried over from the previous per-constellation analysis, unlike the
main panel's per-unit masks): per GROUP, keep participants with finite targets for
every group trait AND finite reps across ALL (constellation, arch, seed), so n is
constant across every point of a panel and constellations are compared on the same
cohort.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from modelsandtraining import AblatedRNN

SUBSETS = ["t0", "t1", "t2", "t01", "t02", "t12", "t012"]
NTASK = {"t0": 1, "t1": 1, "t2": 1, "t01": 2, "t02": 2, "t12": 2, "t012": 3}
GROUPS = {"openness": ["BIG5_open"], "wm": ["WM_composite", "WM_WMU", "WM_SS"]}
ARCHES = ["idrnn", "vanilla"]
Z_DIM = 3
ALPHAS = (0.001, 0.01, 0.1, 1, 10, 100, 1000)
EPS = 1e-8
# DS_S1_EXTRACT=1 → S1-extraction convention (matches the main figure's panel c):
# vanilla h is trial-averaged over SESSION-1 BLOCKS ONLY (identical block set for
# every participant, so the representation cannot encode session coverage — see
# pooled retention confound), cohort restricted to has_s1, outputs to a sibling
# dir. Models stay the pooled-trained canonical runs (fully informed). IDRNN z
# (trained lookup) is unchanged.
S1_EXTRACT = os.environ.get("DS_S1_EXTRACT", "0") == "1"
# DS_COHORT_BOTH=1 → both-session cohort (n=175), FULL both-session extraction:
# vanilla's most favourable sessions-regime, used to show task-scaling flatness
# is not an artifact of the S1-extraction convention. Full-extraction caches are
# clean for these participants (all their blocks genuinely present).
COHORT_BOTH = os.environ.get("DS_COHORT_BOTH", "0") == "1"
# DS_CLEAN_EXTRACT=1 → the CLEAN EXTRACTION cohort (canonical name, 2026-07-13):
# every has_s1 participant (n=236); vanilla h = average over existing trials only
# (single-session participants' rows are their true S1-only average — spliced from
# the s1x cache; no zero-block artefacts, no forward passes needed).
CLEAN_EXTRACT = os.environ.get("DS_CLEAN_EXTRACT", "0") == "1"
assert sum([S1_EXTRACT, COHORT_BOTH, CLEAN_EXTRACT]) <= 1, "choose one variant"
OUTDIR = ("final_plots/thalmann_z3_datascaling_s1x" if S1_EXTRACT
          else "final_plots/thalmann_z3_datascaling_both175" if COHORT_BOTH
          else "final_plots/thalmann_z3_datascaling_cleanextr" if CLEAN_EXTRACT
          else "final_plots/thalmann_z3_datascaling")
N_SEEDS = 10


def s1_block_mask(task_ids):
    """First half of each task's blocks = session 1 (S1/S2 contribute equal block
    counts per task in every data_sub_* layout: 30/30, 1/1, 80/80)."""
    m = np.zeros(len(task_ids), bool)
    for k in np.unique(task_ids):
        pos = np.where(task_ids == k)[0]
        m[pos[:len(pos) // 2]] = True
    return m


def fulldir(sub):
    return f"final_plots/thalmann_z3_ds_{sub}"


def datadir(sub):
    return f"data_sub_{sub}_full"


def seed_dirs(sub, arch):
    return sorted(glob.glob(f"{fulldir(sub)}/canonical/{arch}/runs/seed_*"),
                  key=lambda d: int(d.split("seed_")[1]))[:N_SEEDS]


@torch.no_grad()
def vanilla_h_raw(run_dir, dd):
    """RAW trial-averaged hidden state (N, hidden) — identical forward pass to the
    previous datascaling loader (compute_datascaling_bootstrap.vanilla_h), just
    WITHOUT the full-cohort pc_match at the end."""
    cfg = json.load(open(f"{run_dir}/config.json"))["model_config"]
    ck = sorted(glob.glob(f"{run_dir}/checkpoints/epoch*.pt"), key=lambda p: int(p[-7:-3]))[-1]
    sd = torch.load(ck, map_location="cpu")
    sd = sd["model_state"] if isinstance(sd, dict) and "model_state" in sd else sd
    m = AblatedRNN(hid=cfg["hidden"], in_dim=cfg["dec_in_dim"], A=cfg["A"],
                   block_structure=True, n_tasks=cfg["n_tasks"], task_emb_dim=cfg["task_emb_dim"])
    m.set_task_ids(torch.from_numpy(np.load(f"{dd}/task_ids_per_block.npy")).long())
    m.load_state_dict(sd)
    m.eval()
    xin = torch.from_numpy(np.load(f"{dd}/xin_train.npy")).float()
    _, _, h = m(xin)
    h = h.numpy()
    N, Bk, T, H = h.shape
    valid = (xin[..., 0].numpy() != -100.0)
    if S1_EXTRACT:                                # average over S1 blocks only
        bm = s1_block_mask(np.load(f"{dd}/task_ids_per_block.npy"))
        valid = valid & bm[None, :, None]
    valid = valid.reshape(N, -1)
    hf = h.reshape(N, Bk * T, H)
    hav = np.zeros((N, H), np.float32)
    for i in range(N):
        mm = valid[i]
        if mm.sum():
            hav[i] = hf[i, mm].mean(0)
    return hav


def load_reps_raw(verbose=True):
    """(sub, arch) -> (10, 238, dim); dim = 3 (idrnn z) or hidden (RAW vanilla h).
    Vanilla forward passes are cached once per constellation (serial, single-thread)."""
    reps = {}
    for sub in SUBSETS:
        cache_i = f"{fulldir(sub)}/decoding/_bootreps_idrnn.npy"          # raw z, reusable
        if os.path.exists(cache_i):
            reps[(sub, "idrnn")] = np.load(cache_i)
        else:
            reps[(sub, "idrnn")] = np.stack([np.load(f"{d}/step1_z_lookup.npy")
                                             for d in seed_dirs(sub, "idrnn")])
            os.makedirs(os.path.dirname(cache_i), exist_ok=True)
            np.save(cache_i, reps[(sub, "idrnn")])
        cache_v = (f"{fulldir(sub)}/decoding/_bootreps_vanilla_rawh_s1x.npy" if S1_EXTRACT
                   else f"{fulldir(sub)}/decoding/_bootreps_vanilla_rawh.npy")
        if CLEAN_EXTRACT:
            full = np.load(f"{fulldir(sub)}/decoding/_bootreps_vanilla_rawh.npy")
            s1x  = np.load(f"{fulldir(sub)}/decoding/_bootreps_vanilla_rawh_s1x.npy")
            meta = pd.read_csv("data_thalmann_s2/df_all.csv")
            s1only = (meta["has_s1"].values == 1) & (meta["has_s2"].values == 0)
            arr = full.copy(); arr[:, s1only] = s1x[:, s1only]
            reps[(sub, "vanilla")] = arr
        elif os.path.exists(cache_v):
            reps[(sub, "vanilla")] = np.load(cache_v)
        else:
            dd = datadir(sub)
            arr = np.stack([vanilla_h_raw(d, dd) for d in seed_dirs(sub, "vanilla")])
            np.save(cache_v, arr)
            reps[(sub, "vanilla")] = arr
            if verbose:
                print(f"  cached raw vanilla h for {sub}: {arr.shape}", flush=True)
        if verbose:
            print(f"  {sub}: idrnn {reps[(sub, 'idrnn')].shape}  "
                  f"vanilla {reps[(sub, 'vanilla')].shape}", flush=True)
    return reps


def group_mask(tg, gtraits, reps):
    """Shared per-group participant mask: finite targets for all group traits AND
    finite reps for every (constellation, arch, seed). Under S1_EXTRACT the cohort
    is additionally restricted to has_s1 participants (S2-only participants have
    all-zero S1 summaries, which are finite but meaningless)."""
    n_ref = next(iter(reps.values())).shape[1]
    m = np.ones(n_ref, bool)
    for t in gtraits:
        m &= np.isfinite(tg[t].values.astype(float))
    for arr in reps.values():
        for s in range(arr.shape[0]):
            m &= np.all(np.isfinite(arr[s]), axis=1)
    if S1_EXTRACT or COHORT_BOTH or CLEAN_EXTRACT:
        meta = pd.read_csv("data_thalmann_s2/df_all.csv")
        assert len(meta) == n_ref
        m &= (meta["has_s1"].values == 1)
        if COHORT_BOTH:
            m &= (meta["has_s2"].values == 1)
    return m
