"""Per-DGP configuration for nested CV.

Layout: each dataset declares
  - the HP grid for IDRNN and Vanilla
  - fixed run flags (task_emb_dim, same_enc_dec, continuous_encoder, step1_epochs, ...)
  - which inner-search runs already exist (so we can reuse them) vs need to be launched
  - canonical seeds for stage A (HP search, ~3) and stage C (final per-fold, ~30)

The HP-search code consumes these dicts to (a) emit SLURM jobs and (b) read
cv_val_loss back out from each combo's seed_*/config.json.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Generic seed pools ─────────────────────────────────────────────────────────
INNER_SEEDS = [42, 123, 456]              # for HP search (per fold × combo)
FINAL_SEEDS = [200, 300, 400, 500, 600, 999,
               2021, 2022, 2023, 2024, 2025,
               401, 402, 403, 404, 405, 406, 407, 408, 409,
               410, 411, 412, 413, 414, 415, 416, 417, 418, 419]
N_OUTER_FOLDS = 3


def combo_tag(hp: Dict[str, Any]) -> str:
    """Stable, filesystem-safe tag for an HP combo.  Keys are sorted; floats use
    a compact representation (5e-2 -> 005, 1.0 -> 1)."""
    def _fmt_val(v):
        if isinstance(v, bool):
            return "T" if v else "F"
        if isinstance(v, float):
            return ("%g" % v).replace(".", "p").replace("-", "m")
        return str(v)
    return "_".join(f"{k}{_fmt_val(hp[k])}" for k in sorted(hp.keys()))


def cartesian(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """List of dicts spanning the cartesian product of the grid values."""
    keys = list(grid.keys())
    out, stack = [{}], 0
    for k in keys:
        out2 = []
        for d in out:
            for v in grid[k]:
                d2 = dict(d); d2[k] = v; out2.append(d2)
        out = out2
    return out


@dataclass
class DGPSpec:
    name: str
    archs: List[str]                              # ["idrnn", "vanilla"]
    fixed_flags: Dict[str, Dict[str, Any]]        # arch -> CLI flags shared by every combo
    hp_grid: Dict[str, Dict[str, List[Any]]]     # arch -> {flag_name: [values]}
    reuse_runs: Dict[str, Optional[str]]          # arch -> existing dir pattern w/ {combo} placeholder; None = launch fresh
    final_runs_template: Dict[str, str]           # arch -> final per-fold dir pattern w/ {fold} (relative to repo root)
    canonical_template: str = "final_plots/{dgp}/canonical/{arch}"
    inner_seeds: List[int] = field(default_factory=lambda: list(INNER_SEEDS))
    final_seeds: List[int] = field(default_factory=lambda: list(FINAL_SEEDS))
    synth_dataset_ids: Optional[List[int]] = None  # only for synthetic DGPs

    def combos(self, arch: str) -> List[Dict[str, Any]]:
        return cartesian(self.hp_grid[arch])


# ── Standardised HP search space (shared across DGPs) ──────────────────────────
# IDRNN HPs that are searched for every DGP (Optuna treats each as categorical).
STD_IDRNN_HPS: Dict[str, List[Any]] = {
    "lmbd":         [0.001, 0.01, 0.05, 0.2],   # KL weight on z
    "z":            [1, 2, 3, 5, 8],            # latent dim
    "hidden":       [5, 10, 15],                # decoder GRU width
    "step1_epochs": [200, 1000, 3000],          # decoder pretraining length
    "epochs":       [1000, 3000],               # step-2 length
}
# Vanilla searches only `hidden` over the same sizes as IDRNN.
STD_VANILLA_HPS: Dict[str, List[Any]] = {
    "hidden": [5, 10, 15],
}
# Cartesian-product sizes:
#   IDRNN:    4 * 5 * 3 * 3 * 2 = 360  (× 2 if step2_5 is added → 720 for dezfouli)
#   Vanilla:  3
# Optuna's TPE samples adaptively; a fold's budget needs to be only a fraction
# of the cartesian size to converge.  Vanilla searches only 3 combos so a small
# budget suffices (trial-cache + dedup in optuna_search prevents redundant retrains).
N_OPTUNA_TRIALS_IDRNN   = 80
N_OPTUNA_TRIALS_VANILLA = 6
N_OPTUNA_WORKERS_IDRNN   = 16   # matches QOS-16 concurrent limit
N_OPTUNA_WORKERS_VANILLA = 2    # vanilla searches only 3 combos; dedup inside
                                # the trial catches duplicates, so 2 workers
                                # running 3 trials each is enough.
N_OPTUNA_WORKERS = N_OPTUNA_WORKERS_IDRNN  # back-compat alias

# Back-compat alias (some external scripts / CLI defaults import N_OPTUNA_TRIALS).
N_OPTUNA_TRIALS = N_OPTUNA_TRIALS_IDRNN


def trial_budget_for(spec: 'DGPSpec', arch: str) -> int:
    """Per-arch HP-search budget.  Auto-clamped to 2× the cartesian grid size so
    we never request more trials than the search space has unique combos."""
    if arch == "idrnn":
        base = N_OPTUNA_TRIALS_IDRNN
    else:
        base = N_OPTUNA_TRIALS_VANILLA
    grid_size = len(cartesian(spec.hp_grid[arch]))
    return min(base, max(grid_size * 2, grid_size))


def workers_for(arch: str) -> int:
    """Per-arch SLURM worker count.  Vanilla uses fewer workers since its
    grid has only 3 unique combos."""
    return N_OPTUNA_WORKERS_IDRNN if arch == "idrnn" else N_OPTUNA_WORKERS_VANILLA


def z_tag(hypothesis_z: Optional[int]) -> str:
    """Path/study suffix when z is anchored.  Empty string when z is free.

    With hypothesis_z=1, all stage_a Optuna studies, trial run dirs, winner
    JSONs, stage_c run dirs, and canonical-model dirs get an `_z1` suffix so
    they don't collide with the free-z sweep results.
    """
    return f"_z{hypothesis_z}" if hypothesis_z is not None else ""


# Valid Optuna objectives.  cv_val_loss is the standard NLL-based selection;
# step1_specificity selects HPs by lookup-encoder specificity, which we showed
# correlates with α-decoding R² in the z=1 regime (r=+0.47).
OPTUNA_OBJECTIVES = ("cv_val_loss", "step1_specificity")


def objective_tag(objective: Optional[str]) -> str:
    """Path/study suffix when a non-default objective is used."""
    if objective is None or objective == "cv_val_loss":
        return ""
    if objective == "step1_specificity":
        return "_spec"
    return f"_{objective}"


def path_tag(hypothesis_z: Optional[int] = None,
              objective: Optional[str] = None) -> str:
    """Combined suffix for both z anchoring and objective override."""
    return z_tag(hypothesis_z) + objective_tag(objective)


def effective_hp_grid(spec: 'DGPSpec', arch: str,
                       hypothesis_z: Optional[int] = None) -> Dict[str, List[Any]]:
    """Return the HP grid for `(spec, arch)` with z anchored to a single value
    if hypothesis_z is set.  Otherwise returns spec.hp_grid[arch] unchanged.

    Use this when you want to sweep the remaining HPs at a fixed z (e.g. for
    representational-fidelity analyses where z is set by hypothesis rather
    than by NLL).
    """
    grid = dict(spec.hp_grid[arch])
    if hypothesis_z is not None and "z" in grid:
        grid["z"] = [hypothesis_z]
    return grid


# ── Dezfouli ────────────────────────────────────────────────────────────────────
# Same standard IDRNN grid + step2_5_epochs as the dezfouli-specific 7th HP.
# All other DGPs share the 5-HP grid (no step 2.5).
_DEZ_IDRNN = dict(STD_IDRNN_HPS); _DEZ_IDRNN["step2_5_epochs"] = [0, 400]
DEZFOULI = DGPSpec(
    name="dezfouli",
    archs=["idrnn", "vanilla"],
    fixed_flags={
        "idrnn": {
            "dgp": "dezfouli",
            "latent": "True",
            "same_enc_dec": "True",
            "step2_5_lr": 5e-4,
        },
        "vanilla": {
            "dgp": "dezfouli",
            "latent": "False",
            "epochs": 3000,
        },
    },
    hp_grid={"idrnn": _DEZ_IDRNN, "vanilla": dict(STD_VANILLA_HPS)},
    reuse_runs={"idrnn": None, "vanilla": None},
    final_runs_template={
        "idrnn":   "runs_dezfouli_nested_cv/fold{fold}",
        "vanilla": "runs_vanilla_dezfouli_nested_cv/fold{fold}",
    },
)


# ── Thalmann ───────────────────────────────────────────────────────────────────
# Standard grid; thalmann keeps task_emb_dim=4 + continuous_encoder=True fixed.
# `unif_weight` is dropped from the standardised grid; if you want it back,
# add it to STD_IDRNN_HPS.
THALMANN = DGPSpec(
    name="thalmann",
    archs=["idrnn", "vanilla"],
    fixed_flags={
        "idrnn": {
            "dgp": "thalmann",
            "latent": "True",
            "task_emb_dim": 4,
            "same_enc_dec": "False",
            "continuous_encoder": "True",
        },
        "vanilla": {
            "dgp": "thalmann",
            "latent": "False",
            "epochs": 3000,          # intentional fixed cap (matched-budget baseline)
            "task_emb_dim": 4,
        },
    },
    hp_grid={"idrnn": dict(STD_IDRNN_HPS), "vanilla": dict(STD_VANILLA_HPS)},
    reuse_runs={"idrnn": None, "vanilla": None},   # previous hp_v3 sweep no longer
                                                    # matches the standardised grid
    final_runs_template={
        "idrnn":   "runs_thalmann_nested_cv/fold{fold}",
        "vanilla": "runs_vanilla_thalmann_nested_cv/fold{fold}",
    },
)


# ── Synthetic ──────────────────────────────────────────────────────────────────
SYNTHETIC = DGPSpec(
    name="synthetic",
    archs=["idrnn", "vanilla"],
    fixed_flags={
        "idrnn": {
            "latent": "True",
            "same_enc_dec": "True",
        },
        "vanilla": {
            "latent": "False",
            "epochs": 3000,
        },
    },
    hp_grid={"idrnn": dict(STD_IDRNN_HPS), "vanilla": dict(STD_VANILLA_HPS)},
    reuse_runs={"idrnn": None, "vanilla": None},
    final_runs_template={
        "idrnn":   "runs_dataset{dataset_id}_nested_cv/fold{fold}",
        "vanilla": "runs_vanilla_dataset{dataset_id}_nested_cv/fold{fold}",
    },
    synth_dataset_ids=[0, 1, 2, 3, 5, 7, 10, 12, 15, 17],
)


REGISTRY: Dict[str, DGPSpec] = {
    "dezfouli":  DEZFOULI,
    "thalmann":  THALMANN,
    "synthetic": SYNTHETIC,
}


# ── Thalmann hp_v3 tag adapter ─────────────────────────────────────────────────
# Translate one of our generic combo dicts to the existing hp_v3 directory name.
def thalmann_v3_tag(hp: Dict[str, Any]) -> str:
    uw_map  = {0.0: "00", 0.1: "01", 0.5: "05"}
    lmb_map = {0.05: "005", 0.1: "01", 0.2: "02"}
    return (f"uw{uw_map[hp['unif_weight']]}"
            f"_lmbd{lmb_map[hp['lmbd']]}"
            f"_eh{hp['enc_hidden']}"
            f"_h{hp['hidden']}"
            f"_z{hp['z']}")
