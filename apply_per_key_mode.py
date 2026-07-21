#!/usr/bin/env python3
"""Per-key-mode canonical-HP spec writer (REPLICATION §2 operational note).

With Optuna HP search, stage_e's modal_hp falls through to fold-0's winner; for
thalmann it actually CRASHES (its grid-scan tie-breaker calls the legacy
thalmann_v3_tag which needs a 'unif_weight' key the standardised grid lacks).

So this script writes the canonical spec directly via per-key mode: for each HP
key take the mode of the values across the 3 per-fold winners, breaking 1/1/1
ties by the fold with the lowest inner-CV NLL.  Non-leaky (only inner-CV winners
are touched; outer test cohorts stay sealed).

Writes:
  final_plots/{dgp}{path_tag}/canonical/{arch}_spec.json
(same payload schema as nested_cv.select_canonical.write_canonical_spec)

Usage:
  python apply_per_key_mode.py --dgp thalmann --hypothesis_z 3 --arch both
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, ".")
from nested_cv.config import REGISTRY, combo_tag, path_tag, N_OUTER_FOLDS


def _cv(rec):
    for k in ("winner_mean_cv_val_loss", "winner_cv_val_loss", "winner_metric_value"):
        if rec.get(k) is not None:
            return float(rec[k])
    return float("inf")


def per_key_mode(fold_winners):
    """fold_winners: list of per-fold winner JSON dicts (each has 'winner')."""
    hps = [w["winner"] for w in fold_winners]
    cvs = [_cv(w) for w in fold_winners]
    keys = sorted(set().union(*[set(h.keys()) for h in hps]))
    out, reason = {}, {}
    for k in keys:
        vals = [h.get(k) for h in hps]
        cnt = Counter(v for v in vals if v is not None)
        maxc = max(cnt.values())
        tied = [v for v, c in cnt.items() if c == maxc]
        if len(tied) > 1:
            # 1/1/1 (or split) tie: pick the value from the lowest-NLL fold
            best_i = min(range(len(hps)),
                         key=lambda i: cvs[i] if hps[i].get(k) in tied else float("inf"))
            out[k] = hps[best_i].get(k)
            reason[k] = f"tie->lowest-NLL fold{best_i}"
        else:
            out[k] = tied[0]
            reason[k] = f"mode {maxc}/{len(hps)}"
    return out, reason


def write_spec(dgp, arch, hypothesis_z, metric, n_seeds):
    tag = path_tag(hypothesis_z, metric)
    wdir = os.path.join("nested_cv_winners", f"{dgp}{tag}", arch)
    fold_winners = []
    for f in range(N_OUTER_FOLDS):
        p = os.path.join(wdir, f"fold{f}.json")
        fold_winners.append(json.load(open(p)))

    hp, reason = per_key_mode(fold_winners)
    if hypothesis_z is not None:
        hp["z"] = hypothesis_z      # keep z anchored

    spec = REGISTRY[dgp]
    dgp_tagged = f"{dgp}{tag}"
    out_dir = os.path.join("final_plots", dgp_tagged, "canonical")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "dgp":          dgp,
        "arch":         arch,
        "dataset_id":   None,
        "hypothesis_z": hypothesis_z,
        "hp":           hp,
        "hp_tag":       combo_tag(hp),
        "selection_reason": ("per-key mode across per-fold winners "
                             f"({'; '.join(f'{k}:{reason[k]}' for k in reason)})"),
        "per_fold_winner_tags": [combo_tag(w["winner"]) for w in fold_winners],
        "per_fold_winner_cv_val_loss": [_cv(w) for w in fold_winners],
        "seeds":        list(spec.final_seeds[:n_seeds]),
        "canonical_runs_dir": f"final_plots/{dgp_tagged}/canonical/{arch}/runs",
    }
    spec_p = os.path.join(out_dir, f"{arch}_spec.json")
    with open(spec_p, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[{arch}] -> {spec_p}")
    print(f"   hp = {hp}  (tag {payload['hp_tag']})")
    print(f"   reason = {payload['selection_reason']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgp", required=True)
    ap.add_argument("--hypothesis_z", type=int, default=None)
    ap.add_argument("--metric", default="cv_val_loss")
    ap.add_argument("--arch", choices=["idrnn", "vanilla", "both"], default="both")
    ap.add_argument("--n_seeds", type=int, default=30)
    args = ap.parse_args()
    archs = ["idrnn", "vanilla"] if args.arch == "both" else [args.arch]
    for a in archs:
        write_spec(args.dgp, a, args.hypothesis_z, args.metric, args.n_seeds)


if __name__ == "__main__":
    main()
