"""Aggregate the per-(fold,seed) MARGINALIZED held-out NLL CSVs into the panel-a
inputs:
  per_participant_nll_marginal.csv  — seed-averaged per-subject IDRNN/CP-RNN/Vanilla
      (marginalized+causal) + cog/illspec columns carried over from the point-estimate
      per_participant_nll.csv (cog models are unaffected by the RNN eval).
  per_seed_nll_marginal.csv         — per-seed pooled means (for the robustness panel).
"""
import glob, os
import numpy as np, pandas as pd

D = "final_plots/thalmann_z3_s2_pooled"
MG = "runs_pooled/s2_2task/marginal_nll"

frames = [pd.read_csv(p) for p in sorted(glob.glob(f"{MG}/fold*_seed*.csv"))]
if not frames:
    raise SystemExit(f"no marginal CSVs in {MG} yet")
mg = pd.concat(frames, ignore_index=True)
seeds = sorted(mg["seed"].unique()); folds = sorted(mg["fold"].unique())
print(f"loaded {len(frames)} CSVs: folds={folds} seeds={seeds} rows={len(mg)}")

# per-subject seed-average (each subject is in exactly one fold)
psub = mg.groupby("subid")[["idrnn", "cp_rnn", "vanilla"]].mean().reset_index()
psub = psub.rename(columns={"idrnn": "nll_idrnn_all", "cp_rnn": "nll_cp_rnn_all",
                            "vanilla": "nll_vanilla_all"})
# carry cog/illspec columns from the point-estimate per-participant file
pe = pd.read_csv(f"{D}/per_participant_nll.csv")
cog = pe[["subid", "nll_cog_em", "nll_cog_cp", "nll_ill_em", "nll_ill_cp"]]
out = psub.merge(cog, on="subid", how="left")
out.to_csv(f"{D}/per_participant_nll_marginal.csv", index=False)

# per-seed pooled means (robustness panel)
rows = []
for s in seeds:
    sub = mg[mg["seed"] == s]
    rows.append(dict(seed=s, idrnn=sub["idrnn"].mean(), cp_rnn=sub["cp_rnn"].mean(),
                     vanilla=sub["vanilla"].mean(),
                     cp_minus_idrnn=sub["cp_rnn"].mean() - sub["idrnn"].mean()))
pd.DataFrame(rows).to_csv(f"{D}/per_seed_nll_marginal.csv", index=False)

print(f"\n=== MARGINALIZED held-out NLL (seed-avg over {len(seeds)} seeds, {len(out)} participants) ===")
for k in ["nll_idrnn_all", "nll_cp_rnn_all", "nll_vanilla_all", "nll_cog_em", "nll_cog_cp", "nll_ill_em", "nll_ill_cp"]:
    print(f"  {k:18s} {out[k].mean():.4f}")
g = out["nll_cp_rnn_all"] - out["nll_idrnn_all"]
print(f"  latent gap (CP-RNN − IDRNN): {g.mean():+.4f}   (positive = latent helps)")
print(f"  per-seed gap: {[round(r['cp_minus_idrnn'],4) for r in rows]}")
print(f"\nWrote {D}/per_participant_nll_marginal.csv + per_seed_nll_marginal.csv")
