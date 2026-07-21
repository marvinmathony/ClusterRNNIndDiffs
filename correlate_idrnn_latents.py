#!/usr/bin/env python3
"""Direct Pearson + Spearman correlations between IDRNN canonical latents (step-1
z, 236x3) and each questionnaire / working-memory score.

Unlike the LOO-ridge pred-vs-target r (where a negative value is the no-signal
artifact), these are direct latent<->target associations.  Reports, per target:
  - per-dimension Pearson r (+p) and Spearman rho (+p) for z0,z1,z2
  - the strongest single dimension (|Pearson|, |Spearman|)
  - in-sample multiple R (sqrt R^2 of OLS target~z)  [biased up by k=3]
  - LOO ridge R^2 (proper out-of-sample; <=0 => no generalizable signal)

Outputs:
  final_plots/thalmann_z3_full/decoding/idrnn_latent_correlations.csv
  ...console table.
"""
import os, sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import LeaveOneOut

sys.path.insert(0, ".")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
from decode_thalmann_canonical import load_targets, label_for, ALL_KEYS

FULL = os.environ.get("THAL_FULL", "final_plots/thalmann_z3_full")
z = np.load(f"{FULL}/canonical/idrnn/latents_train_step1.npy")            # (236, 3)
bundle = torch.load(f"{FULL}/canonical/idrnn/latents_idrnn_canonical.pt",
                    map_location="cpu", weights_only=False)
subids = np.asarray(bundle["subids"]).astype(int)
Z_DIM = z.shape[1]
targets = load_targets(subids)


def loo_r2(Z, y, alphas=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
    loo = LeaveOneOut(); preds = np.zeros(len(y))
    for tr, te in loo.split(Z):
        mz, sz = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        my, sy = y[tr].mean(), y[tr].std() + 1e-8
        clf = RidgeCV(alphas=list(alphas)).fit((Z[tr]-mz)/sz, (y[tr]-my)/sy)
        preds[te] = clf.predict((Z[te]-mz)/sz) * sy + my
    ss_res = np.sum((y - preds)**2); ss_tot = np.sum((y - y.mean())**2)
    return 1.0 - ss_res/ss_tot, float(pearsonr(y, preds)[0])


def multiple_R(Z, y):
    clf = LinearRegression().fit(Z, y); pred = clf.predict(Z)
    r2 = 1 - np.sum((y-pred)**2)/np.sum((y-y.mean())**2)
    return float(np.sqrt(max(r2, 0)))


rows = []
print(f"{'target':<14}{'n':>4} | " + " ".join(f"z{d}:Pear(Spear)" for d in range(Z_DIM)) +
      f" | {'multR':>6} {'LOO_R2':>7}")
for key in ALL_KEYS:
    y = targets[key].values.astype(float)
    m = np.isfinite(y) & np.all(np.isfinite(z), axis=1)
    n = int(m.sum())
    if n < 30:
        continue
    Zc, yc = z[m], y[m]
    row = {"target": key, "n": n}
    cellparts = []
    for d in range(Z_DIM):
        pr, pp = pearsonr(Zc[:, d], yc)
        sr, sp = spearmanr(Zc[:, d], yc)
        row[f"z{d}_pearson_r"] = pr; row[f"z{d}_pearson_p"] = pp
        row[f"z{d}_spearman_rho"] = sr; row[f"z{d}_spearman_p"] = sp
        star = lambda p: "*" if p < .05 else ""
        cellparts.append(f"{pr:+.2f}{star(pp)}({sr:+.2f}{star(sp)})")
    mR = multiple_R(Zc, yc)
    r2, loo_r = loo_r2(Zc, yc)
    row["multiple_R"] = mR; row["loo_R2"] = r2; row["loo_pred_r"] = loo_r
    # strongest single dim by |pearson| / |spearman|
    pear = [abs(row[f"z{d}_pearson_r"]) for d in range(Z_DIM)]
    spear = [abs(row[f"z{d}_spearman_rho"]) for d in range(Z_DIM)]
    row["best_dim_pearson"] = int(np.argmax(pear)); row["best_abs_pearson"] = max(pear)
    row["best_dim_spearman"] = int(np.argmax(spear)); row["best_abs_spearman"] = max(spear)
    rows.append(row)
    print(f"{key:<14}{n:>4} | " + " ".join(cellparts) + f" | {mR:>6.2f} {r2:>+7.3f}")

df = pd.DataFrame(rows)
out = f"{FULL}/decoding/idrnn_latent_correlations.csv"
df.to_csv(out, index=False)
print(f"\nSaved {out}")
print("\nNote: '*' = p<.05.  Pearson(Spearman) per z-dim.  LOO_R2<=0 => no out-of-sample signal.")
print("Strongest |Pearson| per target:")
for _, r in df.sort_values("best_abs_pearson", ascending=False).iterrows():
    print(f"  {label_for(r['target']).replace(chr(10),' '):<22} "
          f"z{int(r['best_dim_pearson'])} |r|={r['best_abs_pearson']:.2f}  "
          f"multR={r['multiple_R']:.2f}  LOO_R2={r['loo_R2']:+.3f}")
