# VERDICT — do the main results change?

**6 evidence-band change(s) detected:**
- Panel c PANAS_PA (IDRNN): BF band anecdotal+ → anecdotal0 (R² +0.021 → +0.008)
- Panel c CEI (IDRNN): BF band anecdotal0 → null (R² +0.011 → +0.004)
- Panel c BIG5_open (IDRNN): BF band strong+ → moderate+ (R² +0.053 → +0.032)
- Panel c negMood (IDRNN): BF band strong+ → null (R² -0.014 → -0.015)
- Panel d: traits LOSING BF≥3 with PC1: ['CEI']
- Panel g Curiosity: partial-r BF band moderate+ → anecdotal+

# Pooled-consistency report — panels c/d/e/g: S1-only (236) vs pooled S1+S2 (238)

- OLD (current figure): `final_plots/thalmann_z3_3task_full` latents, `data_thalmann_3task_full` behaviour
- NEW (pooled):         `final_plots/thalmann_z3_s2_3task_full` latents (= ds_t012 canonical runs, identical HP z=3/hid=10/temb=4), `data_sub_t012_full` behaviour
- Targets identical in both (pooled S1+S2 questionnaires); cohort differs by 2 S2-only participants (sids 42, 177).

## Panel c — out-of-sample LOO R² (canonical, leakage-free)

| trait | IDRNN R² old→new | BF_gt old→new (band) | Vanilla R² old→new | vanilla BF_gt old→new |
|---|---|---|---|---|
| PANAS_PA | +0.021 → +0.008 | 2 → 0.429 **[anecdotal+→anecdotal0]** | -0.011 → +0.024 | 0.0106 → 3.15 |
| PANAS_NA | +0.016 → +0.020 | 1 → 1.72 | -0.014 → +0.026 | 0.0107 → 3.86 |
| STICSA | -0.012 → -0.003 | 0.0486 → 0.0923 | -0.012 → -0.012 | 0.0163 → 2.48e+11 |
| PHQ | -0.010 → -0.011 | 0.0268 → 0.0295 | -0.013 → -0.011 | 0.032 → 0.00993 |
| CEI | +0.011 → +0.004 | 0.595 → 0.27 **[anecdotal0→null]** | -0.014 → +0.017 | 0.0192 → 1.12 |
| BIG5_open | +0.053 → +0.032 | 87.6 → 7.71 **[strong+→moderate+]** | +0.002 → +0.006 | 0.155 → 0.299 |
| AxDep | -0.014 → -0.014 | 11.2 → 1.1e+03 | -0.018 → -0.016 | 0.0124 → 3.78e+05 |
| posMood | -0.015 → -0.014 | 0.0231 → 0.0138 | -0.014 → -0.014 | 0.342 → 0.012 |
| negMood | -0.014 → -0.015 | 3.51e+04 → 0.0146 **[strong+→null]** | -0.015 → -0.019 | 0.0274 → 0.0125 |
| Exp | -0.019 → -0.018 | 0.0272 → 0.0231 | -0.014 → -0.014 | 0.621 → 0.0325 |
| WM_composite | +0.093 → +0.116 | 788 → 7.85e+03 | +0.026 → +0.070 | 2.42 → 173 |
| WM_OS | -0.015 → -0.012 | 0.0124 → 0.0173 | -0.020 → -0.016 | 0.0228 → 0.0295 |
| WM_SS | +0.076 → +0.111 | 171 → 4.79e+03 | +0.030 → +0.074 | 4.53 → 401 |
| WM_WMU | +0.122 → +0.118 | 1.25e+04 → 9.67e+03 | +0.021 → +0.073 | 1.32 → 241 |

## Panel d — individual-difference axis: r(PC1, trait) + two-sided JZS BF

| trait | r old→new | BF old→new (band) |
|---|---|---|
| PANAS_PA | +0.174 → +0.153 | 2.88 → 1.32 |
| PANAS_NA | +0.182 → +0.190 | 4.02 → 5.91 |
| STICSA | -0.059 → -0.043 | 0.123 → 0.1 |
| PHQ | -0.096 → -0.086 | 0.241 → 0.193 |
| CEI | +0.185 → +0.137 | 4.57 → 0.735 **[moderate+→anecdotal0]** |
| BIG5_open | +0.292 → +0.246 | 2.6e+03 → 119 |
| AxDep | -0.035 → -0.022 | 0.105 → 0.0984 |
| posMood | +0.091 → +0.055 | 0.194 → 0.123 |
| negMood | +0.012 → +0.072 | 0.0957 → 0.148 |
| Exp | +0.108 → +0.066 | 0.259 → 0.137 |
| WM_composite | -0.279 → -0.380 | 98.5 → 7.02e+04 |
| WM_OS | -0.020 → -0.094 | 0.0978 → 0.202 |
| WM_SS | -0.247 → -0.363 | 20.5 → 1.91e+04 |
| WM_WMU | -0.349 → -0.395 | 6.3e+03 → 2.19e+05 |

Traits with BF≥3 — old: ['BIG5_open', 'CEI', 'PANAS_NA', 'WM_SS', 'WM_WMU', 'WM_composite']
Traits with BF≥3 — new: ['BIG5_open', 'PANAS_NA', 'WM_SS', 'WM_WMU', 'WM_composite']

## Panel e — consensus PC1 vs overall switch rate
- old: r = +0.280 (n=236), BF = 1.06e+03 [strong+]
- new: r = +0.508 (n=238), BF = 1.21e+14 [strong+]

## Panel g — trait signal beyond working memory (raw r vs partial r | WM_composite)

| trait | raw r old→new | partial r|WM old→new | BF(partial) old→new (band) |
|---|---|---|---|
| Openness | +0.302 → +0.273 | +0.278 → +0.246 | 88.4 → 20.6 |
| Curiosity | +0.217 → +0.156 | +0.230 → +0.174 | 9.53 → 1.33 **[moderate+→anecdotal+]** |

## Consensus-axis summary (z2_seed_matched_summary.json)
| key | old | new |
|---|---|---|
| exp_pc | None | None |
| r_wm | None | None |
| r_open | None | None |
| r_cei | None | None |
| r_sw_horizon | None | None |

## Coherence check — panel c openness vs panel f t012 endpoint
- panel f t012 openness (IDRNN): R² = +0.0323 (n=238)
- NEW panel c BIG5_open (IDRNN): R² = +0.0323
- These now share models, cohort, targets and readout → expected ≈ equal. Δ = +0.0000

## Panels already pooled (untouched)
- Panel a (held-out NLL): `final_plots/thalmann_z3_s2_pooled/per_participant_nll_marginal.csv`
- Panel b (cross-task regret): `final_plots/thalmann_z3_s2_full/regret/step1_cross_task_regret.npz`
- Panel f (data-scaling): `final_plots/thalmann_z3_datascaling/datascaling_panel.csv`

## Separate workstream — thalmann_s2 cv-val-NLL pipeline (jobs 38506849/38506850)
- `logs/thal_s2_cvnll_38506849.out`: last line: `Epoch 3000,  Train loss: 0.619,  Train accuracy: 0.05110861361026764, Validation accuracy: 0.051951609551906586`
- `logs/thal_s2_cvnll_38506850.out`: last line: `  [seed 85 final] ep 2800  nll=0.5566`
