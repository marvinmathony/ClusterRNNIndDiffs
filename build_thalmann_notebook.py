#!/usr/bin/env python3
"""Build thalmann_results.ipynb — visualises every analysis for the z=3 models
(S1 and pooled S1+S2), reading the saved CSVs/npz/figures."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md("# Thalmann z=3 — results\n"
   "Individual-difference IDRNN (z=3) vs dimension-matched Vanilla, across tasks "
   "(2-armed + restless; horizon held out). Two canonical models: **S1** "
   "(`thalmann_z3_full`) and **pooled S1+S2** (`thalmann_z3_s2_full`). "
   "Panel a (held-out NLL) is from the S1 nested-CV.")

code(
"import os, numpy as np, pandas as pd\n"
"import matplotlib.pyplot as plt\n"
"from scipy import stats\n"
"from IPython.display import Image, display, Markdown\n"
"MODELS = {'S1 (session 1)':'final_plots/thalmann_z3_full',\n"
"          'Pooled S1+S2':'final_plots/thalmann_z3_s2_full'}\n"
"COL_I, COL_V = '#0272b2', '#ec6f00'\n"
"def show(p, w=950):\n"
"    display(Image(filename=p, width=w)) if os.path.exists(p) else print('missing:', p)")

md("## 1. Composite figure (panels a/b/c/e)")
code("for n,b in MODELS.items():\n"
     "    display(Markdown(f'### {n}'))\n"
     "    show(f'{b}/publication_figure/composite.png')")

md("## 2. Panel b — decodability, two framings\n"
   "**LOO R² > 0 = genuine out-of-sample signal.** The LOO prediction–target *r* is "
   "shown too: its large negatives are the no-information anti-correlation artifact "
   "(not signal). IDRNN carries signal on working memory + openness; Vanilla does not.")
code("for n,b in MODELS.items():\n"
     "    display(Markdown(f'### {n}: LOO R² (left) vs LOO pred–target r (right)'))\n"
     "    show(f'{b}/publication_figure/panel_b_r2.png', 700); show(f'{b}/publication_figure/panel_b_loor.png', 700)")

md("## 3. Seed-averaged decodability (selection-free)\n"
   "Mean ± SD of LOO R² across 10 seeds of the final model — no seed selection. "
   "`frac_pos` = fraction of seeds with R²>0 (robustness).")
code("for n,b in MODELS.items():\n"
     "    df = pd.read_csv(f'{b}/decoding/seed_averaged_decoding.csv')\n"
     "    display(Markdown(f'### {n}'))\n"
     "    display(df[['target','idrnn_loo_r2_mean','idrnn_loo_r2_sd','idrnn_frac_pos','vanilla_loo_r2_mean','vanilla_frac_pos']].round(3))\n"
     "    x=np.arange(len(df)); w=.38; fig,ax=plt.subplots(figsize=(11,3.4))\n"
     "    ax.bar(x-w/2, df.idrnn_loo_r2_mean, w, yerr=df.idrnn_loo_r2_sd, color=COL_I, capsize=2, label='IDRNN z')\n"
     "    ax.bar(x+w/2, df.vanilla_loo_r2_mean, w, yerr=df.vanilla_loo_r2_sd, color=COL_V, capsize=2, label='Vanilla h (dim-matched)')\n"
     "    ax.axhline(0,color='k',lw=.8); ax.set_xticks(x); ax.set_xticklabels(df.target, rotation=45, ha='right', fontsize=7)\n"
     "    ax.set_ylabel('LOO R² (mean±SD, 10 seeds)'); ax.set_title(n); ax.legend(fontsize=8); plt.show()")

md("## 4. IDRNN latent ↔ trait correlations\n"
   "Direct Pearson/Spearman per z-dimension, multiple-R, and LOO R² (the honest metric).")
code("for n,b in MODELS.items():\n"
     "    df = pd.read_csv(f'{b}/decoding/idrnn_latent_correlations.csv')\n"
     "    keep=[c for c in df.columns if any(k in c for k in ['target','pearson_r','spearman_rho','multiple_R','loo_R2'])]\n"
     "    display(Markdown(f'### {n}')); display(df[keep].round(3))")

md("## 5. Panel e — generative cross-task regret (R1→R4)\n"
   "R1 human · R2 exact-env single-seed · R3 exact-env marg-RNG · R4 marg-env+RNG. "
   "Marginalisation recovers the transfer signal the single exact rollout misses.")
code("for n,b in MODELS.items():\n"
     "    display(Markdown(f'### {n}')); show(f'{b}/publication_figure/panel_e.png', 520)")

md("## 6. Regret → held-out target: prediction matrix (both source tasks)\n"
   "Robustness to the fixed-schedule design: predict horizon regret + WM + questionnaires "
   "from **task 0** (top rows) and **task 1** (bottom rows) regret. R4 marginalises over "
   "counterfactual environments, so it does not use the shared schedule.")
code("for n,b in MODELS.items():\n"
     "    display(Markdown(f'### {n}')); show(f'{b}/regret/regret_prediction_matrix.png', 1150)")

md("### 6b. Headline: horizon & WM prediction — task0 vs task1, human (R1) vs counterfactual (R4)")
code("rows=[]\n"
     "for n,b in MODELS.items():\n"
     "    df=pd.read_csv(f'{b}/regret/regret_prediction_matrix.csv')\n"
     "    for tgt in ['horizon (all)','WM composite','WM updating']:\n"
     "        for src in ['task0','task1']:\n"
     "            for reg in ['R1 human','R4 marg-env']:\n"
     "                s=df[(df.source==src)&(df.regret==reg)&(df.target==tgt)]\n"
     "                if len(s): rows.append(dict(model=n,target=tgt,source=src,regret=reg,r=round(s.pearson_r.iloc[0],3),p=round(s.pearson_p.iloc[0],4)))\n"
     "display(pd.DataFrame(rows).pivot_table(index=['model','target'],columns=['source','regret'],values='r'))")

md("## 6c. Generative prediction — 9 panels (pooled S1+S2)\n"
   "{held-out horizon regret, working memory, openness} × {task0+task1 combined, "
   "task0 only, task1 only}, each with all four regressions R1–R4. "
   "Combined predictor = z-scored mean of task0 & task1 regret.")
code("show('final_plots/thalmann_z3_s2_full/regret/generative_prediction/composite_3x3.png', 1100)")
code("sdf=pd.read_csv('final_plots/thalmann_z3_s2_full/regret/generative_prediction/generative_prediction_summary.csv')\n"
     "for tk in ['horizon_regret','working_memory','openness']:\n"
     "    display(Markdown(f'**{tk}** (Pearson r; bold/star = p<.05)'))\n"
     "    display(sdf[sdf.target==tk].pivot_table(index='source',columns='regression',values='r').round(3))")
md("Also saved as individual panels in "
   "`final_plots/thalmann_z3_s2_full/regret/generative_prediction/<target>__<source>.png/pdf`.")

md("## 6d. Held-out horizon regret from **task0 only** — R1–R4 + Steiger(R4 vs human)\n"
   "Standalone & self-contained (loads its own data), in the `synthetic_publication_panels` "
   "aesthetic (Nature palette/fonts, bootstrap-CI error bars, bootstrap dot cloud, per-bar "
   "BF₁₀). The Steiger test between R4 (model counterfactual) and R1 (human raw) is annotated "
   "as a bracket in the same style as the Bayes factors. Pooled S1+S2.")
code(r'''# Held-out HORIZON regret predicted from TASK0 only (pooled S1+S2), synthetic aesthetic.
import numpy as np, matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from pingouin import bayesfactor_pearson
from nature_plot_style import nature_colors
plt.rcParams.update({"pdf.fonttype":42,"font.family":"sans-serif",
    "font.sans-serif":["Nimbus Sans","DejaVu Sans"],"font.size":8,"axes.titlesize":8,
    "axes.titleweight":"bold","axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,
    "legend.frameon":False,"axes.linewidth":1,"axes.spines.top":False,"axes.spines.right":False})
COL_TRUE=nature_colors["Grey"][5]; COL_IDRNN=nature_colors["Blue"][3]
def fmt_bf(bf):
    return "—" if (bf is None or not np.isfinite(bf)) else (f"{bf:.1e}" if bf>=1000 else (f"{bf:.1f}" if bf>=10 else f"{bf:.2f}"))
def bracket(ax,x1,x2,y,h,txt):
    ax.plot([x1,x2],[y+h,y+h],lw=0.5,c="black")
    ax.text((x1+x2)/2,y+h*1.1,txt,ha="center",va="bottom",fontsize=6)
def steiger(r12,r13,r23,n):
    r12,r13,r23=np.clip([r12,r13,r23],-0.9999,0.9999)
    z12,z13=np.arctanh(r12),np.arctanh(r13); rbar=(r12**2+r13**2)/2
    f=(1-r23)/(2*(1-rbar)); h=(1-f*rbar)/(1-rbar); var=(2*(1-r23)/(n-1))*h
    if var<=0: return 0.0,1.0
    z=(z12-z13)/np.sqrt(var); return float(z), float(2*stats.norm.sf(abs(z)))

d=np.load("final_plots/thalmann_z3_s2_full/regret/step1_cross_task_regret.npz")
y=d["hum_regret_task3"]
BARS=[("R1\nhuman",            d["hum_regret_task0"],                   COL_TRUE,  1.00),
      ("R2\nexact·1seed",  d["sim_regret_task0_r2_exact_single"],   COL_IDRNN, 0.45),
      ("R3\nexact·margRNG",d["sim_regret_task0_r3_exact_margrng"],  COL_IDRNN, 0.70),
      ("R4\nmarg env+RNG",      d["sim_regret_task0_r4_marg"],           COL_IDRNN, 1.00)]
recs=[]; rng=np.random.default_rng(1)
for lab,x,col,al in BARS:
    m=np.isfinite(x)&np.isfinite(y); xv,yv=x[m],y[m]; nb=len(xv)
    r=float(pearsonr(xv,yv)[0]); boots=np.empty(2000)
    for i in range(2000):
        idx=rng.integers(0,nb,nb); boots[i]=np.corrcoef(xv[idx],yv[idx])[0,1]
    lo,hi=np.nanpercentile(boots,[2.5,97.5])
    recs.append(dict(label=lab,color=col,alpha=al,r=r,ci=(lo,hi),boots=boots,bf=float(bayesfactor_pearson(r,nb)),n=nb))

fig,ax=plt.subplots(figsize=(4.5,4.5))
xs=np.arange(len(recs)); heights=[r["r"] for r in recs]
lo_err=[r["r"]-r["ci"][0] for r in recs]; hi_err=[r["ci"][1]-r["r"] for r in recs]
for xi,r in zip(xs,recs):
    ax.bar(xi,r["r"],color=r["color"],alpha=r["alpha"],edgecolor="black",linewidth=0.5,zorder=2)
ax.errorbar(xs,heights,yerr=[lo_err,hi_err],fmt="none",ecolor="#444444",elinewidth=0.7,capsize=0,zorder=3)
rng_pts=np.random.default_rng(7)
for xi,r in zip(xs,recs):
    s=rng_pts.choice(r["boots"],80,replace=False); j=rng_pts.normal(0,0.05,80)
    ax.scatter(xi+j,s,s=12,c="black",alpha=0.4,edgecolors="none",zorder=4)
allb=np.concatenate([r["boots"] for r in recs]); y_top=float(np.nanpercentile(allb,99))+0.04
for xi,r in zip(xs,recs):
    ax.text(xi,y_top,fmt_bf(r["bf"]),ha="center",fontsize=7,color="#333333")
# Steiger test: R4 (idx 3) vs human R1 (idx 0), dependent correlations sharing horizon
mm=np.isfinite(y)&np.isfinite(d["hum_regret_task0"])&np.isfinite(d["sim_regret_task0_r4_marg"]); n=int(mm.sum())
ry1=pearsonr(d["hum_regret_task0"][mm],y[mm])[0]; ry4=pearsonr(d["sim_regret_task0_r4_marg"][mm],y[mm])[0]
r14=pearsonr(d["hum_regret_task0"][mm],d["sim_regret_task0_r4_marg"][mm])[0]
z,p=steiger(ry1,ry4,r14,n)
star="***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "n.s."
bracket(ax,0,3,y_top+0.06,0.03,f"Steiger R1 vs R4: z={z:.2f}, p={p:.3f} ({star})")
ax.axhline(0,color="grey",lw=0.5,zorder=1)
ax.set_xticks(xs); ax.set_xticklabels([r["label"] for r in recs],fontsize=7)
ax.set_ylabel("Pearson r  (task0 regret → horizon regret)")
ax.set_title("Held-out horizon regret from task0 (pooled S1+S2)",loc="left")
ax.set_ylim(min(0,min(heights))-0.05, y_top+0.18)
plt.tight_layout(); plt.show()
print(f"per-bar BF10: " + ", ".join(f"{r['label'].splitlines()[0]}={fmt_bf(r['bf'])}" for r in recs))
print(f"Steiger R1(human) vs R4(model marg-env): z={z:.2f}, p={p:.3f} ({star}), n={n}")
''')

md("## 6e. Same panel — human task0 observations from **Session 1 only**\n"
   "Tests whether the human's edge over the model (R1 > R4 in 6d) is just its 2-session "
   "reliability advantage. R1 = human task0 regret from Session 1 only; model bars (R2–R4) "
   "and the held-out horizon target unchanged. **The gap closes** — R4 (model) now matches/"
   "exceeds the single-session human R1. Reuses helpers from cell 6d.")
code(r'''# R1 (human task0 regret) recomputed from SESSION 1 ONLY; R2-R4 + horizon target as in 6d.
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pingouin import bayesfactor_pearson
d=np.load("final_plots/thalmann_z3_s2_full/regret/step1_cross_task_regret.npz")
subids=d["subids"].astype(int); y=d["hum_regret_task3"]
ab=pd.read_csv("data/final2armedBanditSession1.csv"); o={}
for s,g in ab.groupby("ID"):
    a=g[["reward1","reward2"]].values.astype(float); o[int(s)]=float((a.max(1)-g["reward"].values).mean())
R1=np.array([o.get(int(s),np.nan) for s in subids])
BARS=[("R1\nhuman (S1)",      R1,                                      COL_TRUE,  1.00),
      ("R2\nexact·1seed",  d["sim_regret_task0_r2_exact_single"],   COL_IDRNN, 0.45),
      ("R3\nexact·margRNG",d["sim_regret_task0_r3_exact_margrng"],  COL_IDRNN, 0.70),
      ("R4\nmarg env+RNG",      d["sim_regret_task0_r4_marg"],           COL_IDRNN, 1.00)]
recs=[]; rng=np.random.default_rng(1)
for lab,x,col,al in BARS:
    m=np.isfinite(x)&np.isfinite(y); xv,yv=x[m],y[m]; nb=len(xv)
    r=float(pearsonr(xv,yv)[0]); boots=np.empty(2000)
    for i in range(2000):
        idx=rng.integers(0,nb,nb); boots[i]=np.corrcoef(xv[idx],yv[idx])[0,1]
    lo,hi=np.nanpercentile(boots,[2.5,97.5])
    recs.append(dict(label=lab,color=col,alpha=al,r=r,ci=(lo,hi),boots=boots,bf=float(bayesfactor_pearson(r,nb))))
fig,ax=plt.subplots(figsize=(4.5,4.5)); xs=np.arange(4); heights=[r["r"] for r in recs]
lo_err=[r["r"]-r["ci"][0] for r in recs]; hi_err=[r["ci"][1]-r["r"] for r in recs]
for xi,r in zip(xs,recs): ax.bar(xi,r["r"],color=r["color"],alpha=r["alpha"],edgecolor="black",linewidth=0.5,zorder=2)
ax.errorbar(xs,heights,yerr=[lo_err,hi_err],fmt="none",ecolor="#444444",elinewidth=0.7,capsize=0,zorder=3)
rp=np.random.default_rng(7)
for xi,r in zip(xs,recs):
    sm=rp.choice(r["boots"],80,replace=False); ax.scatter(xi+rp.normal(0,0.05,80),sm,s=12,c="black",alpha=0.4,edgecolors="none",zorder=4)
allb=np.concatenate([r["boots"] for r in recs]); y_top=float(np.nanpercentile(allb,99))+0.04
for xi,r in zip(xs,recs): ax.text(xi,y_top,fmt_bf(r["bf"]),ha="center",fontsize=7,color="#333333")
mm=np.isfinite(R1)&np.isfinite(y)&np.isfinite(d["sim_regret_task0_r4_marg"]); n=int(mm.sum())
ry1=pearsonr(R1[mm],y[mm])[0]; ry4=pearsonr(d["sim_regret_task0_r4_marg"][mm],y[mm])[0]; r14=pearsonr(R1[mm],d["sim_regret_task0_r4_marg"][mm])[0]
z,p=steiger(ry1,ry4,r14,n); star="***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "n.s."
bracket(ax,0,3,y_top+0.06,0.03,f"Steiger R1 vs R4: z={z:.2f}, p={p:.3f} ({star})")
ax.axhline(0,color="grey",lw=0.5,zorder=1); ax.set_xticks(xs); ax.set_xticklabels([r["label"] for r in recs],fontsize=7)
ax.set_ylabel("Pearson r  (task0 regret -> horizon regret)")
ax.set_title("Held-out horizon from task0 — human obs = Session 1 only",loc="left")
ax.set_ylim(min(0,min(heights))-0.05,y_top+0.18); plt.tight_layout(); plt.show()
print(f"R1(human S1)={ry1:+.3f}  R4(model)={ry4:+.3f}  gap={ry1-ry4:+.3f}; Steiger z={z:.2f} p={p:.3f} ({star}), n={n}")
''')

md("## 7. Panel a — held-out NLL per trial (S1 nested-CV, all 500 trials)")
code("df=pd.read_csv('final_plots/thalmann_z3/per_participant_nll.csv')\n"
     "cols={'Ill-spec Q-CP':'nll_ill_cp','Ill-spec Q-EM':'nll_ill_em','Cog CP':'nll_cog_cp','Cog EM':'nll_cog_em','CP-RNN':'nll_cp_rnn_all','IDRNN':'nll_idrnn_all','Vanilla':'nll_vanilla_all'}\n"
     "m={k:df[v].mean() for k,v in cols.items() if v in df}\n"
     "display(pd.Series(m, name='mean NLL/trial').round(3))\n"
     "fig,ax=plt.subplots(figsize=(7,3)); ax.bar(list(m), list(m.values()), color='#5799d1'); ax.axhline(np.log(2),ls=':',color='grey')\n"
     "ax.set_ylabel('NLL/trial'); ax.set_ylim(0.45,0.75); plt.xticks(rotation=20); plt.title('Held-out NLL (lower=better)'); plt.show()")

md("## 8. Does adding the horizon task improve the readout? (2-task vs 3-task)\n"
   "Separate 3-task model (2-armed + restless + horizon); horizon stays held-out for "
   "transfer via the 2-task model, and WM/questionnaire targets are external, so this is "
   "not circular. Seed-averaged LOO R².")
code("a=pd.read_csv('final_plots/thalmann_z3_s2_full/decoding/seed_averaged_decoding.csv').set_index('target')\n"
     "b=pd.read_csv('final_plots/thalmann_z3_3task_full/decoding/seed_averaged_decoding.csv').set_index('target')\n"
     "cmp=pd.DataFrame({'IDRNN 2t':a.idrnn_loo_r2_mean,'IDRNN 3t':b.idrnn_loo_r2_mean,\n"
     "                  'Vanilla 2t':a.vanilla_loo_r2_mean,'Vanilla 3t':b.vanilla_loo_r2_mean,\n"
     "                  'Van3t frac+':b.vanilla_frac_pos}).round(3)\n"
     "display(cmp)\n"
     "# Control: does the 3rd task help only IDRNN, or also the dim-matched vanilla?\n"
     "k=['WM_WMU','WM_composite','WM_SS','BIG5_open','CEI']; x=np.arange(len(k)); w=.2\n"
     "fig,ax=plt.subplots(figsize=(9,3.4))\n"
     "ax.bar(x-1.5*w,[a.loc[t,'idrnn_loo_r2_mean'] for t in k],w,label='IDRNN 2t',color='#9dcbec')\n"
     "ax.bar(x-0.5*w,[b.loc[t,'idrnn_loo_r2_mean'] for t in k],w,label='IDRNN 3t',color='#0272b2')\n"
     "ax.bar(x+0.5*w,[a.loc[t,'vanilla_loo_r2_mean'] for t in k],w,label='Vanilla 2t',color='#fbc07f')\n"
     "ax.bar(x+1.5*w,[b.loc[t,'vanilla_loo_r2_mean'] for t in k],w,label='Vanilla 3t',color='#ec6f00')\n"
     "ax.axhline(0,color='k',lw=.8); ax.set_xticks(x); ax.set_xticklabels(k,rotation=20,ha='right',fontsize=8)\n"
     "ax.set_ylabel('seed-avg LOO R²'); ax.set_title('3rd task helps IDRNN bottleneck, not dim-matched vanilla'); ax.legend(fontsize=7,ncol=2); plt.show()")

md("## 8b. 3-task fit — four decodability metrics, IDRNN vs dim-matched Vanilla\n"
   "**Seed-averaged (mean±SD over 10 seeds)** — Pearson r (in-sample multiple-R), Spearman r "
   "(in-sample), LOO r (pred vs target; negative = no-info artifact), LOO R² (out-of-sample). "
   "Seed-averaging is essential: a single lucky seed can show spurious signal (e.g. vanilla "
   "openness) that vanishes on averaging.")
code("show('final_plots/thalmann_z3_3task_full/decoding/four_metrics.png', 1100)")
code("fm=pd.read_csv('final_plots/thalmann_z3_3task_full/decoding/seed_averaged_decoding.csv').set_index('target')\n"
     "cols=[f'{a}_{m}_mean' for a in ['idrnn','vanilla'] for m in ['pearson','spearman','loo_r','loo_r2']]\n"
     "display(fm[cols].round(3))")

md("## 8c. Data scaling — does the readout improve with more training tasks?\n"
   "Clean nested S1 ladder: 1 task (restless) ⊂ 2 tasks (+2-armed) ⊂ 3 tasks (+horizon), "
   "both archs fully retrained at each rung. **IDRNN's readout climbs** (driven by the "
   "exploration/horizon task); the **dim-matched Vanilla stays flat near zero** — no "
   "individual-difference mechanism to exploit the extra data.")
code("show('final_plots/thalmann_data_scaling.png', 1100)")
code("ds=pd.read_csv('final_plots/thalmann_data_scaling_summary.csv')\n"
     "for t in ['WM_WMU','WM_composite','WM_SS','BIG5_open','CEI']:\n"
     "    display(Markdown(f'**{t}** — seed-avg LOO R² (rows=arch, cols=#tasks)'))\n"
     "    display(ds[ds.target==t].pivot(index='arch',columns='ntasks',values='loo_r2').round(3))")

md("## 9. Conclusions\n"
   "1. **Only IDRNN carries generalisable individual-difference signal** (LOO R²>0), "
   "concentrated on **working memory + openness**; robust across 10 seeds; Vanilla 0/14.\n"
   "2. **Generative validity**: counterfactual regret (R4) predicts held-out **horizon** "
   "regret (task0 r≈+0.43) **and working memory** (r≈−0.22) — and survives environment "
   "marginalisation, arguing against a fixed-schedule artifact.\n"
   "3. **Both sessions** strengthen the cross-task transfer; decoding readout is similar.\n"
   "4. Affect/mood/depression show no generalisable signal in any readout.\n"
   "5. **Adding the horizon (exploration) task to training markedly strengthens the WM + "
   "openness readout** (e.g. WM-updating LOO R² +0.018→+0.107, 100% of seeds positive) — "
   "and this gain is **IDRNN-specific**: the dim-matched vanilla barely moves (WM-updating "
   "−0.009→+0.002), so it is the individual-difference bottleneck, not just more data, that "
   "isolates the signal.\n"
   "6. **Data scaling**: across the 1→2→3-task ladder the IDRNN WM readout climbs "
   "(mean WM LOO R² +0.015→+0.008→+0.086; WM-updating 0.001→0.016→0.107) while the "
   "dim-matched Vanilla stays flat near 0 (−0.024→−0.019→+0.009) — IDRNN exploits more "
   "(informative) data, Vanilla cannot.")

nb["cells"] = C
nbf.write(nb, "thalmann_results.ipynb")
print("wrote thalmann_results.ipynb with", len(C), "cells")
