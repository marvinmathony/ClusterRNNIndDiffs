#!/usr/bin/env python3
"""Morning report: does moving panels c/d/e/g of the journal figure from S1-only
(236, final_plots/thalmann_z3_3task_full) to pooled S1+S2 (238,
final_plots/thalmann_z3_s2_3task_full = the ds_t012 canonical runs) change the
main results?

Replicates the EXACT computations of thalmann_results.ipynb cell 88 for both
versions and writes a side-by-side markdown report with BF-band verdicts:
  Panel c: loo_r2_panel.csv (canonical LOO R², one-sided BF, perm p) per trait/arch
  Panel d: r(consensus PC1, trait) + two-sided JZS BF, 14 traits
  Panel e: r(consensus PC1, overall switch rate) + BF
  Panel g: raw r and partial r|WM_composite for BIG5_open + CEI (BF on partial, n-1)
  Coherence: new panel-c openness IDRNN R² vs data-scaling t012 point (panel f)

Robust to missing inputs (upstream job failures are reported, not fatal).
Output: final_plots/thalmann_z3_s2_3task_full/CONSISTENCY_REPORT.md (+ stdout).
"""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import pingouin as pg
from pingouin import bayesfactor_pearson

OLD_FULL = "final_plots/thalmann_z3_3task_full";    OLD_DATA = "data_thalmann_3task_full"
NEW_FULL = "final_plots/thalmann_z3_s2_3task_full"; NEW_DATA = "data_sub_t012_full"
OUT = f"{NEW_FULL}/CONSISTENCY_REPORT.md"

ALL_KEYS = ["PANAS_PA", "PANAS_NA", "STICSA", "PHQ", "CEI", "BIG5_open", "AxDep",
            "posMood", "negMood", "Exp", "WM_composite", "WM_OS", "WM_SS", "WM_WMU"]

L = []          # report lines
def w(s=""):
    L.append(s); print(s)


def band(bf):
    """Evidence band for a BF10."""
    if not np.isfinite(bf): return "n/a"
    if bf >= 10: return "strong+"
    if bf >= 3: return "moderate+"
    if bf > 1: return "anecdotal+"
    if bf > 1/3: return "anecdotal0"
    return "null"


def rr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return (float(pearsonr(a[m], b[m])[0]), int(m.sum())) if m.sum() > 2 else (np.nan, 0)


def safe_bf(r, n, **kw):
    try:
        v = float(bayesfactor_pearson(r, n, **kw))
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def switch_rate(bl):
    s = p = 0
    for b in range(bl.shape[0]):
        ch = bl[b]; ch = ch[ch >= 0]
        if len(ch) < 2: continue
        s += int((ch[1:] != ch[:-1]).sum()); p += len(ch) - 1
    return s / p if p else np.nan


def load_version(full, data):
    """Everything panels d/e/g need for one version; None fields when missing."""
    v = {}
    try:
        v["zc"] = np.load(f"{full}/decoding/z2_consensus_pca.npy")
    except Exception as e:
        v["zc"] = None; v["zc_err"] = str(e)
    try:
        v["tg"] = pd.read_csv(f"{full}/decoding/trait_targets.csv")
    except Exception as e:
        v["tg"] = None; v["tg_err"] = str(e)
    try:
        c = np.load(f"{data}/c_train.npy")
        v["sw_all"] = np.array([switch_rate(c[i]) for i in range(c.shape[0])])
    except Exception as e:
        v["sw_all"] = None; v["sw_err"] = str(e)
    try:
        v["panel"] = pd.read_csv(f"{full}/decoding/loo_r2_panel.csv").set_index("target")
    except Exception as e:
        v["panel"] = None; v["panel_err"] = str(e)
    try:
        v["z2sum"] = json.load(open(f"{full}/decoding/z2_seed_matched_summary.json"))
    except Exception:
        v["z2sum"] = None
    return v


def main():
    w("# Pooled-consistency report — panels c/d/e/g: S1-only (236) vs pooled S1+S2 (238)")
    w()
    w(f"- OLD (current figure): `{OLD_FULL}` latents, `{OLD_DATA}` behaviour")
    w(f"- NEW (pooled):         `{NEW_FULL}` latents (= ds_t012 canonical runs, identical HP z=3/hid=10/temb=4), `{NEW_DATA}` behaviour")
    w("- Targets identical in both (pooled S1+S2 questionnaires); cohort differs by 2 S2-only participants (sids 42, 177).")
    w()

    old = load_version(OLD_FULL, OLD_DATA)
    new = load_version(NEW_FULL, NEW_DATA)
    changes = []           # collected verdict lines
    missing = [k for v, tag in ((old, "old"), (new, "new"))
               for k in ("zc", "tg", "sw_all", "panel") if v[k] is None]

    # ── Panel c: LOO R² decoding ─────────────────────────────────────────────
    w("## Panel c — out-of-sample LOO R² (canonical, leakage-free)")
    if old["panel"] is None or new["panel"] is None:
        w(f"MISSING: old={old.get('panel_err','ok')} new={new.get('panel_err','ok')}")
    else:
        po, pn = old["panel"], new["panel"]
        w()
        w("| trait | IDRNN R² old→new | BF_gt old→new (band) | Vanilla R² old→new | vanilla BF_gt old→new |")
        w("|---|---|---|---|---|")
        for k in ALL_KEYS:
            if k not in po.index or k not in pn.index:
                w(f"| {k} | missing | | | |"); continue
            io, iN = po.loc[k], pn.loc[k]
            flag = ""
            if band(io["idrnn_bf"]) != band(iN["idrnn_bf"]):
                flag = f" **[{band(io['idrnn_bf'])}→{band(iN['idrnn_bf'])}]**"
                changes.append(f"Panel c {k} (IDRNN): BF band {band(io['idrnn_bf'])} → {band(iN['idrnn_bf'])} "
                               f"(R² {io['idrnn_r2']:+.3f} → {iN['idrnn_r2']:+.3f})")
            w(f"| {k} | {io['idrnn_r2']:+.3f} → {iN['idrnn_r2']:+.3f} | "
              f"{io['idrnn_bf']:.3g} → {iN['idrnn_bf']:.3g}{flag} | "
              f"{io['vanilla_r2']:+.3f} → {iN['vanilla_r2']:+.3f} | "
              f"{io['vanilla_bf']:.3g} → {iN['vanilla_bf']:.3g} |")
        w()

    # ── Panels d/e/g need zc + tg ────────────────────────────────────────────
    for tag, v in (("OLD", old), ("NEW", new)):
        for kk in ("zc", "tg", "sw_all"):
            if v[kk] is None:
                w(f"MISSING for {tag}: {kk} ({v.get(kk + '_err', v.get('zc_err', ''))})")

    if all(v[k] is not None for v in (old, new) for k in ("zc", "tg", "sw_all")):
        # Panel d: consensus PC1 vs every trait
        w("## Panel d — individual-difference axis: r(PC1, trait) + two-sided JZS BF")
        w()
        w("| trait | r old→new | BF old→new (band) |")
        w("|---|---|---|")
        sig_old, sig_new = set(), set()
        for k in ALL_KEYS:
            ro, no_ = rr(old["zc"], old["tg"][k].values)
            rn, nn_ = rr(new["zc"], new["tg"][k].values)
            bo, bn = safe_bf(ro, no_), safe_bf(rn, nn_)
            if bo >= 3: sig_old.add(k)
            if bn >= 3: sig_new.add(k)
            flag = f" **[{band(bo)}→{band(bn)}]**" if band(bo) != band(bn) else ""
            w(f"| {k} | {ro:+.3f} → {rn:+.3f} | {bo:.3g} → {bn:.3g}{flag} |")
        w()
        # NB the consensus axis has an arbitrary global sign anchored to horizon switch;
        # both versions are anchored the same way, so signs are comparable.
        gained = sorted(sig_new - sig_old); lost = sorted(sig_old - sig_new)
        if gained: changes.append(f"Panel d: traits GAINING BF≥3 with PC1: {gained}")
        if lost:   changes.append(f"Panel d: traits LOSING BF≥3 with PC1: {lost}")
        w(f"Traits with BF≥3 — old: {sorted(sig_old)}")
        w(f"Traits with BF≥3 — new: {sorted(sig_new)}")
        w()

        # Panel e: consensus PC1 vs overall switch rate
        w("## Panel e — consensus PC1 vs overall switch rate")
        ro, no_ = rr(old["zc"], old["sw_all"]); rn, nn_ = rr(new["zc"], new["sw_all"])
        bo, bn = safe_bf(ro, no_), safe_bf(rn, nn_)
        w(f"- old: r = {ro:+.3f} (n={no_}), BF = {bo:.3g} [{band(bo)}]")
        w(f"- new: r = {rn:+.3f} (n={nn_}), BF = {bn:.3g} [{band(bn)}]")
        if band(bo) != band(bn):
            changes.append(f"Panel e: switch-rate BF band {band(bo)} → {band(bn)}")
        w()

        # Panel g: raw vs partial | WM
        w("## Panel g — trait signal beyond working memory (raw r vs partial r | WM_composite)")
        w()
        w("| trait | raw r old→new | partial r|WM old→new | BF(partial) old→new (band) |")
        w("|---|---|---|---|")
        for k, lbl in (("BIG5_open", "Openness"), ("CEI", "Curiosity")):
            row = {}
            for tag, v in (("old", old), ("new", new)):
                dfw = pd.DataFrame({"z": v["zc"], "WM": v["tg"]["WM_composite"].values,
                                    k: v["tg"][k].values}).dropna()
                n = len(dfw)
                raw = float(pg.corr(dfw["z"], dfw[k])["r"].values[0])
                par = float(pg.partial_corr(dfw, x="z", y=k, covar="WM")["r"].values[0])
                row[tag] = (raw, par, safe_bf(par, n - 1), n)
            (ro_, po_, bo, _), (rn_, pn_, bn, _) = row["old"], row["new"]
            flag = f" **[{band(bo)}→{band(bn)}]**" if band(bo) != band(bn) else ""
            if band(bo) != band(bn):
                changes.append(f"Panel g {lbl}: partial-r BF band {band(bo)} → {band(bn)}")
            w(f"| {lbl} | {ro_:+.3f} → {rn_:+.3f} | {po_:+.3f} → {pn_:+.3f} | {bo:.3g} → {bn:.3g}{flag} |")
        w()

    # ── z2 consensus summaries ───────────────────────────────────────────────
    if old["z2sum"] and new["z2sum"]:
        w("## Consensus-axis summary (z2_seed_matched_summary.json)")
        keys = ["exp_pc", "r_wm", "r_open", "r_cei", "r_sw_horizon"]
        w("| key | old | new |"); w("|---|---|---|")
        for k in keys:
            w(f"| {k} | {old['z2sum'].get(k)} | {new['z2sum'].get(k)} |")
        w()

    # ── Coherence: panel c openness vs panel f (data-scaling) t012 ──────────
    w("## Coherence check — panel c openness vs panel f t012 endpoint")
    try:
        ds = pd.read_csv("final_plots/thalmann_z3_datascaling/datascaling_panel.csv")
        t012 = ds[(ds.group == "openness") & (ds.arch == "idrnn") & (ds.subset == "t012")]
        pf = float(t012["canonical_r2"].iloc[0])
        pc = float(new["panel"].loc["BIG5_open", "idrnn_r2"]) if new["panel"] is not None else np.nan
        w(f"- panel f t012 openness (IDRNN): R² = {pf:+.4f} (n={int(t012['n'].iloc[0])})")
        w(f"- NEW panel c BIG5_open (IDRNN): R² = {pc:+.4f}")
        w(f"- These now share models, cohort, targets and readout → expected ≈ equal. Δ = {pc - pf:+.4f}")
        if np.isfinite(pc) and abs(pc - pf) > 0.005:
            changes.append(f"Coherence: panel c vs f t012 differ by {pc - pf:+.4f} (expected ≈ 0) — check masks")
    except Exception as e:
        w(f"MISSING: {e}")
    w()

    # ── Other panels status ──────────────────────────────────────────────────
    w("## Panels already pooled (untouched)")
    w("- Panel a (held-out NLL): `final_plots/thalmann_z3_s2_pooled/per_participant_nll_marginal.csv`")
    w("- Panel b (cross-task regret): `final_plots/thalmann_z3_s2_full/regret/step1_cross_task_regret.npz`")
    w("- Panel f (data-scaling): `final_plots/thalmann_z3_datascaling/datascaling_panel.csv`")
    w()

    # ── Separate workstream: yesterday's s2_cvnll pipeline ──────────────────
    w("## Separate workstream — thalmann_s2 cv-val-NLL pipeline (jobs 38506849/38506850)")
    for lg in sorted(glob.glob("logs/thal_s2_cvnll_*.out")):
        try:
            tail = open(lg).read().strip().split("\n")
            w(f"- `{lg}`: last line: `{tail[-1][:120]}`")
        except Exception:
            pass
    w()

    # ── Verdict ──────────────────────────────────────────────────────────────
    hdr = ["# VERDICT — do the main results change?", ""]
    if missing:
        hdr.append(f"**INCOMPLETE — missing inputs ({', '.join(sorted(set(missing)))}); "
                   "an upstream job failed or is still running. Verdict below covers computed sections only.**")
    if changes:
        hdr.append(f"**{len(changes)} evidence-band change(s) detected:**")
        hdr += [f"- {c}" for c in changes]
    elif not missing:
        hdr.append("**No evidence-band changes — all panel c/d/e/g conclusions hold under pooled S1+S2.**")
    hdr.append("")
    full_report = "\n".join(hdr + L)
    os.makedirs(NEW_FULL, exist_ok=True)
    with open(OUT, "w") as f:
        f.write(full_report)
    print("\n".join(hdr))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
