"""
report.py - Stage 8: interpretive report.

Assembles a text file summarizing the OLS and ML results.  The file is written
alongside the tables and figures and is meant to be scanned quickly rather
than to be extensive and cover the whole interpretation.
"""
from pathlib import Path

from config import MIN_N_TRAIN_ML


def write_interpretive_report(ols_uni, ols_multi, predictions, meta,
                              groups=("EU/Europe", "Global")):
    """Write ``INTERPRETIVE_REPORT.txt`` in the CWD."""
    print(f"\n{'='*78}\nSTAGE 8 - Interpretive report\n{'='*78}")

    lines = []
    lines.append("=" * 78)
    lines.append(f"THESIS ANALYSIS - INTERPRETIVE REPORT")
    lines.append("Auto-generated from analysis outputs.")
    lines.append("=" * 78)
    lines.append("")

    # ── Univariate OLS ──
    lines.append("1.  UNIVARIATE OLS  (log-log; VRE% predictor; plain OLS SE)")
    lines.append("-" * 78)
    for t in ["storage_gw", "battery_gw"]:
        r = ols_uni.get(t)
        if r is None:
            continue
        sig = ("significant at p < 0.05" if r["p"] < 0.05
               else "NOT significant at p < 0.05")
        lines.append(f"  {t}:")
        lines.append(f"    N = {r['n']}, β = {r['beta']:+.3f}, SE = {r['se']:.3f}, "
                     f"p = {r['p']:.3f}, R² = {r['r2']:.3f}")
        lines.append(f"    95% CI: [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  → {sig}")
        lines.append("    Interpretation: a 1% relative rise in VRE share is "
                     "associated with")
        lines.append(f"    a {r['beta']:+.2f}% change in projected "
                     f"{t.replace('_',' ')}.")
        if r["n"] < 12:
            lines.append(f"    Note: with N = {r['n']}, treat the point estimate "
                         "as indicative.")
        lines.append("")

    # ── Multivariate OLS ──
    lines.append("2.  MULTIVARIATE OLS  (battery ~ log(solar) + log(wind); plain SE)")
    lines.append("-" * 78)
    if ols_multi is not None:
        lines.append(f"  N = {ols_multi['n']}, R² = {ols_multi['r2']:.3f}")
        lines.append(f"    β(solar) = {ols_multi['beta_solar']:+.3f}  "
                     f"SE = {ols_multi['se_solar']:.3f}  "
                     f"p = {ols_multi['p_solar']:.3f}")
        lines.append(f"    β(wind)  = {ols_multi['beta_wind']:+.3f}  "
                     f"SE = {ols_multi['se_wind']:.3f}  "
                     f"p = {ols_multi['p_wind']:.3f}")
        lines.append("")
        if "corr_solar_wind" in ols_multi:
            lines.append("  Multicollinearity diagnostic:")
            lines.append(f"    corr(log solar, log wind) = "
                         f"{ols_multi['corr_solar_wind']:+.3f}")
            lines.append(f"    VIF(solar) = {ols_multi['vif_solar']:.1f}   "
                         f"VIF(wind) = {ols_multi['vif_wind']:.1f}")
            lines.append(f"    Univariate fallback slopes on the same "
                         f"{ols_multi['n']} rows:")
            lines.append(f"      β(solar) = {ols_multi['univ_solar_beta']:+.3f}  "
                         f"R² = {ols_multi['univ_solar_r2']:.3f}")
            lines.append(f"      β(wind)  = {ols_multi['univ_wind_beta']:+.3f}  "
                         f"R² = {ols_multi['univ_wind_r2']:.3f}")
            lines.append("")
        if abs(ols_multi.get("corr_solar_wind", 0)) > 0.9:
            lines.append("  With corr(solar, wind) > 0.9, the individual "
                         "multivariate")
            lines.append("  coefficients are NOT identified on this sample.  "
                         "Report as")
            lines.append("  a diagnostic in Table 6 and use the two univariate "
                         "scatters")
            lines.append("  (Figure 5) for the compositional discussion in "
                         "§5.3 / §6.1.")
        if ols_multi["n"] < 12:
            lines.append(f"  With N = {ols_multi['n']}, report as INDICATIVE "
                         "evidence, not confirmatory.")
        lines.append("")
    else:
        lines.append("  Multivariate OLS did not run (N < 5 after aggregation).")
        lines.append("")

    # ── ML predictions ──
    lines.append("3.  ML PREDICTIONS  (missing-target rows, by group)")
    lines.append("-" * 78)
    lines.append("  ElasticNet and Random Forest are compared by leave-one-out CV on the")
    lines.append("  target rows; the winner is refit on the full training set and")
    lines.append("  used to predict on the rows where the target is unobserved.  Groups")
    lines.append(f"  with n_train < {MIN_N_TRAIN_ML} are not fitted - see Vabalas et al.")
    lines.append("  (2019) on ML validation with limited sample size.")
    lines.append("")
    for group in groups:
        lines.append(f"  --- {group} ---")
        any_fit = False
        for target in ["storage_gw", "battery_gw"]:
            if target in predictions[group]:
                any_fit = True
                out = predictions[group][target]
                m = meta[group][target]
                col = f"pred_{target}_gw"
                import numpy as np
                yhat = out[col].values
                lines.append(f"  {target}: {len(out)} rows predicted")
                lines.append(f"    LOO-CV comparison on n_train = {m['n_train']}:")
                for name, sc in m["scores"].items():
                    winner_mark = "  ← winner" if name == m["winner"] else ""
                    lines.append(f"      {name:13s}  "
                                 f"LOO-R² = {sc['R2']:+.3f}   "
                                 f"LOO-MAE = {sc['MAE_GW']:6.1f} GW{winner_mark}")
                lines.append(f"    Model used   : {m['winner']}")
                lines.append(f"    Range        : {yhat.min():.0f} – {yhat.max():.0f} GW")
                lines.append(f"    Median       : {np.median(yhat):.0f} GW")
                lines.append(f"    Reliability  : {m['reliability']}")
                lines.append("")
            else:
                lines.append(f"  {target}: NOT FITTED  (n_train too small "
                             f"or nothing to predict)")
                lines.append("")
        if not any_fit:
            lines.append(f"  → no predictions produced for {group}.")
            lines.append("")

    # ── Key numbers ──
    lines.append("=" * 78)
    lines.append("KEY NUMBERS TO CITE IN CHAPTER 5")
    lines.append("-" * 78)
    if ols_uni.get("storage_gw") is not None:
        r = ols_uni["storage_gw"]
        lines.append(f"  Storage: β = {r['beta']:+.3f}  (SE {r['se']:.3f}, "
                     f"95% CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}], "
                     f"R² = {r['r2']:.3f}, N = {r['n']})")
    if ols_uni.get("battery_gw") is not None:
        r = ols_uni["battery_gw"]
        lines.append(f"  Battery: β = {r['beta']:+.3f}  (SE {r['se']:.3f}, "
                     f"95% CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}], "
                     f"R² = {r['r2']:.3f}, N = {r['n']})")
    if ols_multi is not None:
        lines.append("  Multivariate battery (diagnostic - see §5.3):")
        lines.append(f"    β(solar) = {ols_multi['beta_solar']:+.3f}  "
                     f"(SE {ols_multi['se_solar']:.3f}, "
                     f"CI [{ols_multi['ci_solar'][0]:+.3f}, "
                     f"{ols_multi['ci_solar'][1]:+.3f}], "
                     f"p = {ols_multi['p_solar']:.3f})")
        lines.append(f"    β(wind)  = {ols_multi['beta_wind']:+.3f}  "
                     f"(SE {ols_multi['se_wind']:.3f}, "
                     f"CI [{ols_multi['ci_wind'][0]:+.3f}, "
                     f"{ols_multi['ci_wind'][1]:+.3f}], "
                     f"p = {ols_multi['p_wind']:.3f})")
        lines.append(f"    R² = {ols_multi['r2']:.3f}, N = {ols_multi['n']}")
    lines.append("=" * 78)

    out_path = Path(f"INTERPRETIVE_REPORT.txt")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved → {out_path.name}")