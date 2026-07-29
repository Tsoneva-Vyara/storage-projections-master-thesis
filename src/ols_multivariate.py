"""
ols_multivariate.py - Stage 6: Table 6 + Figure 5.

Multivariate log-log OLS of battery capacity on solar and wind capacity:
    log(Battery) = alpha + beta_s * log(Solar) + beta_w * log(Wind) + eps

WHY FIGURE 5 IS TWO UNIVARIATE SCATTERS
On the six aggregated rows, corr(log solar, log wind) is essentially +1
and the VIF for each predictor is >100.  The multivariate coefficients
(beta_solar, beta_wind) are therefore unidentified, and the sign flip on
wind (positive in the univariate, negative in the multivariate) is a
multicollinearity issue rather than a compositional finding.  Two
univariate scatters visualize what the six rows
indicate; each panel plots the bivariate relationship between
battery capacity and the predictor of interest without using
small-N multivariate OLS for something it cannot support.  The
multivariate coefficient table (Table 6) is retained as a diagnostic;
sections 4.5 and 5.3 of the thesis explain the multicollinearity issue.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from plotting import dataframe_to_png


# ── Fit ─────────────────────────────────────────────────────────────────────
def run_multivariate_ols(df, target="battery_gw",
                         group_filter=("EU/Europe",),
                         aggregate_brackets=True):
    """Multivariate log-log OLS of battery on solar + wind capacity."""
    predictors = ["solar_gw", "wind_gw"]
    data = df.dropna(subset=[target] + predictors).copy()
    data = data[data["analysis_group"].isin(group_filter)]

    if aggregate_brackets and len(data) > 0:
        key_cols = ["publisher_author", "publishing_year", "year",
                    "solar_gw", "wind_gw"]
        agg_dict = {target: "median",
                    "analysis_group": "first", "region": "first",
                    "study": "first"}
        n_before = len(data)
        data = data.groupby(key_cols, as_index=False).agg(agg_dict)
        n_collapsed = n_before - len(data)
        if n_collapsed > 0:
            print(f"    [aggregation: collapsed {n_collapsed} bracket rows]")

    n = len(data)
    if n < 5:
        print(f"  {target}: skipped (n={n}<5)")
        return None

    data["_y"]  = np.log(np.clip(data[target].values,   0.01, None))
    data["_x1"] = np.log(np.clip(data["solar_gw"].values, 0.01, None))
    data["_x2"] = np.log(np.clip(data["wind_gw"].values,  0.01, None))

    model = smf.ols("_y ~ _x1 + _x2", data=data).fit()

    r = {
        "target": target, "model": model, "data": data, "n": n,
        "beta_solar":  model.params["_x1"],
        "beta_wind":   model.params["_x2"],
        "intercept":   model.params["Intercept"],
        "se_solar":    model.bse["_x1"],
        "se_wind":     model.bse["_x2"],
        "se_intc":     model.bse["Intercept"],
        "p_solar":     model.pvalues["_x1"],
        "p_wind":      model.pvalues["_x2"],
        "p_intc":      model.pvalues["Intercept"],
        "ci_solar":    model.conf_int().loc["_x1"].values,
        "ci_wind":     model.conf_int().loc["_x2"].values,
        "ci_intc":     model.conf_int().loc["Intercept"].values,
        "r2":          model.rsquared,
    }
    print(f"  n={n}  R²={r['r2']:.3f}")
    print(f"    β(solar) = {r['beta_solar']:+.3f}  SE = {r['se_solar']:.3f}  "
          f"p = {r['p_solar']:.3f}  95% CI = [{r['ci_solar'][0]:+.3f}, {r['ci_solar'][1]:+.3f}]")
    print(f"    β(wind)  = {r['beta_wind']:+.3f}  SE = {r['se_wind']:.3f}  "
          f"p = {r['p_wind']:.3f}  95% CI = [{r['ci_wind'][0]:+.3f}, {r['ci_wind'][1]:+.3f}]")
    return r


# ── Table 6 ─────────────────────────────────────────────────────────────────
def _write_table6(res):
    rows = [
        {"Predictor": "log(Solar, GW)",
         "β":       f"{res['beta_solar']:+.3f}",
         "SE":      f"{res['se_solar']:.3f}",
         "95% CI":  f"[{res['ci_solar'][0]:+.3f}, {res['ci_solar'][1]:+.3f}]",
         "p-value": f"{res['p_solar']:.3f}"},
        {"Predictor": "log(Wind, GW)",
         "β":       f"{res['beta_wind']:+.3f}",
         "SE":      f"{res['se_wind']:.3f}",
         "95% CI":  f"[{res['ci_wind'][0]:+.3f}, {res['ci_wind'][1]:+.3f}]",
         "p-value": f"{res['p_wind']:.3f}"},
        {"Predictor": "Intercept",
         "β":       f"{res['intercept']:+.3f}",
         "SE":      f"{res['se_intc']:.3f}",
         "95% CI":  f"[{res['ci_intc'][0]:+.3f}, {res['ci_intc'][1]:+.3f}]",
         "p-value": f"{res['p_intc']:.3f}"},
    ]
    tbl = pd.DataFrame(rows)
    csv_path = f"table6_multivariate_ols.csv"
    png_path = f"table6_multivariate_ols.png"
    tbl.to_csv(csv_path, index=False)
    dataframe_to_png(
        tbl, png_path,
        title="Table 6: Multivariate log-log OLS of battery capacity on solar and wind capacity (diagnostic)",
        footnote=(f"R² = {res['r2']:.3f}   |   N = {res['n']}   |   "
                  "Standard errors are conventional OLS.  Multicollinearity "
                  "diagnostic reported below; see §5.3 of the thesis and Figure 5."),
    )
    print(f"  Saved → {csv_path} / .png / .pdf")


# ── Figure 5 (two univariate scatters - see module docstring for rationale) ─
def _plot_figure5(res):
    data_mv = res["data"].copy()
    m_solar = smf.ols("_y ~ _x1", data=data_mv).fit()
    m_wind  = smf.ols("_y ~ _x2", data=data_mv).fit()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    panels = [
        ("Solar", m_solar, "solar_gw", "_x1", "Solar capacity (GW, log scale)"),
        ("Wind",  m_wind,  "wind_gw",  "_x2", "Wind capacity (GW, log scale)"),
    ]
    for ax, (predictor, model_uni, xkey, xkey_log, xlabel) in zip(axes, panels):
        ax.scatter(data_mv[xkey], data_mv["battery_gw"],
                   s=100, color="#1f77b4", edgecolors="white",
                   linewidths=1.4, alpha=0.85, zorder=3,
                   label=f"Observations (n = {res['n']})")

        x_min = data_mv[xkey].min() * 0.7
        x_max = data_mv[xkey].max() * 1.4
        xp = np.linspace(x_min, x_max, 200)
        pred = model_uni.get_prediction(
            pd.DataFrame({xkey_log: np.log(np.clip(xp, 0.01, None))})
        ).summary_frame(alpha=0.05)

        ax.plot(xp, np.exp(pred["mean"].values), "-",
                color="#2ca02c", lw=2.2,
                label=f"Univariate OLS fit\nβ = {model_uni.params[xkey_log]:+.3f}",
                zorder=4)
        ax.fill_between(xp,
                        np.exp(pred["mean_ci_lower"].values),
                        np.exp(pred["mean_ci_upper"].values),
                        color="#2ca02c", alpha=0.18, label="95% CI", zorder=2)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Battery capacity (GW, log scale)", fontsize=11)
        panel = f"({'a' if predictor == 'Solar' else 'b'}) Battery vs {predictor}"
        ax.set_title(f"{panel}\nR² = {model_uni.rsquared:.3f}", fontsize=11)
        ax.legend(fontsize=9, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, ls=":")

    plt.tight_layout()
    plt.savefig(f"fig5_multivariate_scatter.png",
                bbox_inches="tight", dpi=200)
    plt.savefig(f"fig5_multivariate_scatter.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → fig5_multivariate_scatter.png / .pdf")

    return m_solar, m_wind


# ── Multicollinearity diagnostic ────────────────────────────────────────────
def _diagnose_multicollinearity(res, m_solar, m_wind):
    data_mv = res["data"]
    corr_sw = data_mv[["_x1", "_x2"]].corr().iloc[0, 1]
    Xvif = sm.add_constant(data_mv[["_x1", "_x2"]])
    vif_s = variance_inflation_factor(Xvif.values, 1)
    vif_w = variance_inflation_factor(Xvif.values, 2)

    res["corr_solar_wind"] = corr_sw
    res["vif_solar"]       = vif_s
    res["vif_wind"]        = vif_w
    res["univ_solar_beta"] = m_solar.params["_x1"]
    res["univ_solar_r2"]   = m_solar.rsquared
    res["univ_wind_beta"]  = m_wind.params["_x2"]
    res["univ_wind_r2"]    = m_wind.rsquared

    print("  Multicollinearity diagnostic:")
    print(f"    corr(log solar, log wind) = {corr_sw:+.3f}   "
          f"VIF(solar) = {vif_s:.1f}   VIF(wind) = {vif_w:.1f}")
    print(f"  Univariate fallbacks (log-log OLS on the same {res['n']} rows):")
    print(f"    β(solar) = {m_solar.params['_x1']:+.3f}  R² = {m_solar.rsquared:.3f}")
    print(f"    β(wind)  = {m_wind.params['_x2']:+.3f}  R² = {m_wind.rsquared:.3f}")


# ── Stage entry point ───────────────────────────────────────────────────────
def run_multivariate_stage(df):
    """Run the multivariate OLS stage end-to-end."""
    print(f"\n{'='*78}\nSTAGE 6 - Multivariate OLS  battery ~ log(solar) + log(wind)  → Figure 5\n{'='*78}")

    res = run_multivariate_ols(df, target="battery_gw")
    if res is None:
        return None

    _write_table6(res)
    m_solar, m_wind = _plot_figure5(res)
    _diagnose_multicollinearity(res, m_solar, m_wind)
    return res