"""
ols_univariate.py - Stage 5: Table 5 + Figure 4.

Univariate log-log OLS of total storage and battery capacity on VRE share:
    log(Y) = alpha + beta * log(VRE%) + eps

Conventional OLS standard errors.  Clustered SE are NOT appropriate at
these sample sizes - Cameron and Miller (2015, "A Practitioner's Guide to
Cluster-Robust Inference", J. Human Resources, 50(2), 317-372) document
the small-cluster failure of cluster-robust SE.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from config import (
    MIN_N_GLOBAL_OLS,
    GROUP_COLOR, SRC_MARKER
)
from plotting import dataframe_to_png


# ── Fit ─────────────────────────────────────────────────────────────────────
def run_univariate_ols(df, target, group_filter=("EU/Europe",),
                       aggregate_brackets=True, verbose=True):
    """
    Univariate log-log OLS of ``target`` on VRE% (filled).

    Bracket aggregation: rows sharing (publisher, publication year, horizon
    year, VRE%) collapse to their median, so a single study with several
    parametric scenario brackets does not dominate the fit.  Returns a dict
    with the model, the aggregated data, and the point estimates the
    thesis reports.  Returns ``None`` if fewer than four rows are available.
    """
    data = df.dropna(subset=[target, "vre_pct_filled"]).copy()
    data = data[data["analysis_group"].isin(group_filter)]

    if aggregate_brackets and len(data) > 0:
        key_cols = ["publisher_author", "publishing_year", "year",
                    "vre_pct_filled"]
        agg_dict = {target: "median", "vre_pct_source": "first",
                    "analysis_group": "first", "region": "first",
                    "study": "first"}
        n_before = len(data)
        data = data.groupby(key_cols, as_index=False).agg(agg_dict)
        n_collapsed = n_before - len(data)
        if verbose and n_collapsed > 0:
            print(f"    [aggregation: collapsed {n_collapsed} bracket rows to medians]")

    n = len(data)
    if n < 4:
        if verbose:
            print(f"  {target}: skipped (n={n})")
        return None

    data["_y"] = np.log(np.clip(data[target].values,     0.01, None))
    data["_x"] = np.log(np.clip(data["vre_pct_filled"].values, 0.01, None))

    model = smf.ols("_y ~ _x", data=data).fit()

    beta = model.params["_x"]
    se   = model.bse["_x"]
    pval = model.pvalues["_x"]
    r2   = model.rsquared
    intc = model.params["Intercept"]
    ci_lo, ci_hi = model.conf_int().loc["_x"].values

    if verbose:
        grp_str = "+".join(group_filter)
        print(f"  {target:11s} [{grp_str}]  n={n}  β={beta:+.3f}  SE={se:.3f}  "
              f"95% CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  R²={r2:.3f}  p={pval:.3f}")

    return {
        "target": target, "model": model, "data": data,
        "n": n, "beta": beta, "se": se, "p": pval, "r2": r2,
        "intercept": intc, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "group_filter": group_filter,
    }


# ── Table 5 ─────────────────────────────────────────────────────────────────
def _write_table5(ols_uni):
    labels = {"storage_gw": "log(Total storage, GW)",
              "battery_gw": "log(Battery, GW)"}
    rows = []
    for t in ["storage_gw", "battery_gw"]:
        r = ols_uni.get(t)
        if r is None:
            continue
        rows.append({
            "Dependent variable": labels[t],
            "β (VRE%)":           f"{r['beta']:+.3f}",
            "SE":                 f"{r['se']:.3f}",
            "95% CI":             f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]",
            "R²":                 f"{r['r2']:.3f}",
            "p-value":            f"{r['p']:.3f}",
            "N":                  r["n"],
        })
    tbl = pd.DataFrame(rows)
    csv_path = f"table5_univariate_ols.csv"
    png_path = f"table5_univariate_ols.png"
    tbl.to_csv(csv_path, index=False)
    dataframe_to_png(
        tbl, png_path,
        title="Table 5: Univariate log-log OLS estimates of storage and battery capacity on VRE share",
        footnote="Standard errors are conventional OLS. Both regressions are estimated on study-level medians after bracket aggregation.",
    )
    print(f"  Saved → {csv_path} / .png / .pdf")


# ── Figure 4 (main EU/Europe scatter) ───────────────────────────────────────
def _plot_figure4(ols_uni):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, target in zip(axes, ["storage_gw", "battery_gw"]):
        r = ols_uni.get(target)
        if r is None:
            continue
        data = r["data"]

        for src, marker in SRC_MARKER.items():
            mask = (data["vre_pct_source"] == src)
            if mask.sum() == 0:
                continue
            src_label = "observed" if src == "observed" else "imputed"
            ax.scatter(data.loc[mask, "vre_pct_filled"],
                       data.loc[mask, target],
                       color=GROUP_COLOR["EU/Europe"], marker=marker,
                       s=85, alpha=0.85, edgecolors="white", linewidths=1.4,
                       label=f"EU/Europe ({src_label}) n={mask.sum()}")

        x_min = max(0.5, data["vre_pct_filled"].min() * 0.85)
        x_max = min(100, data["vre_pct_filled"].max() * 1.15)
        xp = np.linspace(x_min, x_max, 200)
        xp_log = np.log(np.clip(xp, 0.01, None))

        pred = r["model"].get_prediction(pd.DataFrame({"_x": xp_log})) \
                          .summary_frame(alpha=0.05)
        ax.plot(xp, np.exp(pred["mean"].values), "-",
                color="#2ca02c", lw=2.2, label="OLS fit", zorder=3)
        ax.fill_between(xp,
                        np.exp(pred["mean_ci_lower"].values),
                        np.exp(pred["mean_ci_upper"].values),
                        color="#2ca02c", alpha=0.18, label="95% CI",
                        zorder=2)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("VRE share of electricity generation (%, log scale)",
                      fontsize=11)
        ylab = {"storage_gw": "Total storage capacity (GW, log scale)",
                "battery_gw": "Battery capacity (GW, log scale)"}
        ax.set_ylabel(ylab[target], fontsize=11)
        panel = "(a) Total storage" if target == "storage_gw" else "(b) Battery"
        ax.set_title(f"{panel}\nN = {r['n']},  R² = {r['r2']:.3f}", fontsize=11)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, ls=":")

    plt.tight_layout()
    plt.savefig(f"fig4_ols_scatter.png",
                bbox_inches="tight", dpi=200)
    plt.savefig(f"fig4_ols_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → fig4_ols_scatter.png / .pdf  (EU/Europe only)")


# ── Optional Figure 4b (Global-only) ────────────────────────────────────────
def _plot_figure4b(ols_uni_global):
    targets = [t for t in ["storage_gw", "battery_gw"]
               if ols_uni_global.get(t) is not None]
    if not targets:
        print("  Note: Global sample too small on both targets to produce a "
              "separate Global-only Figure 4b.  Global observations excluded "
              "from the main analysis.")
        return

    fig, axes = plt.subplots(1, len(targets),
                             figsize=(6.5 * len(targets), 5.5), squeeze=False)
    axes = axes[0]
    for ax, target in zip(axes, targets):
        r = ols_uni_global[target]
        data = r["data"]

        for src, marker in SRC_MARKER.items():
            mask = (data["vre_pct_source"] == src)
            if mask.sum() == 0:
                continue
            src_label = "observed" if src == "observed" else "imputed"
            ax.scatter(data.loc[mask, "vre_pct_filled"],
                       data.loc[mask, target],
                       color=GROUP_COLOR["Global"], marker=marker,
                       s=85, alpha=0.85, edgecolors="white", linewidths=1.4,
                       label=f"Global ({src_label}) n={mask.sum()}")

        x_min = max(0.5, data["vre_pct_filled"].min() * 0.85)
        x_max = min(100, data["vre_pct_filled"].max() * 1.15)
        xp = np.linspace(x_min, x_max, 200)
        xp_log = np.log(np.clip(xp, 0.01, None))
        pred = r["model"].get_prediction(pd.DataFrame({"_x": xp_log})) \
                          .summary_frame(alpha=0.05)
        ax.plot(xp, np.exp(pred["mean"].values), "-",
                color="#d62728", lw=2.2, label="OLS fit", zorder=3)
        ax.fill_between(xp,
                        np.exp(pred["mean_ci_lower"].values),
                        np.exp(pred["mean_ci_upper"].values),
                        color="#d62728", alpha=0.18, label="95% CI", zorder=2)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("VRE share of electricity generation (%, log scale)",
                      fontsize=11)
        ylab = ("Total storage capacity (GW, log scale)"
                if target == "storage_gw"
                else "Battery capacity (GW, log scale)")
        ax.set_ylabel(ylab, fontsize=11)
        panel = "Total storage" if target == "storage_gw" else "Battery"
        ax.set_title(f"{panel} (Global only)\nN = {r['n']},  R² = {r['r2']:.3f}",
                     fontsize=11)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, ls=":")

    plt.tight_layout()
    plt.savefig(f"fig4b_ols_scatter_global.png",
                bbox_inches="tight", dpi=200)
    plt.savefig(f"fig4b_ols_scatter_global.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → fig4b_ols_scatter_global.png / .pdf  "
          f"(Global-only fit for {targets})")


# ── Stage entry point ───────────────────────────────────────────────────────
def run_univariate_stage(df):
    """
    Run the univariate OLS stage end-to-end.

    Returns a tw-element tuple ``(ols_uni, ols_uni_global)`` for later stages
    (interpretive report).  ``ols_uni_global`` may be empty if the Global
    sample falls below the reporting threshold.
    """
    print(f"\n{'='*78}\nSTAGE 5 - Univariate OLS  (log-log, plain SE)  → Figure 4\n{'='*78}")

    print("  Main analysis (EU/Europe only):")
    ols_uni = {
        "storage_gw": run_univariate_ols(df, "storage_gw"),
        "battery_gw": run_univariate_ols(df, "battery_gw"),
    }

    print("\n  Optional Global-only analysis:")
    ols_uni_global = {}
    for tgt in ["storage_gw", "battery_gw"]:
        peek = df.dropna(subset=[tgt, "vre_pct_filled"])
        peek = peek[peek["analysis_group"] == "Global"]
        if len(peek) == 0:
            print(f"    {tgt}: no Global rows with reported target - skipped")
            continue
        n_agg = peek.groupby(["publisher_author", "publishing_year",
                              "year", "vre_pct_filled"]).ngroups
        if n_agg < MIN_N_GLOBAL_OLS:
            print(f"    {tgt}: only {n_agg} aggregated Global observations "
                  f"(< {MIN_N_GLOBAL_OLS}) - Global sample too small to report "
                  "a separate fit")
            ols_uni_global[tgt] = None
            continue
        ols_uni_global[tgt] = run_univariate_ols(df, tgt,
                                                 group_filter=("Global",))

    _write_table5(ols_uni)
    _plot_figure4(ols_uni)
    _plot_figure4b(ols_uni_global)

    return ols_uni, ols_uni_global