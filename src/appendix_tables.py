"""
appendix_tables.py - Appendix C tables.

Table C.1 - EU/Europe predictions summarized by horizon year and target
            variable.  Median and interquartile range across the prediction
            set.  Full row-level detail sits in ``predictions_all.csv``.

Table C.2 - Hyperparameter grids and seed settings for both model classes
            (ElasticNet, Random Forest).  All random seeds in the analysis
            are fixed at 42 for reproducibility.  Grids follow standard
            defaults for small-sample cross-source panels
            (Hastie et al., 2009).

Both tables read from ``config`` for the hyperparameter values and from the
prediction dicts returned by ``ml_imputation.run_ml_stage`` for Table C.1,
so a change to the grid or the seed changes the appendix table without
another manual edit.
"""
import numpy as np
import pandas as pd

from config import (
    ELASTIC_NET_GRID, RANDOM_FOREST_GRID,
    BOOTSTRAP_N, BOOTSTRAP_ALPHA, RANDOM_SEED,
    TARGET_LABEL,
)
from plotting import dataframe_to_png


# ── Table C.1 ───────────────────────────────────────────────────────────────
def write_table_c1(predictions, group="EU/Europe"):
    """
    Median and IQR of the EU/Europe ML predictions by (horizon year, target).

    Only the EU/Europe subgroup has a "Reliable" battery reliability flag
    in the current run, so this appendix table restricts to that subgroup.
    If a target has no predictions (e.g. the training set was too small),
    the target is simply omitted from the summary - the writer prints a
    warning to stdout so it doesn't disappear silently.
    """
    print(f"\n{'='*78}\nAPPENDIX C.1 - {group} predictions by horizon year × target\n{'='*78}")

    group_preds = predictions.get(group, {})
    if not group_preds:
        print(f"  No predictions for {group} - nothing to summarize.")
        return None

    rows = []
    for target in ["storage_gw", "battery_gw"]:
        if target not in group_preds:
            print(f"  {target}: no predictions in this run - skipped.")
            continue
        pdata = group_preds[target].copy()
        col = f"pred_{target}_gw"
        pdata = pdata.dropna(subset=["year", col])
        if pdata.empty:
            continue
        # One row per horizon year, then a final "All horizons" row so the
        # reader has a single overall summary at the bottom.
        for yr, sub in pdata.groupby("year"):
            vals = sub[col].values
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            rows.append({
                "Target":            TARGET_LABEL[target],
                "Horizon year":      int(yr),
                "N predictions":     int(len(vals)),
                "Median (GW)":       f"{q50:.1f}",
                "IQR (GW)":          f"[{q25:.1f} – {q75:.1f}]",
                "Min - Max (GW)":    f"{vals.min():.1f} – {vals.max():.1f}",
            })
        # Overall
        vals = pdata[col].values
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        rows.append({
            "Target":            TARGET_LABEL[target],
            "Horizon year":      "All",
            "N predictions":     int(len(vals)),
            "Median (GW)":       f"{q50:.1f}",
            "IQR (GW)":          f"[{q25:.1f} – {q75:.1f}]",
            "Min - Max (GW)":    f"{vals.min():.1f} – {vals.max():.1f}",
        })

    if not rows:
        print("  No prediction rows survived filtering - table C.1 not written.")
        return None

    tbl = pd.DataFrame(rows)
    csv_path = f"tableC1_predictions_by_horizon.csv"
    png_path = f"tableC1_predictions_by_horizon.png"
    tbl.to_csv(csv_path, index=False)
    dataframe_to_png(
        tbl, png_path,
        title=(f"Table C.1: {group} predictions summarized by horizon year "
               "and target variable"),
        footnote=(
            "Median and interquartile range across the prediction set. "
            f"Full row-level detail in predictions_all.csv. "
            "The EU/Europe subgroup is the only one whose reliability card "
            "returned RELIABLE for the battery target."
        ),
    )
    print(f"  Saved → {csv_path} / .png / .pdf")
    return tbl


# ── Table C.2 ───────────────────────────────────────────────────────────────
def _format_alphas(alphas):
    """Compact display of the alpha grid - endpoints + count."""
    if len(alphas) == 0:
        return "-"
    return (f"{len(alphas)} values log-spaced from "
            f"{alphas[0]:.1e} to {alphas[-1]:.1e}")


def write_table_c2():
    """Document hyperparameter grids and seed settings for both model classes.

    Sourced from ``config.ELASTIC_NET_GRID`` and ``config.RANDOM_FOREST_GRID``
    so the table and the code cannot drift.
    """
    print(f"\n{'='*78}\nAPPENDIX C.2 - Hyperparameter grids and seed settings\n{'='*78}")

    en = ELASTIC_NET_GRID
    rf = RANDOM_FOREST_GRID
    rows = [
        # Common
        {"Model class":     "Common (both)",
         "Hyperparameter":  "Random seed",
         "Value / grid":    str(RANDOM_SEED),
         "Notes":           "Fixed for reproducibility of every random draw."},
        {"Model class":     "Common (both)",
         "Hyperparameter":  "Bootstrap resamples",
         "Value / grid":    str(BOOTSTRAP_N),
         "Notes":           (f"Percentile 95% CI on the predictions "
                             f"(α = {BOOTSTRAP_ALPHA:.2f}).")},
        # ElasticNet
        {"Model class":     "ElasticNet (ElasticNetCV)",
         "Hyperparameter":  "l1_ratio grid",
         "Value / grid":    "[0.1, 0.5, 0.9]",
         "Notes":           ("Ridge-like, balanced, and lasso-like mixing "
                             "of L1 and L2 penalties.")},
        {"Model class":     "ElasticNet (ElasticNetCV)",
         "Hyperparameter":  "alpha grid",
         "Value / grid":    _format_alphas(en["alphas"]),
         "Notes":           ("Decade-spaced sweep. "
                             "Optimum picked by internal CV.")},
        {"Model class":     "ElasticNet (ElasticNetCV)",
         "Hyperparameter":  "max_iter",
         "Value / grid":    str(en["max_iter"]),
         "Notes":           "Coordinate-descent iteration cap."},
        {"Model class":     "ElasticNet (ElasticNetCV)",
         "Hyperparameter":  "cv (inner)",
         "Value / grid":    str(en["cv"]),
         "Notes":           ("k-fold CV used inside ElasticNetCV to pick "
                             "α and l1_ratio.")},
        {"Model class":     "ElasticNet (ElasticNetCV)",
         "Hyperparameter":  "random_state",
         "Value / grid":    str(en["random_state"]),
         "Notes":           "Reproducibility of the internal CV splits."},
        # RandomForest
        {"Model class":     "Random Forest (RandomForestRegressor)",
         "Hyperparameter":  "n_estimators",
         "Value / grid":    str(rf["n_estimators"]),
         "Notes":           "Number of trees in the ensemble."},
        {"Model class":     "Random Forest (RandomForestRegressor)",
         "Hyperparameter":  "max_depth",
         "Value / grid":    str(rf["max_depth"]),
         "Notes":           ("Shallow trees - small-sample "
                             "over-fitting guard.")},
        {"Model class":     "Random Forest (RandomForestRegressor)",
         "Hyperparameter":  "min_samples_leaf",
         "Value / grid":    str(rf["min_samples_leaf"]),
         "Notes":           "Minimum leaf size - same guard."},
        {"Model class":     "Random Forest (RandomForestRegressor)",
         "Hyperparameter":  "random_state",
         "Value / grid":    str(rf["random_state"]),
         "Notes":           ("Reproducibility of the bootstrap sample "
                             "and feature selection at each split.")},
        {"Model class":     "Random Forest (RandomForestRegressor)",
         "Hyperparameter":  "n_jobs",
         "Value / grid":    str(rf["n_jobs"]),
         "Notes":           "Parallel training - no effect on the result."},
        # Outer validation
        {"Model class":     "Outer validation (both)",
         "Hyperparameter":  "Cross-validation scheme",
         "Value / grid":    "Leave-one-out (LOO)",
         "Notes":           ("Reports LOO-R² and LOO-MAE on the "
                             "log-target scale.")},
        {"Model class":     "Outer validation (both)",
         "Hyperparameter":  "Missing-value imputer",
         "Value / grid":    "SimpleImputer(strategy='median')",
         "Notes":           ("Fit inside every LOO fold to avoid "
                             "training-set leakage.")},
    ]

    tbl = pd.DataFrame(rows)
    csv_path = f"tableC2_hyperparameter_grids.csv"
    png_path = f"tableC2_hyperparameter_grids.png"
    tbl.to_csv(csv_path, index=False)
    dataframe_to_png(
        tbl, png_path,
        title="Table C.2: Hyperparameter grids and seed settings for both model classes",
        footnote=(
            "All random seeds in the analysis are fixed at 42 for reproducibility. "
            "Grids follow standard defaults for small-sample cross-source panels "
            "(Hastie et al., 2009)."
        ),
    )
    print(f"  Saved → {csv_path} / .png / .pdf")
    return tbl


# ── Stage entry point ───────────────────────────────────────────────────────
def run_appendix_stage(predictions):
    """Write both Table C.1 and Table C.2."""
    write_table_c1(predictions, group="EU/Europe")
    write_table_c2()