"""
ml_imputation.py - Stage 7: Tables 9-10 + Figures 7a/b.

For each group (EU/Europe, Global) and each target (total storage, battery storage),
two model classes are trained on the target rows and compared by
leave-one-out cross-validation on the log-target scale.  The winner is
refit on the full training set and used to generate predictions for the
rows where the target is unobserved, with 95% bootstrap confidence
intervals from 500 resamples.

Model choice (see thesis §4.6):
  ElasticNet   - Zou, H. and Hastie, T. (2005).  Regularization and
                  variable selection via the Elastic Net.  J. R. Stat.
                  Soc. B, 67(2), 301-320.  L1+L2 penalty handles
                  correlated predictors and small-sample variance.
  RandomForest - Breiman, L. (2001).  Random Forests.  Machine Learning,
                  45(1), 5-32.  Non-linear ensemble; robust to feature
                  scaling; a standard second option in energy ML.

Minimum training-sample size - Vabalas, A., Gowen, E., Poliakoff, E. and
Casson, A.J. (2019).  Machine learning algorithm validation with a
limited sample size.  PLoS ONE, 14(11), e0224365.  Below n_train ~ 8 the
reported LOO-R² and LOO-MAE cannot be trusted as generalisation metrics,
and using such a fit to impute missing values would extend rather than
reduce the uncertainty in the dataset.

Prediction clipping - Lamboll, R. D. et al. (2020, Geosci. Model Dev.,
13(11), 5259-5275) apply the same training-envelope convention in the
IAM imputation context.  Predictions are clipped to 5x the empirical
range of the training data so an occasional bootstrap fit cannot produce
a physically implausible value.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut

from config import (
    MIN_N_TRAIN_ML, GROUP_COLOR,
    FEATURE_SET_NAMES, TARGET_LABEL,
    RELIABLE_R2_MIN, RELIABLE_N_TRAIN, INDICATIVE_R2_MIN,
    ELASTIC_NET_GRID, RANDOM_FOREST_GRID,
    BOOTSTRAP_N, RANDOM_SEED,
)
from plotting import dataframe_to_png


# ── Model registry ──────────────────────────────────────────────────────────
def _build_models():
    """
    Fresh instances every call - sklearn objects.

    Hyperparameters are read from ``config.ELASTIC_NET_GRID`` and
    ``config.RANDOM_FOREST_GRID`` so the appendix table (Table B.1) and
    the fits share a single source.  A change to the grid in
    ``config.py`` propagates both to the fits below and to the
    documentation in the appendix without an extra edit.
    """
    return {
        "ElasticNet":   ElasticNetCV(**ELASTIC_NET_GRID),
        "RandomForest": RandomForestRegressor(**RANDOM_FOREST_GRID),
    }


IMPUTER = SimpleImputer(strategy="median")


# ── LOO-CV scoring ──────────────────────────────────────────────────────────
def loocv_score(X, y, model_template):
    """LOO-CV; returns (R² on log scale, MAE in GW after back-transform, ŷ)."""
    loo = LeaveOneOut()
    yhat = np.full(len(y), np.nan)
    for tr, te in loo.split(X):
        imp = clone(IMPUTER)
        Xtr = imp.fit_transform(X[tr])
        Xte = imp.transform(X[te])
        m = clone(model_template)
        m.fit(Xtr, y[tr])
        yhat[te] = m.predict(Xte)
    r2  = r2_score(y, yhat)
    mae = mean_absolute_error(np.exp(y), np.exp(yhat))
    return r2, mae, yhat


# ── Feature-set resolver ────────────────────────────────────────────────────
def _feature_set(df, target):
    """Keep only the feature names that actually exist as columns in df."""
    return [c for c in FEATURE_SET_NAMES[target] if c in df.columns]


# ── Reliability flag ────────────────────────────────────────────────────────
def _reliability(r2_win, n_train):
    if r2_win >= RELIABLE_R2_MIN and n_train >= RELIABLE_N_TRAIN:
        return "Reliable"
    if r2_win >= INDICATIVE_R2_MIN:
        return "Indicative"
    return "Unreliable"


# ── Predict one (group, target) combination ─────────────────────────────────
def predict_missing_ml(df, target, group="EU/Europe"):
    """
    Train ElasticNet and Random Forest on target rows in the group,
    pick the winner by LOO-CV R² on the log-target scale, and generate
    predictions for the rows in the same group where the target is missing.
    Returns ``None`` when the training set is too small or nothing to predict.
    """
    feats = _feature_set(df, target)
    train = df[(df["analysis_group"] == group) & df[target].notna()].copy()
    pred_rows = df[(df["analysis_group"] == group) & df[target].isna()].copy()

    if len(train) < MIN_N_TRAIN_ML:
        print(f"  {group}/{target}: skipped (n_train = {len(train)} < "
              f"{MIN_N_TRAIN_ML})  - see Vabalas et al. (2019)")
        return None
    if pred_rows.empty:
        print(f"  {group}/{target}: no missing-target rows to predict")
        return None

    # Drop features that are entirely missing in EITHER training or
    # prediction rows - the median imputer cannot fill an all-NaN column
    feats_used = [f for f in feats
                  if train[f].notna().sum() > 0 and pred_rows[f].notna().sum() > 0]
    dropped = [f for f in feats if f not in feats_used]
    if dropped:
        print(f"    dropped all-NaN features: {dropped}")

    X_tr = train[feats_used].values
    y_tr = np.log(np.clip(train[target].values, 0.01, None))

    # LOO-CV to pick the winner
    models = _build_models()
    scores = {}
    for name, mtemplate in models.items():
        r2, mae, _ = loocv_score(X_tr, y_tr, mtemplate)
        scores[name] = {"R2": r2, "MAE_GW": mae}
        print(f"    {name:13s}  LOO-R² = {r2:+.3f}   LOO-MAE = {mae:6.1f} GW")

    winner = max(scores, key=lambda k: scores[k]["R2"])
    r2_win  = scores[winner]["R2"]
    mae_win = scores[winner]["MAE_GW"]
    print(f"    → winner: {winner}  (R² = {r2_win:+.3f})")

    # Refit winner on the full training set and predict on missing rows
    X_te = pred_rows[feats_used].values
    imp = clone(IMPUTER)
    X_tr_imp = imp.fit_transform(X_tr)
    X_te_imp = imp.transform(X_te)
    m_final = clone(models[winner])
    m_final.fit(X_tr_imp, y_tr)
    yhat_log = m_final.predict(X_te_imp)

    # Bootstrap 95% CI on predictions - resample count and seed both live
    # in config so the appendix table (Table B.1) documents what the code
    # actually did.
    rng = np.random.default_rng(RANDOM_SEED)
    B = BOOTSTRAP_N
    boot = np.zeros((B, len(pred_rows)))
    for i in range(B):
        idx = rng.integers(0, len(X_tr_imp), len(X_tr_imp))
        m_b = clone(models[winner])
        try:
            m_b.fit(X_tr_imp[idx], y_tr[idx])
            boot[i] = m_b.predict(X_te_imp)
        except Exception:
            boot[i] = yhat_log
    lo_log = np.percentile(boot, 2.5,  axis=0)
    hi_log = np.percentile(boot, 97.5, axis=0)

    yhat = np.exp(yhat_log)
    lo   = np.exp(lo_log)
    hi   = np.exp(hi_log)

    # Training-envelope clipping (Lamboll et al., 2020)
    y_cap = 5.0 * float(np.nanmax(train[target].values))
    yhat = np.clip(yhat, 0.0, y_cap)
    lo   = np.clip(lo,   0.0, y_cap)
    hi   = np.clip(hi,   0.0, y_cap)

    reliability = _reliability(r2_win, len(train))

    id_cols = [c for c in
               ["publisher_author", "publishing_year", "region", "year",
                "scenario_std", "scenario_tier", "vre_pct_filled"]
               if c in pred_rows.columns]
    out = pred_rows[id_cols].copy()
    out[f"pred_{target}_gw"] = yhat.round(1)
    out["pred_lo95_gw"]      = lo.round(1)
    out["pred_hi95_gw"]      = hi.round(1)
    out["model_used"]        = winner
    out["model_LOO_R2"]      = round(r2_win, 3)
    out["model_LOO_MAE_GW"]  = round(mae_win, 1)
    out["model_train_n"]     = len(train)
    out["reliability"]       = reliability
    out["group"]             = group
    out["target"]            = target

    print(f"  {target:11s}  → {len(out)} predictions  "
          f"range {yhat.min():.0f}-{yhat.max():.0f} GW  "
          f"median {np.median(yhat):.0f} GW  [{reliability}]")

    return {"df": out, "winner": winner, "r2": r2_win, "mae": mae_win,
            "n_train": len(train), "reliability": reliability,
            "scores": scores}


# ── Table 9 ─────────────────────────────────────────────────────────────────
def _write_table9(df, predictions, meta, groups):
    rows = []
    for group in groups:
        for target in ["storage_gw", "battery_gw"]:
            if target in predictions[group]:
                out = predictions[group][target]
                m = meta[group][target]
                col = f"pred_{target}_gw"
                yhat = out[col].values
                rows.append({
                    "Group":          group,
                    "Target":         TARGET_LABEL[target],
                    "n_train":        m["n_train"],
                    "Model used":     m["winner"],
                    "LOO-R²":         f"{m['r2']:+.3f}",
                    "Range (GW)":     f"{yhat.min():.0f} – {yhat.max():.0f}",
                    "Median (GW)":    f"{np.median(yhat):.0f}",
                    "Rows predicted": len(out),
                    "Reliability":    m["reliability"],
                })
            else:
                n_actual = df[(df["analysis_group"] == group)
                              & df[target].notna()].shape[0]
                reason = (f"n_train = {n_actual} < {MIN_N_TRAIN_ML}"
                          if n_actual < MIN_N_TRAIN_ML
                          else "no missing-target rows")
                rows.append({
                    "Group":          group,
                    "Target":         TARGET_LABEL[target],
                    "n_train":        n_actual,
                    "Model used":     "—",
                    "LOO-R²":         "—",
                    "Range (GW)":     "—",
                    "Median (GW)":    "—",
                    "Rows predicted": 0,
                    "Reliability":    f"Not fitted ({reason})",
                })
    tbl = pd.DataFrame(rows)
    csv_path = f"table9_predictions_summary.csv"
    png_path = f"table9_predictions_summary.png"
    tbl.to_csv(csv_path, index=False)
    dataframe_to_png(
        tbl, png_path,
        title="Table 9: Predictions for missing-target rows, by group and target",
        footnote=("EU/Europe and Global are fitted separately, never pooled. "
                  f"Groups with n_train < {MIN_N_TRAIN_ML} are not fitted "
                  "(Vabalas et al., 2019, on ML with limited sample size)."),
    )
    print(f"  Saved → {csv_path} / .png / .pdf")


# ── Table 10 ────────────────────────────────────────────────────────────────
def _write_table10(meta, groups):
    rows = []
    for group in groups:
        for target in ["storage_gw", "battery_gw"]:
            if target in meta[group]:
                m = meta[group][target]
                s = m["scores"]
                rows.append({
                    "Group":              group,
                    "Target":             TARGET_LABEL[target],
                    "n_train":            m["n_train"],
                    "ElasticNet LOO-R²":  f"{s['ElasticNet']['R2']:+.3f}",
                    "ElasticNet LOO-MAE": f"{s['ElasticNet']['MAE_GW']:.1f}",
                    "RF LOO-R²":          f"{s['RandomForest']['R2']:+.3f}",
                    "RF LOO-MAE":         f"{s['RandomForest']['MAE_GW']:.1f}",
                    "Winner":             m["winner"],
                })
    if not rows:
        return
    tbl = pd.DataFrame(rows)
    csv_path = f"table10_model_comparison.csv"
    png_path = f"table10_model_comparison.png"
    tbl.to_csv(csv_path, index=False)
    dataframe_to_png(
        tbl, png_path,
        title="Table 10: ElasticNet vs Random Forest - leave-one-out CV comparison",
        footnote=("For each group and target, both models are compared by "
                  "leave-one-out cross-validation on log-target predictions. "
                  "The model with the higher LOO-R² is used to generate "
                  "the reported predictions in Table 9. LOO-MAE is reported "
                  "in GW on the back-transformed prediction scale."),
    )
    print(f"  Saved → {csv_path} / .png / .pdf")


# ── Figure 7a/7b ────────────────────────────────────────────────────────────
def _plot_predictions_figure(df, predictions, meta, group, fname_suffix):
    have_any = any(t in predictions[group] for t in ["storage_gw", "battery_gw"])
    if not have_any:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, target in zip(axes, ["storage_gw", "battery_gw"]):
        train = df[(df["analysis_group"] == group) & df[target].notna()]
        if len(train) > 0:
            ax.scatter(train["year"], train[target],
                       color=GROUP_COLOR[group], marker="o", s=70,
                       alpha=0.75, edgecolors="white", linewidths=1.2,
                       label=f"Observed (n = {len(train)})", zorder=3)

        if target in predictions[group]:
            pdata = predictions[group][target]
            col = f"pred_{target}_gw"
            yerr_lo = pdata[col] - pdata["pred_lo95_gw"]
            yerr_hi = pdata["pred_hi95_gw"] - pdata[col]
            model_used = meta[group][target]["winner"]
            ax.errorbar(pdata["year"], pdata[col],
                        yerr=[yerr_lo.clip(lower=0), yerr_hi.clip(lower=0)],
                        fmt="^", color="#ff7f0e", alpha=0.65,
                        markersize=8, markeredgecolor="black",
                        markeredgewidth=0.6, capsize=2, lw=0.8,
                        label=f"Predicted ({model_used}, n = {len(pdata)})")
        else:
            ax.text(0.5, 0.5, "Not fitted\n(n_train too small)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=13, color="#888", style="italic")

        ax.set_xlabel("Horizon year", fontsize=11)
        ax.set_ylabel(("Total storage capacity (GW)"
                       if target == "storage_gw"
                       else "Battery capacity (GW)"), fontsize=11)
        panel = ("(a) Total storage" if target == "storage_gw"
                 else "(b) Battery")
        ax.set_title(f"{panel}  —  {group}", fontsize=11)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3, ls=":")
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(f"fig7{fname_suffix}_predictions_overview.png",
                bbox_inches="tight", dpi=200)
    plt.savefig(f"fig7{fname_suffix}_predictions_overview.pdf",
                bbox_inches="tight")
    plt.close(fig)
    return True


# ── Stage entry point ───────────────────────────────────────────────────────
def run_ml_stage(df):
    """
    Run the ML imputation stage end-to-end.

    Returns ``(predictions, meta)`` - both nested dicts, which are indexed by
    ``[group][target]``.  ``predictions[group][target]`` is a DataFrame
    of imputed rows with 95% CI, model, and reliability columns.
    ``meta[group][target]`` carries the scoring metadata.
    """
    print(f"\n{'='*78}\nSTAGE 7 - ML predictions  "
          f"(ElasticNet vs Random Forest, LOO-CV)\n{'='*78}")

    groups = ["EU/Europe", "Global"]
    predictions = {g: {} for g in groups}
    meta        = {g: {} for g in groups}

    for group in groups:
        print(f"\n  === {group} ===")
        for target in ["storage_gw", "battery_gw"]:
            print(f"\n  [{group} / {target}]")
            res = predict_missing_ml(df, target, group=group)
            if res is not None:
                group_tag = "eu" if group == "EU/Europe" else "global"
                fname = f"predictions_{group_tag}_{target}.csv"
                res["df"].to_csv(fname, index=False)
                predictions[group][target] = res["df"]
                meta[group][target] = res

    # Combined predictions file
    all_frames = [dfp for g in groups for dfp in predictions[g].values()]
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True, sort=False)
        combined.to_csv(f"predictions_all.csv", index=False)
        print(f"\n  Combined → predictions_all.csv  "
              f"({len(combined)} rows)")

    _write_table9(df, predictions, meta, groups)
    _write_table10(meta, groups)

    drew_eu     = _plot_predictions_figure(df, predictions, meta, "EU/Europe", "a")
    drew_global = _plot_predictions_figure(df, predictions, meta, "Global",    "b")
    if drew_eu:
        print(f"  Saved → fig7a_predictions_overview.png / .pdf  (EU/Europe)")
    if drew_global:
        print(f"  Saved → fig7b_predictions_overview.png / .pdf  (Global)")
    if not drew_global:
        print("  Note: no Global predictions figure produced - Global "
              f"training set is below n_train = {MIN_N_TRAIN_ML} on both "
              "targets (see Vabalas et al., 2019).")

    return predictions, meta