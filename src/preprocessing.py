"""
preprocessing.py - everything that mutates the loaded DataFrame before analysis.

Order:
  1. assign_analysis_group   - EU/Europe vs Global vs EXCLUDED
  2. classify_scenario_tier  - 0 (conservative), 1 (central), 2 (ambitious)
  3. impute_vre_share        - Methods A / B / C, see thesis §4.3
  4. build_log_predictors    - log(x + 1) shift for capacity variables
  5. build_coverage_table    - writes variable_coverage.csv

Each function takes the DataFrame and returns it (mutated in place and
returned so the pipeline reads left-to-right).
"""
import numpy as np
import pandas as pd

from config import (
    ANALYSIS_GROUP_MAP,
    AMBIT_KW, CONS_KW,
    CF_SOLAR, CF_WIND, CF_HYDRO,
    COV_VARS,
    SOURCE_CATEGORY_RULES, SOURCE_CATEGORY_DEFAULT_ACADEMIC
)


# ── 1. Geographic grouping ──────────────────────────────────────────────────
def _region_norm(r):
    return (str(r).strip()
            .replace("\u2011", "-").replace("\u2013", "-").lower())


def _region_group(r):
    r = _region_norm(r)
    if "eu-28"   in r: return "EU28"
    if "eu-27"   in r: return "EU27"
    if "germany" in r: return "Germany"
    if "global"  in r: return "Global"
    if "europe"  in r: return "Europe"
    return "Other"


def assign_analysis_group(df):
    """Add ``region_group`` and ``analysis_group`` columns."""
    df["region_group"] = df["region"].apply(_region_group)
    df["analysis_group"] = df["region_group"].map(ANALYSIS_GROUP_MAP)
    n_excluded = (df["analysis_group"] == "EXCLUDED").sum()
    print(f"  region groups: "
          f"EU/Europe={sum(df['analysis_group'] == 'EU/Europe')}, "
          f"Global={sum(df['analysis_group'] == 'Global')}, "
          f"EXCLUDED (Germany/Other)={n_excluded}")
    return df


# ── 1b. Source-category classification ──────────────────────────────────────
# Lookup based on the rule of the (trimmed, lower-cased) publisher/author string.
# Peer-reviewed academic sources are recognized by the "et al." convention
# in the publisher column and by the residual set (anything the rules do
# not classify explicitly).
def _source_category(publisher):
    if pd.isna(publisher):
        return "Unclassified"
    p = str(publisher).strip().lower()
    for substr, category in SOURCE_CATEGORY_RULES:
        if substr in p:
            return category
    if SOURCE_CATEGORY_DEFAULT_ACADEMIC and (
        "et al" in p or p.replace(".", "").split()[-1].isalpha()
    ):
        return "Academic"
    return "Unclassified"


def assign_source_category(df):
    """Add ``source_category`` column (Institutional / Industry / Network-based / Academic)."""
    df["source_category"] = df["publisher_author"].apply(_source_category)
    counts = df["source_category"].value_counts().to_dict()
    print(f"  source categories: {counts}")
    return df


# ── 2. Scenario-tier classification ─────────────────────────────────────────
def _classify(t):
    if pd.isna(t):
        return 1
    t = str(t).lower()
    if any(k in t for k in AMBIT_KW): return 2
    if any(k in t for k in CONS_KW):  return 0
    return 1


def classify_scenario_tier(df):
    """Add ``scenario_tier`` (0/1/2) and ``year_rel`` (year - 2018)."""
    df["scenario_tier"] = df["scenario_std"].apply(_classify)
    df["year_rel"] = df["year"] - 2018
    return df


# ── 3. VRE-share imputation ─────────────────────────────────────────────────
# Method A : use directly reported vre_pct where present.
# Method B : back-calculate from the study's own solar_gw + wind_gw + demand.
# Method C : decompose the study's own RES% with the use of its own capacity breakdown.
def _impute_vre_pct(row):
    if pd.notna(row["vre_pct"]):
        return row["vre_pct"], "observed"
    s = row.get("solar_gw")
    w = row.get("wind_gw")
    d = row.get("elec_demand_gwh")
    if pd.notna(s) and pd.notna(w) and pd.notna(d) and d > 0:
        v = (s * CF_SOLAR + w * CF_WIND) * 8760.0 / d * 100.0
        return min(v, 100.0), "imputed_CF"
    h   = row.get("hydro_gw")
    res = row.get("res_pct")
    if (pd.notna(s) and pd.notna(w) and pd.notna(h) and pd.notna(res)
            and (s * CF_SOLAR + w * CF_WIND + h * CF_HYDRO) > 0):
        vre_gen = s * CF_SOLAR + w * CF_WIND
        all_ren_gen = vre_gen + h * CF_HYDRO
        v = res * vre_gen / all_ren_gen
        return min(v, 100.0), "imputed_RESdecomp"
    return np.nan, "missing"


def impute_vre_share(df):
    """Add ``vre_pct_filled`` and ``vre_pct_source`` columns."""
    imp_res = df.apply(_impute_vre_pct, axis=1, result_type="expand")
    df["vre_pct_filled"] = imp_res[0]
    df["vre_pct_source"] = imp_res[1]
    n_obs  = (df["vre_pct_source"] == "observed").sum()
    n_impB = (df["vre_pct_source"] == "imputed_CF").sum()
    n_impC = (df["vre_pct_source"] == "imputed_RESdecomp").sum()
    print(f"  VRE% coverage: observed={n_obs}  imputed_CF={n_impB}  "
          f"imputed_RESdecomp={n_impC}  "
          f"total={df['vre_pct_filled'].notna().sum()}")
    return df


# ── 4. Log-transformed predictors ───────────────────────────────────────────
# TRANSFORMATION RULE (see thesis §4.2):
#     log_x = log(max(x, 0) + 1)      NaN treated as 0 before shifting.
# The +1 shift keeps the transformation defined at zero (a scenario with no
# natural-gas capacity or no PHS is a valid observation and must remain in
# the sample).  Electricity demand is floored at 1 GWh rather than shifted,
# since zero demand is not a valid scenario and any imputed near-zero value
# should be treated as a minimum plausible bound.
def build_log_predictors(df):
    """Add the log-transformed capacity and demand columns used by the ML step."""
    df["log_elec_demand"]     = np.log(df["elec_demand_gwh"].clip(lower=1.0))
    df["log_natural_gas_gw"]  = np.log(df["natural_gas_gw"].clip(lower=0).fillna(0) + 1.0)
    df["log_hydro_gw"]        = np.log(df["hydro_gw"].clip(lower=0).fillna(0) + 1.0)
    df["log_solar_gw"]        = np.log(df["solar_gw"].clip(lower=0).fillna(0) + 1.0)
    df["log_wind_gw"]         = np.log(df["wind_gw"].clip(lower=0).fillna(0) + 1.0)
    df["log_pumped_hydro_gw"] = np.log(df["pumped_hydro_gw"].clip(lower=0).fillna(0) + 1.0)
    return df


# ── 5. Coverage snapshot ────────────────────────────────────────────────────
def build_coverage_table(df):
    """
    Save the per-variable, per-group coverage table used in Appendix A.

    Writes ``variable_coverage.csv`` in the current working
    directory and returns the DataFrame.
    """
    rows = []
    for g in ["EU/Europe", "Global"]:
        sub = df[df["analysis_group"] == g]
        for v in COV_VARS:
            rows.append({
                "Group": g, "Variable": v,
                "n_total": sub[v].notna().sum() if v in sub.columns else 0,
                "n_with_storage_gw":
                    (sub[v].notna() & sub["storage_gw"].notna()).sum()
                    if v in sub.columns else 0,
                "n_with_battery_gw":
                    (sub[v].notna() & sub["battery_gw"].notna()).sum()
                    if v in sub.columns else 0,
            })
    cov = pd.DataFrame(rows)
    cov.to_csv(f"variable_coverage.csv", index=False)
    return cov