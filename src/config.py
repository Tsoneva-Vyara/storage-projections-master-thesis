"""
config.py - constants and configuration for the analysis pipeline.

Everything in this file is a plain constant.  No computation runs at
import time.  Import from this module wherever you need a threshold, a
label, or a color, so that changing a value here changes it everywhere.
"""

# ── Dataset discovery ────────────────────────────────────────────────────────
# The pipeline searches the current working directory for the file
# that matches this name.  Add new candidates to the front.
CANDIDATE_FILES = [
    "Energy_Storage_Data_Collection_Vyara_Tsoneva.xlsx",
]

# ── Column renaming (source → internal) ──────────────────────────────────────
RENAME = {
    "№": "study_no", "Study": "study", "Publisher/Author": "publisher_author",
    "Publishing Year": "publishing_year", "Study Additional Info": "study_info",
    "Region": "region", "Horizon (Year)": "year",
    "Total Storage (GW)": "storage_gw", "Total Storage (GWh)": "storage_gwh",
    "Battery (GW)": "battery_gw", "Battery (GWh)": "battery_gwh",
    "Pumped Hydro (GW)": "pumped_hydro_gw",
    "Hydrogen Turbine (GW)": "hydrogen_turbine_gw",
    "Hydrogen Turbine (GWh)": "hydrogen_turbine_gwh",
    "VRE (%)": "vre_pct", "VRE (GW)": "vre_gw",
    "RES (%)": "res_pct", "RES (GW)": "res_gw",
    "Solar (%)": "solar_pct", "Solar (GW)": "solar_gw",
    "Wind (%)": "wind_pct", "Wind (GW)": "wind_gw",
    "Nuclear/Coal (%)": "nuclear_coal_pct",
    "Nuclear (GW)": "nuclear_gw", "Coal (GW)": "coal_gw",
    "Natural Gas/Hydro (%)": "natural_gas_hydro_pct",
    "Natural Gas (GW)": "natural_gas_gw",
    "Hydro (GW)": "hydro_gw", "Oil (GW)": "oil_gw",
    "H2 Electrolysers (GWh)": "h2_electrolysers_gwh",
    "H2 Electrolyzers (GW)": "h2_electrolyzers_gw",
    "Electricity Demand (GWh)": "elec_demand_gwh",
    "Other Flexibility, H/M/L": "other_flexibility",
    "Standardized Scenario Category": "scenario_std",
    "Scenario Assumption / Notes": "scenario_raw",
}

# ── Source-category classification (publisher → category) ────────────────────
# Used by Figure 1 to distinguish where each projection comes from.  Matching
# is done on the lower-cased publisher_author string with substring rules in
# preprocessing._source_category, so minor spelling variants (e.g. trailing
# spaces, "ENTSO-E" vs "ENTSO‑E") are grouped correctly.
#
# The four categories follow the substantive distinction used in the thesis:
#   Institutional  — intergovernmental / EU / national research bodies
#   Industry       — industry associations, market analysts, trade groups
#   Network-based  — the European TSO/system-operator networks
#   Academic       — peer-reviewed academic model runs
SOURCE_CATEGORY_RULES = [
    # (substring on lowercased publisher, category)
    ("entso",                            "Network-based"),
    ("iea",                              "Institutional"),
    ("irena",                            "Institutional"),
    ("european commission",              "Institutional"),
    ("joint research centre",            "Institutional"),
    ("jrc",                              "Institutional"),
    ("german environment agency",        "Institutional"),
    ("solarpower europe",                "Industry"),
    ("windeurope",                       "Industry"),
    ("ease",                             "Industry"),
    ("ember",                            "Industry"),
    ("statista",                         "Industry"),
    ("wood mackenzie",                   "Industry"),
    ("bloomberg",                        "Industry"),
    ("global renewables alliance",       "Industry"),
]
# Anything not matched by the rules above and containing "et al." (or ending
# in a lowercase surname) is treated as Academic - the fallback in
# preprocessing._source_category.
SOURCE_CATEGORY_DEFAULT_ACADEMIC = True

# Marker style + color per source category on Figure 1.
SOURCE_CATEGORY_STYLE = {
    "Institutional": {"marker": "o", "color": "#1f77b4"},
    "Industry":      {"marker": "s", "color": "#d62728"},
    "Network-based": {"marker": "^", "color": "#9467bd"},
    "Academic":      {"marker": "D", "color": "#8c564b"},
}

# ── Analytical grouping (region → group) ─────────────────────────────────────
# EU/Europe pools EU-27, EU-28, and broader-Europe studies on the physical
# grounds of a single ENTSO-E synchronous area.  Germany-only and unattributed
# studies are excluded from the modelling stage.
ANALYSIS_GROUP_MAP = {
    "EU27":    "EU/Europe",
    "EU28":    "EU/Europe",
    "Europe":  "EU/Europe",
    "Global":  "Global",
    "Germany": "EXCLUDED",
    "Other":   "EXCLUDED",
}

# ── Scenario tier classification keywords ────────────────────────────────────
# Tier 2 (ambitious), 1 (central, default), 0 (conservative).
AMBIT_KW = [
    "nze", "nez", "technical potential", "upper bound", "opt-mix",
    "supreme", "decarbonizing", "net-zero", "decarbonisation",
    "100% renewable",
]
CONS_KW = [
    "historical", "no dsm", "nodsm", "minimum required", "lower bound",
    "outlook", "projection", "limvre", "forecast", "conservative",
    "low storage", "low renewable",
]

# ── VRE-share imputation: European fleet capacity factors (CFs) ───────────────
# CF_SOLAR : European fleet PV CF (IRENA, 2025a).
# CF_WIND  : European fleet wind CF, blended on- and off-shore
#            (IRENA, 2025a; IEA, 2024b).
# CF_HYDRO : European fleet hydropower CF.  Sits within the 0.35-0.42
#            range reported by IRENA (2025a) for European hydropower, which
#            reflects the older fleet age and the hydrological variability
#            of the region (Quaranta et al., 2025).
CF_SOLAR = 0.13
CF_WIND  = 0.28
CF_HYDRO = 0.40

# ── Sample-size thresholds ───────────────────────────────────────────────────
# Below MIN_N_GLOBAL_OLS aggregated Global observations, no separate Global OLS
# is reported.  Below MIN_N_TRAIN_ML training rows, no ML imputation is
# attempted for that group / target - see Vabalas et al. (2019).
MIN_N_GLOBAL_OLS = 4
MIN_N_TRAIN_ML   = 8

# ── Colors for scatter groups and imputation-source markers ────────────────
GROUP_COLOR = {"EU/Europe": "#1f77b4", "Global": "#d62728"}
SRC_MARKER  = {"observed": "o", "imputed_CF": "s", "imputed_RESdecomp": "^"}

# ── Reliability-flag thresholds for ML predictions ───────────────────────────
# "Reliable"   - R² >= 0.30 AND n_train >= 15
# "Indicative" - R² >=  0.0 AND above the minimum training threshold
# "Unreliable" - R² <   0.0
RELIABLE_R2_MIN     = 0.30
RELIABLE_N_TRAIN    = 15
INDICATIVE_R2_MIN   = 0.0

# ── Variables reported in the descriptive-statistics table (Table 4) ─────────
DESC_VARS = [
    ("storage_gw",       "Total storage (GW)"),
    ("battery_gw",       "Battery (GW)"),
    ("vre_pct_filled",   "VRE share (%)"),
    ("solar_gw",         "Solar (GW)"),
    ("wind_gw",          "Wind (GW)"),
    ("pumped_hydro_gw",  "Pumped hydro (GW)"),
    ("hydro_gw",         "Hydropower (GW)"),
    ("natural_gas_gw",   "Natural gas (GW)"),
    ("elec_demand_gwh",  "Electricity demand (GWh)"),
    ("year",             "Horizon year"),
]

# ── Variables shown on the coverage heatmap (Figure 3) ───────────────────────
HEATMAP_VARS = [
    ("storage_gw",        "Total storage (GW)"),
    ("battery_gw",        "Battery (GW)"),
    ("vre_pct",           "VRE share (%) - observed"),
    ("vre_pct_filled",    "VRE share (%) - filled (obs + imputed)"),
    ("solar_gw",          "Solar capacity (GW)"),
    ("wind_gw",           "Wind capacity (GW)"),
    ("pumped_hydro_gw",   "Pumped hydro (GW)"),
    ("elec_demand_gwh",   "Electricity demand (GWh)"),
    ("natural_gas_gw",    "Natural gas (GW)"),
    ("hydro_gw",          "Hydropower (GW)"),
    ("scenario_tier",     "Scenario tier (ordinal)"),
    ("year_rel",          "Horizon year offset"),
]

# ── Coverage snapshot (for the appendix and diagnostic) ──────────────────────
COV_VARS = [
    "storage_gw", "battery_gw", "vre_pct", "vre_pct_filled",
    "solar_gw", "wind_gw", "pumped_hydro_gw", "log_elec_demand",
    "natural_gas_gw", "hydro_gw", "scenario_tier", "year_rel",
]

# ── Target labels for reporting ──────────────────────────────────────────────
TARGET_LABEL = {"storage_gw": "Total storage", "battery_gw": "Battery"}

# ── ML feature sets - physical drivers plus minimal scenario context ─────────
# Continuous capacity and demand variables enter on the log scale via the
# +1-shift transformation, which is applied in preprocessing.build_log_predictors;
# bounded shares (VRE%) and ordinal / count variables enter on their
# native scale.  Missing predictors are median-imputed inside each LOO fold.
FEATURE_SET_NAMES = {
    "storage_gw": [
        "vre_pct_filled",
        "log_solar_gw",
        "log_wind_gw",
        "log_elec_demand",
        "log_natural_gas_gw",
        "log_hydro_gw",
        "scenario_tier",
        "year_rel",
    ],
    "battery_gw": [
        "vre_pct_filled",
        "log_solar_gw",
        "log_wind_gw",
        "log_elec_demand",
        "log_natural_gas_gw",
        "log_hydro_gw",
        "log_pumped_hydro_gw",
        "scenario_tier",
        "year_rel",
    ],
}

# ── Random-seed policy ───────────────────────────────────────────────────────
# All random draws in the pipeline are seeded from this value:
#   - RandomForestRegressor(random_state=RANDOM_SEED)
#   - ElasticNetCV(random_state=RANDOM_SEED)
#   - np.random.default_rng(RANDOM_SEED) for the bootstrap resamples
# Fixing the seed at import time keeps every re-run bitwise-identical.
RANDOM_SEED = 42

# ── Hyperparameter grids for the ML step ─────────────────────────────────────
# Documented here so the appendix table (Table C.2) and the ML code read from
# a single source of truth.  Grids follow standard defaults for small-sample
# cross-source panels (Hastie et al., 2009, "The Elements of Statistical
# Learning", 2nd ed., §7.10).  The ElasticNet grid keeps three L1-ratio
# candidates spanning ridge-like (0.1), balanced (0.5), and lasso-like (0.9)
# behavior; the alpha grid is a decade-spaced logarithmic sweep from 10^-3
# to 10^2.  The Random Forest grid uses shallow trees (max_depth=4) and a
# minimum-leaf floor (min_samples_leaf=3) to counter overfitting on the
# small training set.
import numpy as _np  # local alias - keeps the grids importable as constants

ELASTIC_NET_GRID = {
    "l1_ratio":     [0.1, 0.5, 0.9],
    "alphas":       _np.logspace(-3, 2, 10),
    "max_iter":     3000,
    "cv":           5,
    "random_state": RANDOM_SEED,
}
RANDOM_FOREST_GRID = {
    "n_estimators":     200,
    "max_depth":        4,
    "min_samples_leaf": 3,
    "random_state":     RANDOM_SEED,
    "n_jobs":           -1,
}

# Bootstrap settings for the 95% CI on predictions.
BOOTSTRAP_N = 500
BOOTSTRAP_ALPHA = 0.05  # → 2.5% and 97.5% percentiles