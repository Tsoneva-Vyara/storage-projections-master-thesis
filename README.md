# Cross-Source Econometric Analysis of European Storage Projections

Analysis code for the master's thesis
**_Estimating Future Energy Storage Requirements for Renewable Integration in Europe: A Gap-Filling Approach Using Machine Learning_**
by _Vyara Tsoneva_, _Vienna University of Economics and Business_, 05.08.2026.
_Department of Information Systems and Operations Management Institute for Data, Energy, and Sustainability (IDEaS)_, Supervisor: MSc Robin Fischer, Co-supervisor: DSc Behnam Zakeri.

---

## Overview

This repository contains the full analysis pipeline for the thesis. The pipeline reads a cross-source dataset of 141 scenario-observations, which are extracted from 45 published European storage projections (2018-2026) and produces every table and figure reported in Chapter 5 (and also tables and figures, which are discussed in Chapter 2 and Appendix A, B, and C) of the thesis.

The pipeline is split into small modules of one per processing step, so each stage can be re-read and debugged in isolation. A single entry point (`src/run_analysis.py`) orchestrates them in order.

There are three analytical layers:

1. **Descriptive layer**: dataset construction, VRE-share imputation, variable-coverage heatmap.
2. **Econometric layer**: univariate log-log OLS of total storage and battery capacity on VRE share, plus a multivariate diagnostic of battery capacity on solar and wind capacity.
3. **Machine-learning layer**: imputation of missing storage and battery values for scenarios that report the drivers but do not include the target, using ElasticNet vs Random Forest with leave-one-out cross-validation, per-group (EU/Europe vs Global) fits, and bootstrap 95% confidence intervals with reliability flags.

---

## Requirements

- Python **3.9 or higher** (tested on 3.10, 3.11, 3.12)
- Dependencies pinned in [`requirements.txt`](requirements.txt)

Core packages: `numpy`, `pandas`, `matplotlib`, `statsmodels`, `scikit-learn`, `openpyxl`.

---

## Installation

Clone or download this repository, then create a virtual environment and install the dependencies:

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Or, if using the included `Makefile`:

```bash
make install
```

---

## Usage

1. **Place the dataset** in the `data/` directory. Expected filename:
```
   data/Energy_Storage_Data_Collection_Vyara_Tsoneva.xlsx
```
   See [`data/README.md`](data/README.md) for provenance and formatting notes.

2. **First-time setup:** create the `outputs/` folder if it does not already exist:

```bash
   mkdir outputs
```

3. **Run the pipeline** from the repository root:

   ```bash
   make run
   ```

   or manually:

   ```bash
   cd outputs
   cp ../data/Energy_Storage_Data_Collection_Vyara_Tsoneva.xlsx .
   python ../src/run_analysis.py
   ```

   The pipeline writes all outputs (CSVs, PNGs, PDFs, and the interpretive report) into the current working directory, which is why running it from `outputs/` keeps everything organized.

4. **Inspect the outputs.** All figures, tables, and the interpretive report appear in `outputs/`.

To remove all generated outputs and start clean:

```bash
make clean
```

---

## Repository structure

```
storage-projections-thesis/
├── README.md              This file
├── LICENSE                MIT
├── CITATION.cff           How to cite this software
├── requirements.txt       Pinned Python dependencies
├── Makefile               install / run / check / clean targets
├── .gitignore             Excludes venv, __pycache__, data, outputs
├── src/                   Analysis pipeline; one module per step
│   ├── run_analysis.py         Entry point; orchestrates the pipeline
│   ├── config.py               All constants: paths, thresholds, colours,
│   │                            feature sets, source-category rules,
│   │                            hyperparameter grids, random seed
│   ├── plotting.py             Shared plot helpers (rcParams, dataframe_to_png)
│   ├── io_utils.py             Dataset discovery and loading
│   ├── preprocessing.py        Region grouping, source category, scenario tier,
│   │                            VRE imputation, log transforms
│   ├── figure1_projections.py  Stage 3b: Figure 1 (§2.3)
│   ├── descriptives.py         Stage 4: Table 6 descriptive statistics
│   ├── coverage.py             Stage 4b: Figure 4 variable coverage heatmap
│   ├── ols_univariate.py       Stage 5: Table 7 + Figure 5 (+ optional Figure 5b)
│   ├── ols_multivariate.py     Stage 6: Table 8 + Figure 6 (+ multicollinearity)
│   ├── ml_imputation.py        Stage 7: Tables 9-10 + Figures 7a/7b
│   ├── appendix_tables.py      Stage 7b: Tables C.1 (predictions summary) and
│   │                            B.1 (hyperparameter grids and seed settings)
│   └── report.py               Stage 8: plain-language interpretive report
├── data/
│   └── README.md          Where to place the dataset
└── outputs/               Created on first run; regenerable analysis outputs arrive here
```

### How the modules connect

`run_analysis.py` calls each stage in order. The stages exchange state through the pandas DataFrame and (in the OLS/ML stages) through result dicts:

```
                              ┌──────────────────┐
                              │  config.py       │  constants
                              │  plotting.py     │  shared helpers
                              └──────────────────┘
                                    ▲
                                    │ imports
                                    │
  io_utils ─► preprocessing ─► figure1_projections ─► descriptives ─► coverage
                     │                                                     │
                     └─► ols_univariate ─► ols_multivariate ─► ml_imputation
                                                                     │
                                                        appendix_tables ─► report
```

Every module has a docstring at the top, which explains what it does and, where relevant, which thesis section it corresponds to.

---

## Outputs and their mapping to the thesis

Each output file corresponds to a specific table or figure in the thesis.

| Output file                                        | Thesis reference     | Content                                                    |
|----------------------------------------------------|----------------------|------------------------------------------------------------|
| `v18_fig1_projections_by_source.{png,pdf}`         | Figure 1, §2.3       | EU/Europe storage projections by horizon year and source category |
| `v18_table6_descriptive_stats.{csv,png,pdf}`       | Table 6, §5.1        | Descriptive statistics, EU/Europe subgroup                 |
| `v18_fig4_variable_coverage.{png,pdf}`             | Figure 4, §4.2       | Variable-coverage heatmap by subgroup                      |
| `v18_table7_univariate_ols.{csv,png,pdf}`          | Table 7, §5.2        | Univariate log-log OLS results (storage, battery)          |
| `v18_fig5_ols_scatter.{png,pdf}`                   | Figure 5, §5.2       | Univariate OLS scatters with 95% CI band                  |
| `v18_fig5b_ols_scatter_global.{png,pdf}`           | (optional)           | Global-only OLS if sample allows; not produced     |
| `v18_table8_multivariate_ols.{csv,png,pdf}`        | Table 8, §5.3        | Multivariate battery ~ solar + wind (diagnostic)           |
| `v18_fig6_multivariate_scatter.{png,pdf}`          | Figure 6, §5.3       | Two univariate scatters (Battery vs Solar, vs Wind)        |
| `v18_table9_predictions_summary.{csv,png,pdf}`     | Table 9, §5.4        | ML predictions summary by group and target                 |
| `v18_table10_model_comparison.{csv,png,pdf}`        | Table 10, §5.4        | ElasticNet vs Random Forest LOO-CV comparison              |
| `v18_fig7a_predictions_overview.{png,pdf}`         | Figure 7, §5.4      | EU/Europe observed vs predicted, by horizon year           |
| `v18_fig7b_predictions_overview.{png,pdf}`         | (optional)           | Global overview if sample allows; not produced     |
| `v18_tableC1_predictions_by_horizon.{csv,png,pdf}` | Table C.1, App. C    | EU/Europe predictions summarized by horizon year × target  |
| `v18_tableB1_hyperparameter_grids.{csv,png,pdf}`   | Table B.1, App. B    | Hyperparameter grids and seed settings for both models     |
| `v18_predictions_eu_battery_gw.csv`                | Chapter 5 supp.      | Row-level battery predictions with 95% bootstrap CI       |
| `v18_predictions_eu_storage_gw.csv`                | Chapter 5 supp.      | Row-level storage predictions with 95% bootstrap CI       |
| `v18_predictions_all.csv`                          | Chapter 5 supp.      | Combined prediction file across groups                     |
| `v18_variable_coverage.csv`                        | Chapter 4 supp.; Table A1, App. A | Full variable-coverage table by subgroup                   |
| `v18_INTERPRETIVE_REPORT.txt`                      | (working document)   | Auto-generated interpretive summary; key numbers   |

---

## Reproducibility

- **Random seed 42** is used for the bootstrap resamples on the ML predictions, for the Random Forest fits, and for the ElasticNet inner-CV splits. The seed lives in `config.RANDOM_SEED`; every consumer imports it from there. Re-running the pipeline on the same dataset produces bitwise-identical outputs.
- **Hyperparameter grids** (`config.ELASTIC_NET_GRID`, `config.RANDOM_FOREST_GRID`) are shared between the ML step and Appendix Table B.1 so the code and the documentation cannot drift apart. If a grid entry is changed in `config.py` and both are updated simultaneously.
- All computed statistics (OLS coefficients, CV scores, prediction ranges) are printed to `stdout` during execution and also written to `INTERPRETIVE_REPORT.txt`.
- The pipeline applies a minimum training-sample size of eight rows for the ML step, which follows Vabalas et al. (2019). Groups that fall below the threshold are reported as *not fitted* rather than silently omitted.
- Bracket aggregation for the OLS stage collapses rows, which share (publisher × publication year × horizon year × VRE share) to their median. This step is implemented in the OLS regressions but **not** to the ML training step, because the ML training set is already small and every independent row contributes to the median-imputed feature space.
- The multivariate specification is retained as a diagnostic in Table 8 but is **not used for a compositional interpretation**: on the six-row sample, `corr(log solar, log wind) ≈ +0.99` and `VIF > 100` on both predictors. Section 5.3 uses the two univariate scatters in Figure 6 instead. This is documented inline in `src/ols_multivariate.py`.

---

## Methodological references

The four methodological references cited inline in the code:

- **Vabalas, A., Gowen, E., Poliakoff, E., and Casson, A. J.** (2019). Machine learning algorithm validation with a limited sample size. *PLoS ONE*, 14(11), e0224365. - n_train ≥ 8 threshold for ML imputation.
- **Zou, H., and Hastie, T.** (2005). Regularization and variable selection via the Elastic Net. *Journal of the Royal Statistical Society B*, 67(2), 301-320. - ElasticNet.
- **Breiman, L.** (2001). Random Forests. *Machine Learning*, 45(1), 5-32. - Random Forest.
- **Cameron, A. C., and Miller, D. L.** (2015). A Practitioner's Guide to Cluster-Robust Inference. *Journal of Human Resources*, 50(2), 317-372. - why clustered SE fails at small N.

Substantive references are documented in the thesis reference list.

---

## Extending the pipeline

Because each processing step is a separate module, common extensions are localised:

| To change …                             | Edit …                       |
|-----------------------------------------|------------------------------|
| Column names in the source spreadsheet  | `config.RENAME`              |
| European fleet capacity factors    | `config.CF_SOLAR`, `CF_WIND`, `CF_HYDRO` |
| Region grouping rules                   | `config.ANALYSIS_GROUP_MAP` + `preprocessing._region_group` |
| Source-category assignment              | `config.SOURCE_CATEGORY_RULES` |
| Source-category markers/colors         | `config.SOURCE_CATEGORY_STYLE` |
| Scenario-tier keywords                  | `config.AMBIT_KW`, `CONS_KW` |
| Minimum training-sample threshold       | `config.MIN_N_TRAIN_ML`      |
| ML feature sets                         | `config.FEATURE_SET_NAMES`   |
| ML hyperparameter grids                 | `config.ELASTIC_NET_GRID`, `config.RANDOM_FOREST_GRID` |
| Random seed for reproducibility         | `config.RANDOM_SEED`         |
| Bootstrap resamples for prediction CIs  | `config.BOOTSTRAP_N`, `config.BOOTSTRAP_ALPHA` |
| Reliability-flag thresholds             | `config.RELIABLE_R2_MIN` etc. |
| Table look (fonts, colors)             | `plotting.py`                |
| Scatter-plot colours per group          | `config.GROUP_COLOR`         |

Adding a new stage:

1. Create a new module, e.g. `src/robustness.py`, which follows the same pattern (a `run_*_stage(df, …)` function that writes its outputs and returns whatever downstream stages will need).
2. Import it in `run_analysis.py` and call it in the right position.
3. Add its output filenames to the table above.

---

## Syntax check

You can verify that every module imports without running the pipeline:

```bash
make check
```

This runs `python -m py_compile` on every file in `src/` and reports the first syntax error, if any.

---

## Citation

If you use this code, please cite the thesis and the software separately. The recommended citation format is provided in [`CITATION.cff`](CITATION.cff).

Short-form citation:

> _Vyara Tsoneva_ (2026). *Estimating Future Energy Storage Requirements for Renewable Integration in Europe: A Gap-Filling Approach Using Machine Learning* [Master's thesis, _Vienna University of Economics and Business_]. Software available at _[REPOSITORY URL]_.

---

## License

MIT License - see [`LICENSE`](LICENSE) for full text.

---