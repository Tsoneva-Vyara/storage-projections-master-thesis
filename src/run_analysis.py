#!/usr/bin/env python3
"""
run_analysis.py - entry point for the thesis analysis pipeline.

Reads the cross-source scenario dataset from the current working directory,
runs the full pipeline (preprocessing → descriptives → OLS → ML imputation →
interpretive report), and writes every table and figure into the CWD.

Run it as::

    cd outputs && cp ../data/Energy_Storage_Data_Collection_Vyara_Tsoneva.xlsx .
    python ../src/run_analysis.py

or via the Makefile from the repository root::

    make run

The pipeline is intentionally split across small modules of one per
processing step, so each stage can be re-read and debugged in isolation.
The order matters: each stage assumes the columns, which are produced by the
previous ones.
"""
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# The pipeline modules live alongside this file.  When invoked as
# ``python ../src/run_analysis.py`` from outputs/, the interpreter needs
# src/ on sys.path so the plain imports below resolve.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils        import discover_dataset_file, load_dataset
from preprocessing   import (
    assign_analysis_group, assign_source_category,
    classify_scenario_tier,
    impute_vre_share, build_log_predictors, build_coverage_table,
)
from figure1_projections import plot_figure1
from descriptives    import run_descriptives
from coverage        import plot_coverage_heatmap
from ols_univariate  import run_univariate_stage
from ols_multivariate import run_multivariate_stage
from ml_imputation   import run_ml_stage
from appendix_tables import run_appendix_stage
from report          import write_interpretive_report


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"THESIS ANALYSIS - modular pipeline")
    print("  Univariate + multivariate log-log OLS · plain OLS SE")
    print("  Bracket aggregation on OLS only.  ML kept for the missing-value step.")
    print("=" * 78)

    # Stage 1 - Load
    path = discover_dataset_file()
    df = load_dataset(path)

    # Stages 2-3 - Preprocessing
    df = assign_analysis_group(df)
    df = assign_source_category(df)
    df = classify_scenario_tier(df)
    df = impute_vre_share(df)
    df = build_log_predictors(df)
    build_coverage_table(df)

    # Stage 3b - Figure 1 (source-category scatter, §2.3)
    plot_figure1(df)

    # Stage 4 - Descriptives (Table 6)
    run_descriptives(df)

    # Stage 4b - Coverage heatmap (Figure 4)
    plot_coverage_heatmap(df)

    # Stage 5 - Univariate OLS (Table 7, Figure 5, optional Figure 5b)
    ols_uni, _ols_uni_global = run_univariate_stage(df)

    # Stage 6 - Multivariate OLS (Table 8, Figure 6)
    ols_multi = run_multivariate_stage(df)

    # Stage 7 - ML imputation (Tables 9-10, Figure 7a/b)
    predictions, meta = run_ml_stage(df)

    # Stage 7b - Appendix B and C tables (Table C.1 predictions summary, Table B.1 hyperparameters)
    run_appendix_stage(predictions)

    # Stage 8 - Interpretive report
    write_interpretive_report(ols_uni, ols_multi, predictions, meta)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*78}\nFINAL SUMMARY\n{'='*78}")
    print(f"Runtime: {time.time() - t0:.1f} seconds")
    print("\nFiles produced (in the current working directory):")
    for f in sorted(Path.cwd().glob("*.csv")):
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")
    print("\nDone.")


if __name__ == "__main__":
    main()