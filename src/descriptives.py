"""
descriptives.py - Stage 4: Table 6 descriptive statistics.

Reports N, mean, median, std, min, max for each variable in the EU/Europe
subgroup.  Renders both a CSV and a PNG that can be pasted into the thesis.
"""
import pandas as pd

from config import DESC_VARS
from plotting import dataframe_to_png


def run_descriptives(df):
    """Write ``table6_descriptive_stats.{csv,png,pdf}``."""
    print(f"\n{'='*78}\nSTAGE 4 - Descriptive statistics\n{'='*78}")

    sub = df[df["analysis_group"] == "EU/Europe"]
    rows = []
    for col, label in DESC_VARS:
        if col not in sub.columns:
            continue
        v = sub[col].dropna()
        if len(v) == 0:
            continue
        rows.append({
            "Variable": label,
            "N":        int(len(v)),
            "Mean":     f"{v.mean():.2f}",
            "Median":   f"{v.median():.2f}",
            "Std":      f"{v.std():.2f}",
            "Min":      f"{v.min():.2f}",
            "Max":      f"{v.max():.2f}",
        })

    tbl = pd.DataFrame(rows)
    csv_path = f"table6_descriptive_stats.csv"
    png_path = f"table6_descriptive_stats.png"
    tbl.to_csv(csv_path, index=False)
    dataframe_to_png(
        tbl, png_path,
        title="Table 6: Descriptive statistics of the cross-source dataset (EU/Europe scope)",
        footnote="N differs across variables because not every source reports every field.",
    )
    print(f"  Saved → {csv_path} / .png / .pdf")
    return tbl