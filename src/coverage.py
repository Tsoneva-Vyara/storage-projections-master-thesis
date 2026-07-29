"""
coverage.py - Stage 4b: Figure 3 variable-coverage heatmap.

Renders a two-column heatmap (EU/Europe vs Global), which shows, for every
variable, the row count and the corresponding percentage of the subgroup that reports
it.  Color scales on the percentage, so a small subgroup with a good
coverage rate still reads as strong.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import HEATMAP_VARS


def plot_coverage_heatmap(df):
    """Write ``fig3_variable_coverage.{png,pdf}``."""
    print(f"\n{'='*78}\nSTAGE 4b - Variable coverage heatmap (Figure 3)\n{'='*78}")

    n_eu     = int((df["analysis_group"] == "EU/Europe").sum())
    n_global = int((df["analysis_group"] == "Global").sum())

    rows = []
    for col, label in HEATMAP_VARS:
        if col not in df.columns:
            rows.append({"label": label, "eu_n": 0, "gl_n": 0,
                         "eu_pct": 0.0, "gl_pct": 0.0})
            continue
        eu_n = int(df.loc[df["analysis_group"] == "EU/Europe", col].notna().sum())
        gl_n = int(df.loc[df["analysis_group"] == "Global",    col].notna().sum())
        rows.append({
            "label":  label,
            "eu_n":   eu_n,
            "gl_n":   gl_n,
            "eu_pct": 100.0 * eu_n / max(n_eu, 1),
            "gl_pct": 100.0 * gl_n / max(n_global, 1),
        })

    hm = pd.DataFrame(rows)
    M = np.column_stack([hm["eu_pct"].values, hm["gl_pct"].values])

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"EU/Europe\n(n = {n_eu})",
                        f"Global\n(n = {n_global})"], fontsize=10)
    ax.set_yticks(np.arange(len(hm)))
    ax.set_yticklabels(hm["label"].values, fontsize=9)

    for i, row in hm.reset_index(drop=True).iterrows():
        for j, n_key, p_key in [(0, "eu_n", "eu_pct"),
                                (1, "gl_n", "gl_pct")]:
            val_n = int(row[n_key])
            val_p = row[p_key]
            colour = "black" if M[i, j] < 55 else "white"
            ax.text(j, i, f"{val_n}\n({val_p:.0f}%)",
                    ha="center", va="center", fontsize=9, color=colour)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Share of the subgroup that reports the variable (%)",
                   fontsize=9)

    ax.set_title("Figure 3: Variable coverage by analytical subgroup", pad=10)
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(f"fig3_variable_coverage.png",
                bbox_inches="tight", dpi=220)
    plt.savefig(f"fig3_variable_coverage.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → fig3_variable_coverage.png / .pdf")