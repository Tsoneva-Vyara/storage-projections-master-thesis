"""
figure1_projections.py - Stage 3b: Figure 1.

Total energy storage capacity projections for Europe by horizon year,
color-and-shape-coded by source category (Institutional, Industry,
Network-based, Academic).  Referenced in §3.1 of the thesis.

Scope filter:  region ∈ {EU-27, EU-28, Europe} (analysis_group == "EU/Europe")
                AND storage_gw is reported (non-null).

The n, which is reported in the title, is the number of surviving scenario-observations
after the filter and is printed to stdout during the run.
"""
import matplotlib.pyplot as plt

from config import SOURCE_CATEGORY_STYLE


# Fixed legend order - matches the target visualization in the thesis.
_CATEGORY_ORDER = ["Institutional", "Industry", "Network-based", "Academic"]


def plot_figure1(df):
    """Write ``fig1_projections_by_source.{png,pdf}``.

    Filters to EU/Europe scope (EU-27, EU-28, broader Europe) and to rows
    that report a total storage capacity value.  Each source category
    gets its own marker style and color so the four communities of
    projection producers can be visually separated at a glance.
    """
    print(f"\n{'='*78}\nSTAGE 3b - Figure 1 (projections by source category)\n{'='*78}")

    sub = df[(df["analysis_group"] == "EU/Europe")
             & df["storage_gw"].notna()].copy()
    n = len(sub)
    print(f"  EU/Europe rows with reported total storage: n = {n}")
    if n == 0:
        print("  Nothing to plot - skipped.")
        return

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    # One scatter per category
    # + consistent color combination.
    for cat in _CATEGORY_ORDER:
        mask = (sub["source_category"] == cat)
        if mask.sum() == 0:
            continue
        style = SOURCE_CATEGORY_STYLE[cat]
        ax.scatter(
            sub.loc[mask, "year"],
            sub.loc[mask, "storage_gw"],
            marker=style["marker"], color=style["color"],
            s=95, alpha=0.90,
            edgecolors="black", linewidths=0.9,
            label=cat, zorder=3,
        )

    # Anything not in the four canonical categories is drawn in grey so it
    # is visible in the plot but not silently omitted; a footnote flags it.
    other = sub[~sub["source_category"].isin(_CATEGORY_ORDER)]
    if len(other) > 0:
        ax.scatter(other["year"], other["storage_gw"],
                   marker="x", color="#888", s=70, alpha=0.7,
                   label=f"Unclassified (n = {len(other)})", zorder=3)

    ax.set_xlabel("Horizon year", fontsize=11)
    ax.set_ylabel("Total storage capacity (GW)", fontsize=11)
    ax.set_title(
        "European energy storage capacity projections by source category\n"
        f"(EU-27, EU-28, and broader Europe only; n = {n} scenario observations)",
        fontsize=11, pad=8,
    )
    ax.grid(True, alpha=0.35, ls=":")
    ax.legend(title="Source category", loc="upper left",
              fontsize=9, title_fontsize=9, framealpha=0.95)
    # A small headroom above the tallest point keeps the marker off the frame.
    y_max = sub["storage_gw"].max()
    ax.set_ylim(0, y_max * 1.10)

    plt.tight_layout()
    plt.savefig(f"fig1_projections_by_source.png",
                bbox_inches="tight", dpi=200)
    plt.savefig(f"fig1_projections_by_source.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → fig1_projections_by_source.png / .pdf")