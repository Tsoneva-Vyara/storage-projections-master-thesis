"""
plotting.py - shared plot helpers.

Contains the global matplotlib rcParams and the dataframe_to_png helper, which is
used by every stage that produces a table for the thesis.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Global matplotlib defaults ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.titleweight":"bold",
    "figure.dpi":      150,
})


def dataframe_to_png(df_in, path, title=None, footnote=None,
                     col_widths=None, header_shade="#D9E1F2",
                     min_col_chars=6, char_inch=0.11, max_fig_width=22.0):
    """
    Generates a DataFrame as a PNG table.

    The header row is shaded, every cell has a thin black border, and the
    optional footnote sits underneath.  Column widths scale to the longest
    string in each column (header or value), so long-text columns get the
    space they need and short-text columns stay compact.  The overall figure
    width scales to fit the total text length, capped at ``max_fig_width``
    inches so a page can hold it.

    Also writes a companion PDF at ``path.replace('.png', '.pdf')``.
    """
    import pandas as pd  # local import: keeps plotting.py light

    df_render = df_in.copy().astype(str).fillna("")
    n_rows, n_cols = df_render.shape

    # Per-column longest string (header + all values)
    col_char_len = np.array([
        max(len(c),
            max((len(x) for x in df_render[c].values), default=0),
            min_col_chars)
        for c in df_render.columns
    ], dtype=float)
    frac_widths = col_char_len / col_char_len.sum()
    fig_width = min(max_fig_width,
                    max(6.5, col_char_len.sum() * char_inch))
    row_h = 0.42
    fig_height = row_h * (n_rows + 1) + 0.9
    if title:    fig_height += 0.35
    if footnote: fig_height += 0.30

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_axis_off()

    if title:
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold",
                     pad=6)

    tbl = ax.table(
        cellText=df_render.values.tolist(),
        colLabels=list(df_render.columns),
        colWidths=frac_widths.tolist(),
        loc="upper left",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#333")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(header_shade)
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")

    if footnote:
        fig.text(0.01, 0.02, footnote, fontsize=8, style="italic",
                 color="#333")

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)