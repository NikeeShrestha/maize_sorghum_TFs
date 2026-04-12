#!/usr/bin/env python3
"""
Ortholog Pair Activity Visualization — Combined (Action + RecategorizeAction)
==============================================================================
For each ortholog pair (Sorghum–Maize), classifies the pair into one of
four activity categories:

  1. Both Active     – both Sorghum and Maize are Activator
  2. Both Inactive   – neither is Activator
  3. Sb Active / Zm Inactive – Sorghum active, Maize inactive
  4. Zm Active / Sb Inactive – Maize active, Sorghum inactive

Runs for both the 'Action' and 'RecategorizeAction' columns.

Produces (per column):
  - results/ortholog_pair_activity_proportions.csv            (Action)
  - results/ortholog_pair_activity_stacked.png                (Action)
  - results/ortholog_pair_activity_recategorized_proportions.csv  (RecategorizeAction)
  - results/ortholog_pair_activity_recategorized_stacked.png     (RecategorizeAction)

Notes:
  - Only ortholog pairs where BOTH genes have activity data are included.
  - Many-to-many orthologs are all included as separate pairs.
  - RecategorizeAction plot uses the same family order as Action (sorted by
    total pairs descending in the Action dataset) for direct comparison.
"""

import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Nimbus Sans",
    "font.size": 18,
})

# ---- Configuration ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SORGHUM_FILE  = os.path.join(BASE_DIR, "sorghumdata", "Sorghum_ActivityAnnotated.csv")
MAIZE_FILE    = os.path.join(BASE_DIR, "maizedata",   "maize_ActivityAnnotated.csv")
ORTHOLOG_FILE = os.path.join(BASE_DIR, "input",     "sorghumversion3_maizeversion3.csv")

CATEGORIES = ["Both Active", "Both Inactive", "Sb Active / Zm Inactive", "Zm Active / Sb Inactive"]
COLORS     = ["#4CAF7D", "#E05C5C", "#5B9BD5", "#F4A84A"]

ACTIVITY_CONFIGS = [
    {
        "action_col": "Action",
        "label":      "Action",
        "out_csv":    os.path.join(BASE_DIR, "results", "ortholog_pair_activity_proportions.csv"),
        "out_png":    os.path.join(BASE_DIR, "results", "ortholog_pair_activity_stacked.png"),
    },
    {
        "action_col": "RecategorizeAction",
        "label":      "RecategorizeAction",
        "out_csv":    os.path.join(BASE_DIR, "results", "ortholog_pair_activity_recategorized_proportions.csv"),
        "out_png":    os.path.join(BASE_DIR, "results", "ortholog_pair_activity_recategorized_stacked.png"),
    },
]


# ---- Helper functions --------------------------------------------------------

def classify_pair(sb_active, zm_active):
    """Classify an ortholog pair into one of four categories."""
    if sb_active and zm_active:
        return "Both Active"
    elif not sb_active and not zm_active:
        return "Both Inactive"
    elif sb_active and not zm_active:
        return "Sb Active / Zm Inactive"
    else:
        return "Zm Active / Sb Inactive"


def build_pairs(ortho, sorghum_cols, maize_cols, action_col):
    """Merge ortholog table with activity annotations for both species."""
    pairs = ortho.merge(
        sorghum_cols.rename(columns={
            "gene_ID": "SorghumGene",
            action_col: "Sb_Action",
            "family": "family",
        }),
        on="SorghumGene", how="inner",
    )
    maize_for_merge = maize_cols.rename(columns={
        "old_version3_gene_ID": "Ortholog",
        action_col: "Zm_Action",
    })[["Ortholog", "gene_ID", "Zm_Action"]]
    pairs = pairs.merge(maize_for_merge, on="Ortholog", how="inner")
    pairs = pairs.dropna(subset=["Sb_Action", "Zm_Action"])
    pairs["Category"] = [
        classify_pair(sb == "Activator", zm == "Activator")
        for sb, zm in zip(pairs["Sb_Action"], pairs["Zm_Action"])
    ]
    return pairs


def compute_stats(pairs):
    """Compute per-family and overall counts/proportions."""
    family_counts = (
        pairs.groupby(["family", "Category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CATEGORIES, fill_value=0)
    )
    family_counts["Total"] = family_counts.sum(axis=1)
    family_props = family_counts[CATEGORIES].div(
        family_counts["Total"].replace(0, float("nan")), axis=0
    ).fillna(0)

    overall_counts = pairs["Category"].value_counts().reindex(CATEGORIES, fill_value=0)
    overall_total  = overall_counts.sum()
    overall_props  = overall_counts / overall_total

    return family_counts, family_props, overall_counts, overall_total, overall_props


def save_csv(family_counts, family_props, out_csv):
    """Save per-family proportions to CSV."""
    out_df = family_counts.copy()
    for cat in CATEGORIES:
        out_df[f"Prop_{cat.replace(' ', '_').replace('/', '')}"] = family_props[cat].round(4)
    out_df.reset_index(inplace=True)
    out_df.to_csv(out_csv, index=False)


def make_stacked_bar(family_props, family_counts, overall_props, overall_counts,
                     family_order, out_png):
    """Create and save the stacked bar chart."""
    family_props_sorted  = family_props.reindex(family_order, fill_value=0)
    family_counts_sorted = family_counts.reindex(family_order, fill_value=0)

    # Prepend "Overall" row
    overall_row        = pd.DataFrame([overall_props.values],  columns=CATEGORIES, index=["Overall"])
    overall_counts_row = pd.DataFrame([overall_counts.values], columns=CATEGORIES, index=["Overall"])
    plot_data   = pd.concat([overall_row,        family_props_sorted])
    plot_counts = pd.concat([overall_counts_row, family_counts_sorted[CATEGORIES]])

    n_bars = len(plot_data)
    fig, ax = plt.subplots(figsize=(max(16, n_bars * 0.6), 8))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    x = np.arange(n_bars)
    bar_width = 0.72

    bottoms = np.zeros(n_bars)
    for cat, color in zip(CATEGORIES, COLORS):
        vals   = plot_data[cat].values.astype(float)
        counts = plot_counts[cat].values.astype(int)
        ax.bar(x, vals, bottom=bottoms, color=color, width=bar_width,
               label=cat, edgecolor="white", linewidth=0.6)
        for xi, (v, b, n) in enumerate(zip(vals, bottoms, counts)):
            if n > 0:
                ax.text(xi, b + v / 2, str(n), ha="center", va="center",
                        fontsize=18, color="white", fontweight="bold", rotation=90)
        bottoms += vals

    # Total count on top of each bar
    totals = plot_counts[CATEGORIES].sum(axis=1).values.astype(int)
    for xi, total in enumerate(totals):
        ax.text(xi, 1.02, str(total), ha="center", va="bottom",
                fontsize=14, color="black", fontweight="bold")

    # Separator between Overall and families
    ax.axvline(x=0.5, color="black", linewidth=1.4, linestyle="--", alpha=0.7)

    # X-axis labels
    labels = plot_data.index.tolist()
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=50, ha="right")
    ax.set_xlim(-0.5, n_bars - 0.5)

    ax.set_ylabel("Percentage of Ortholog Pairs", fontsize=24, labelpad=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(COLORS, CATEGORIES)]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(CATEGORIES),
        framealpha=0.0,
        edgecolor="none",
        fontsize=24,
        handlelength=1.4,
        borderpad=0.5,
        columnspacing=1.2,
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    for spine in ax.spines.values():
        spine.set_linewidth(4)
    ax.tick_params(width=4, length=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# ---- Load data once ----------------------------------------------------------
sorghum_full = pd.read_csv(SORGHUM_FILE)
maize_full   = pd.read_csv(MAIZE_FILE)
ortho        = pd.read_csv(ORTHOLOG_FILE)

# ---- Derive canonical family order from Action pairs (used by both plots) ----
_action_pairs = build_pairs(
    ortho,
    sorghum_full[["gene_ID", "family", "Action"]],
    maize_full[["gene_ID", "old_version3_gene_ID", "family", "Action"]],
    "Action",
)
canonical_family_order = (
    _action_pairs.groupby("family").size()
    .sort_values(ascending=False)
    .index.tolist()
)

# ==============================================================================
# Loop over both activity columns
# ==============================================================================
for cfg in ACTIVITY_CONFIGS:
    action_col = cfg["action_col"]
    label      = cfg["label"]
    out_csv    = cfg["out_csv"]
    out_png    = cfg["out_png"]

    print("=" * 72)
    print(f"  Activity column: {action_col}")
    print("=" * 72)

    # Build pairs
    sb_cols = sorghum_full[["gene_ID", "family", action_col]]
    zm_cols = maize_full[["gene_ID", "old_version3_gene_ID", "family", action_col]]
    pairs = build_pairs(ortho, sb_cols, zm_cols, action_col)

    print(f"  Total ortholog pairs with data: {len(pairs)}")
    print(f"  Unique families: {pairs['family'].nunique()}")

    # Compute stats
    family_counts, family_props, overall_counts, overall_total, overall_props = compute_stats(pairs)

    # For RecategorizeAction, reindex to canonical order so families match
    family_counts = family_counts.reindex(canonical_family_order, fill_value=0)
    family_counts["Total"] = family_counts[CATEGORIES].sum(axis=1)
    family_props = family_counts[CATEGORIES].div(
        family_counts["Total"].replace(0, float("nan")), axis=0
    ).fillna(0)

    # Print overall
    print(f"\n  === Overall Proportions ===")
    for cat in CATEGORIES:
        n = overall_counts[cat]
        p = overall_props[cat]
        print(f"    {cat:<30s}: {n:5d} pairs  ({p:.1%})")

    # Save CSV
    save_csv(family_counts, family_props, out_csv)
    print(f"\n  Per-family table saved to: {out_csv}")

    # Plot
    make_stacked_bar(family_props, family_counts, overall_props, overall_counts,
                     canonical_family_order, out_png)
    print(f"  Plot saved to: {out_png}\n")
