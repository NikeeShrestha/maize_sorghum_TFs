#!/usr/bin/env python3
"""
Ortholog Pair Activity Visualization
======================================
For each ortholog pair (Sorghum–Maize), classifies the pair into one of
four activity categories based on the 'Action' column:

  1. Both Active     – both Sorghum and Maize have activation domain (Activator)
  2. Both Inactive   – both have no activation domain
  3. Sb Active / Zm Inactive – Sorghum active, Maize inactive
  4. Zm Active / Sb Inactive – Maize active, Sorghum inactive

Produces:
  - A stacked bar chart of proportions per gene family
  - An overall (all families pooled) bar chart
  - results/ortholog_pair_activity_proportions.csv  (per-family counts & proportions)

Notes:
  - Uses the 'Action' column (Activator = active, No activity = inactive).
  - Only ortholog pairs where BOTH genes have activity data are included.
  - Many-to-many orthologs are all included as separate pairs.

Output:
  results/ortholog_pair_activity_proportions.csv
  results/ortholog_pair_activity_stacked.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---- Configuration ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SORGHUM_FILE  = os.path.join(BASE_DIR, "sorghumdata", "Sorghum_ActivityAnnotated.csv")
MAIZE_FILE    = os.path.join(BASE_DIR, "maizedata",   "maize_ActivityAnnotated.csv")
ORTHOLOG_FILE = os.path.join(BASE_DIR, "results",     "sorghumversion3ortholog.csv")
OUT_CSV       = os.path.join(BASE_DIR, "results",     "ortholog_pair_activity_proportions.csv")
OUT_PNG       = os.path.join(BASE_DIR, "results",     "ortholog_pair_activity_stacked.png")

# ---- Load data ----
sorghum = pd.read_csv(SORGHUM_FILE)[["gene_ID", "family", "Action"]]
maize   = pd.read_csv(MAIZE_FILE)[["gene_ID", "family", "Action"]]
ortho   = pd.read_csv(ORTHOLOG_FILE)

# ---- Build ortholog-pair table ----
# Merge sorghum info
pairs = ortho.merge(
    sorghum.rename(columns={"gene_ID": "SorghumGene", "Action": "Sb_Action", "family": "family"}),
    on="SorghumGene", how="inner"
)
# Merge maize info
pairs = pairs.merge(
    maize.rename(columns={"gene_ID": "Ortholog", "Action": "Zm_Action"})[["Ortholog", "Zm_Action"]],
    on="Ortholog", how="inner"
)

# Drop pairs where either action is missing
pairs = pairs.dropna(subset=["Sb_Action", "Zm_Action"])

print(f"Total ortholog pairs with data: {len(pairs)}")
print(f"Unique families: {pairs['family'].nunique()}")

# ---- Classify each pair ----
def classify_pair(row):
    sb = row["Sb_Action"] == "Activator"
    zm = row["Zm_Action"] == "Activator"
    if sb and zm:
        return "Both Active"
    elif not sb and not zm:
        return "Both Inactive"
    elif sb and not zm:
        return "Sb Active / Zm Inactive"
    else:
        return "Zm Active / Sb Inactive"

pairs["Category"] = pairs.apply(classify_pair, axis=1)

CATEGORIES = ["Both Active", "Both Inactive", "Sb Active / Zm Inactive", "Zm Active / Sb Inactive"]
COLORS = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]

# ---- Per-family counts ----
family_counts = (
    pairs.groupby(["family", "Category"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=CATEGORIES, fill_value=0)
)

family_counts["Total"] = family_counts.sum(axis=1)

# Proportions
family_props = family_counts[CATEGORIES].div(family_counts["Total"], axis=0)

# ---- Overall (pooled) ----
overall_counts = pairs["Category"].value_counts().reindex(CATEGORIES, fill_value=0)
overall_total  = overall_counts.sum()
overall_props  = overall_counts / overall_total

print("\n=== Overall Proportions (all families) ===")
for cat in CATEGORIES:
    n = overall_counts[cat]
    p = overall_props[cat]
    print(f"  {cat:<30s}: {n:5d} pairs  ({p:.1%})")

# ---- Save CSV ----
out_df = family_counts.copy()
for cat in CATEGORIES:
    out_df[f"Prop_{cat.replace(' ', '_').replace('/', '')}"] = family_props[cat].round(4)
out_df.reset_index(inplace=True)
out_df.to_csv(OUT_CSV, index=False)
print(f"\nPer-family table saved to: {OUT_CSV}")

# ============================================================
# ---- Plot: stacked bar per family + overall ----
# ============================================================
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Nimbus Sans",
    'font.size':18
})

# Softer, accessible color palette
COLORS = ["#4CAF7D", "#E05C5C", "#5B9BD5", "#F4A84A"]

# Sort families by total pairs (descending) for readability
family_order         = family_counts["Total"].sort_values(ascending=False).index.tolist()
family_props_sorted  = family_props.loc[family_order]
family_counts_sorted = family_counts.loc[family_order]

# Prepend "Overall" row (proportions + counts)
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
    # Count label rotated vertically; only show when count > 0
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

# Subtle horizontal gridlines behind bars
# ax.set_axisbelow(True)
# ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#CCCCCC", alpha=0.8)

# Separator between Overall and families
ax.axvline(x=0.5, color="black", linewidth=1.4, linestyle="--", alpha=0.7)
# ax.text(0.51, 0.99, "Per family →", transform=ax.get_xaxis_transform(),
#         va="top", ha="left", fontsize=9.5, color="#666666", style="italic")

# X-axis labels
labels = plot_data.index.tolist()
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=50, ha="right")
ax.set_xlim(-0.5, n_bars - 0.5)



ax.set_ylabel("Percentage of Ortholog Pairs", fontsize=24, labelpad=10)
# ax.set_title(
#     "Ortholog Pair Activity States per Gene Family",
#     fontsize=16, fontweight="bold", pad=16
# )
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(COLORS, CATEGORIES)]
leg = ax.legend(
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
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close()
print(f"Plot saved to: {OUT_PNG}")
