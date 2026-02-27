#!/usr/bin/env python3
"""
Activator vs Non-Activity: Count of Regulated Genes Test
==========================================================
Tests whether maize TFs classified as "Activator" regulate significantly
more genes than TFs with "No activity", using a one-sided Mann-Whitney U test.

Run for both 'Action' and 'RecategorizeAction' columns.

Output:
  results/activator_count_genes_test.csv
  results/activator_count_genes_boxplot.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import mannwhitneyu

# ---- Configuration ----
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTIVITY_FILE = os.path.join(BASE_DIR, "maizedata", "maize_ActivityAnnotated.csv")
COUNT_FILE    = os.path.join(BASE_DIR, "maizedata", "maize_gene_count_2pcs_0.01.csv")
OUT_CSV       = os.path.join(BASE_DIR, "results", "activator_count_genes_test.csv")
OUT_PNG       = os.path.join(BASE_DIR, "results", "activator_count_genes_boxplot.png")

# ---- Load & merge ----
activity = pd.read_csv(ACTIVITY_FILE)[["gene_ID", "Action", "RecategorizeAction"]]
counts   = pd.read_csv(COUNT_FILE)   # Gene, Count
df = activity.merge(counts, left_on="gene_ID", right_on="Gene", how="inner")
print(f"Genes with both activity and count data: {len(df)}\n")

# ---- Run test for each action column ----
rows = []
for col in ["Action", "RecategorizeAction"]:
    sub      = df.dropna(subset=[col])
    active   = sub.loc[sub[col] == "Activator", "Count"]
    inactive = sub.loc[sub[col] != "Activator", "Count"]
    U, p     = mannwhitneyu(active, inactive, alternative="greater")
    r        = 1 - (2 * U) / (len(active) * len(inactive))  # rank-biserial r

    rows.append({
        "Column":            col,
        "N_Activator":       len(active),
        "N_NoActivity":      len(inactive),
        "Mean_Activator":    round(active.mean(), 2),
        "Mean_NoActivity":   round(inactive.mean(), 2),
        "Median_Activator":  round(active.median(), 1),
        "Median_NoActivity": round(inactive.median(), 1),
        "MannWhitneyU":      round(U, 1),
        "P_value":           p,
        "RankBiserial_r":    round(r, 4),
    })
    print(f"=== {col} ===")
    print(f"  Activator  : n={len(active)}, median={active.median():.0f}, mean={active.mean():.1f}")
    print(f"  No activity: n={len(inactive)}, median={inactive.median():.0f}, mean={inactive.mean():.1f}")
    print(f"  Mann-Whitney U={U:.0f}, p={p:.4e}, rank-biserial r={r:.4f}\n")

result_df = pd.DataFrame(rows)
result_df.to_csv(OUT_CSV, index=False)
print(f"Results saved to: {OUT_CSV}")

# ============================================================
# ---- Plot: boxplots for Action and RecategorizeAction ----
# ============================================================
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Nimbus Sans",
    "font.size": 16,
})

COLORS = ["#4CAF7D", "#E05C5C"]
LABELS = ["Activator", "No activity"]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
# fig.patch.set_facecolor("#FAFAFA")

for ax, col, title_col in zip(
    axes,
    ["Action", "RecategorizeAction"],
    [r"\textbf{Action}", r"\textbf{RecategorizeAction}"]
):
    sub      = df.dropna(subset=[col])
    active   = sub.loc[sub[col] == "Activator", "Count"].values
    inactive = sub.loc[sub[col] != "Activator", "Count"].values
    data     = [active, inactive]

    # ax.set_facecolor("#FAFAFA")
    bp = ax.boxplot(
        data, patch_artist=True, widths=0.5,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=3, alpha=0.4, linestyle="none"),
    )
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    # Jittered individual points
    # for i, (d, color) in enumerate(zip(data, COLORS), start=1):
    #     jitter = np.random.uniform(-0.15, 0.15, size=len(d))
    #     ax.scatter(i + jitter, d, alpha=0.2, s=10, color=color, zorder=3)

    # Median labels
    for i, d in enumerate(data, start=1):
        ax.text(i, np.median(d) + 100, f"median={np.median(d):.0f}",
                ha="center", fontsize=11, color="black", fontweight="bold")

    # p-value from results
    row   = result_df[result_df["Column"] == col].iloc[0]
    p_str = fr"$p = {row['P_value']:.4f}$"
    ax.set_title(fr"{title_col}" + "\n" + p_str, fontsize=14)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(LABELS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(width=2, length=5)

    yticks=ax.get_yticks()
    ax.set_yticks(yticks[1:])
    ax.set_yticklabels([f"{int(y)}" for y in yticks[1:]])

axes[0].set_ylabel("Number of Regulated Genes", fontsize=16)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close()
print(f"Plot saved to: {OUT_PNG}")
