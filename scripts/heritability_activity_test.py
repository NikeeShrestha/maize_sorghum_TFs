#!/usr/bin/env python3
"""
Heritability vs Activity: Mann-Whitney U Test + Boxplots
=========================================================
Tests whether Activator genes have significantly different heritability
compared to non-Activator genes, for both Maize and Sorghum,
using both the Action and RecategorizeAction columns.

Outputs:
  results/heritability_activity_mannwhitney.csv  – test statistics
  results/heritability_activity_boxplot.png      – figure
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SORGHUM_ACT   = os.path.join(BASE_DIR, "sorghumdata", "Sorghum_ActivityAnnotated.csv")
MAIZE_ACT     = os.path.join(BASE_DIR, "maizedata",   "maize_ActivityAnnotated.csv")
SORGHUM_VAR   = os.path.join(BASE_DIR, "sorghumdata", "sorghumgene_variance.csv")
MAIZE_VAR     = os.path.join(BASE_DIR, "maizedata",   "maize_693_genemodels_variance.csv")
OUT_CSV       = os.path.join(BASE_DIR, "results",     "heritability_activity_mannwhitney.csv")
OUT_FIG       = os.path.join(BASE_DIR, "results",     "heritability_activity_boxplot.png")

# ── Load data ────────────────────────────────────────────────────────────────
sb_act = pd.read_csv(SORGHUM_ACT, usecols=["gene_ID", "Action", "RecategorizeAction"])
zm_act = pd.read_csv(MAIZE_ACT,   usecols=["gene_ID", "Action", "RecategorizeAction"])
sb_var = pd.read_csv(SORGHUM_VAR, usecols=["Trait", "Heritability"])
zm_var = pd.read_csv(MAIZE_VAR,   usecols=["Trait", "Heritability"])

# Rename 'Trait' → 'gene_ID' for merging
sb_var = sb_var.rename(columns={"Trait": "gene_ID"})
zm_var = zm_var.rename(columns={"Trait": "gene_ID"})

# ── Merge activity + heritability ─────────────────────────────────────────────
sb = sb_act.merge(sb_var, on="gene_ID", how="inner").dropna(subset=["Heritability"])
zm = zm_act.merge(zm_var, on="gene_ID", how="inner").dropna(subset=["Heritability"])

print(f"Sorghum genes with heritability + activity: {len(sb):,}")
print(f"Maize   genes with heritability + activity: {len(zm):,}\n")

# ── Mann-Whitney test helper ──────────────────────────────────────────────────
def run_mw(df, action_col, species):
    """Split into Activator vs Non-Activator and run Mann-Whitney U."""
    valid = df.dropna(subset=[action_col])
    act     = valid.loc[valid[action_col] == "Activator", "Heritability"]
    non_act = valid.loc[valid[action_col] != "Activator", "Heritability"]
    if len(act) < 5 or len(non_act) < 5:
        return None
    stat, p = mannwhitneyu(act, non_act, alternative="two-sided")
    return {
        "Species":    species,
        "Action_col": action_col,
        "N_Activator":     int(len(act)),
        "N_NonActivator":  int(len(non_act)),
        "Median_Activator":    round(act.median(),    4),
        "Median_NonActivator": round(non_act.median(), 4),
        "Mean_Activator":    round(act.mean(),    4),
        "Mean_NonActivator": round(non_act.mean(), 4),
        "MannWhitney_U":  round(stat, 2),
        "P_value":        p,
        "Significant_05": p < 0.05,
    }

results = []
for action_col in ["Action", "RecategorizeAction"]:
    results.append(run_mw(sb, action_col, "Sorghum"))
    results.append(run_mw(zm, action_col, "Maize"))

res_df = pd.DataFrame([r for r in results if r])
res_df.to_csv(OUT_CSV, index=False)
print("── Mann-Whitney Results ──────────────────────────────────────")
print(res_df[["Species","Action_col","N_Activator","N_NonActivator",
              "Median_Activator","Median_NonActivator","P_value","Significant_05"]].to_string(index=False))
print(f"\n→ Saved: {OUT_CSV}")

# ── Boxplots ──────────────────────────────────────────────────────────────────
ACTION_COLS  = ["Action", "RecategorizeAction"]
COL_LABELS   = {"Action": "Action", "RecategorizeAction": "Recategorize Action"}
SPECIES_DATA = {"Sorghum": sb, "Maize": zm}

COLOR_ACT  = "#D62728"   # red for Activator
COLOR_NACT = "#1F77B4"   # blue for Non-Activator

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

for col_idx, action_col in enumerate(ACTION_COLS):
    for row_idx, (species, df) in enumerate(SPECIES_DATA.items()):
        ax = axes[row_idx][col_idx]

        valid = df.dropna(subset=[action_col])
        act_h     = valid.loc[valid[action_col] == "Activator",  "Heritability"].values
        nonact_h  = valid.loc[valid[action_col] != "Activator",  "Heritability"].values

        data   = [act_h, nonact_h]
        labels = [f"Activator\n(n={len(act_h):,})", f"Non-Activator\n(n={len(nonact_h):,})"]
        colors = [COLOR_ACT, COLOR_NACT]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.45,
            medianprops=dict(color="black", linewidth=2.5),
            whiskerprops=dict(color="#555555", linewidth=1.2),
            capprops=dict(color="#555555", linewidth=1.2),
            flierprops=dict(marker="o", markersize=2, alpha=0.4,
                            markerfacecolor="#888888", markeredgecolor="none"),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.70)

        # Significance annotation
        row = res_df[(res_df["Species"] == species) & (res_df["Action_col"] == action_col)]
        if not row.empty:
            p = row.iloc[0]["P_value"]
            if p < 0.001:
                sig_text = f"p = {p:.2e} ***"
            elif p < 0.01:
                sig_text = f"p = {p:.4f} **"
            elif p < 0.05:
                sig_text = f"p = {p:.4f} *"
            else:
                sig_text = f"p = {p:.4f} ns"
            ax.text(0.5, 0.98, sig_text, ha="center", va="top",
                    fontsize=10, color="black", fontweight="bold",
                    transform=ax.transAxes)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels, fontsize=9, color="black")
        ax.set_ylabel("Heritability (H²)", fontsize=10, color="black")
        ax.set_title(f"{species} — {COL_LABELS[action_col]}", fontsize=12,
                     fontweight="bold", color="black", pad=8)
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_edgecolor("#AAAAAA")

        # # Medians as text
        # for xi, (vals, col) in enumerate(zip(data, colors), start=1):
        #     if len(vals) > 0:
        #         med = np.median(vals)
        #         ax.text(1,0.8, f"med={med:.3f}", ha="center", va="top",
        #                 fontsize=8, color=col, transform=ax.get_xaxis_transform())

# Legend
patch_act  = mpatches.Patch(color=COLOR_ACT,  alpha=0.85, label="Activator")
patch_nact = mpatches.Patch(color=COLOR_NACT, alpha=0.85, label="Non-Activator")
fig.legend(handles=[patch_act, patch_nact], loc="upper center",
           ncol=2, fontsize=11, frameon=False,
           bbox_to_anchor=(0.5, 1.01))

fig.suptitle("Heritability: Activator vs Non-Activator Genes\n(Mann-Whitney U Test)",
             fontsize=15, fontweight="bold", color="black", y=1.04)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"\n→ Figure saved: {OUT_FIG}")
print("Done.")
