#!/usr/bin/env python3
"""
Gene Family Summary (Wide Format) — based on Action column

For each TF family, produces a single-row summary with sorghum and maize
in separate columns. Activity classification uses the original 'Action' column.

Columns:
  - Sb_Total, Zm_Total, Zm_Sb_Ratio
  - Sb_Prop_No_Ortholog, Zm_Prop_No_Ortholog
  - Sb_Prop_Active, Zm_Prop_Active
  - Sb_Prop_NoOrtho_Active, Zm_Prop_NoOrtho_Active
  - Sb_N_NoOrtho_Active, Zm_N_NoOrtho_Active
  - Sb_Active_Prop_Ortholog, Zm_Active_Prop_Ortholog   (of active genes, proportion that are orthologs)
  - Sb_Active_Prop_NoOrtholog, Zm_Active_Prop_NoOrtholog (of active genes, proportion that are non-orthologs)

Output:
    - results/gene_family_summary_wide_sorghumversion3_action.csv
    - results/gene_family_summary_wide_counts_sorghumversion3_action.csv
    - results/gene_family_gene_lists_sorghumversion3_action.pkl
"""

import pandas as pd
import os
import pickle

# ---- Configuration ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SORGHUM_FILE = os.path.join(BASE_DIR, "sorghumdata", "Sorghum_ActivityAnnotated.csv")
MAIZE_FILE = os.path.join(BASE_DIR, "maizedata", "maize_ActivityAnnotated.csv")
ORTHOLOG_FILE = os.path.join(BASE_DIR, "results", "sorghumversion3ortholog.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "results", "gene_family_summary_wide_sorghumversion3_action.csv")

# ---- Load data ----
sorghum = pd.read_csv(SORGHUM_FILE)
maize = pd.read_csv(MAIZE_FILE)
ortho = pd.read_csv(ORTHOLOG_FILE)

sorghum_in_ortho = set(ortho["SorghumGene"])
maize_in_ortho = set(ortho["Ortholog"])

sorghum["Has_Ortholog"] = sorghum["gene_ID"].isin(sorghum_in_ortho)
maize["Has_Ortholog"] = maize["gene_ID"].isin(maize_in_ortho)

# Count orthologs per family (using sorghum family assignment)
ortho_with_family = ortho.merge(
    sorghum[["gene_ID", "family"]],
    left_on="SorghumGene", right_on="gene_ID", how="inner"
)
ortho_per_family = ortho_with_family.groupby("family").size().to_dict()

# Identify genes with ortholog but partner has no data
maize_genes_with_data = set(maize["gene_ID"])
sorghum_genes_with_data = set(sorghum["gene_ID"])

# Sorghum genes whose maize ortholog partner is NOT in maize activity data
ortho_sb_matched = set(ortho.loc[ortho["Ortholog"].isin(maize_genes_with_data), "SorghumGene"])
sb_ortho_no_partner = sorghum_in_ortho - ortho_sb_matched  # has ortholog, partner missing

# Maize genes whose sorghum ortholog partner is NOT in sorghum activity data
ortho_zm_matched = set(ortho.loc[ortho["SorghumGene"].isin(sorghum_genes_with_data), "Ortholog"])
zm_ortho_no_partner = maize_in_ortho - ortho_zm_matched

sorghum["Ortho_NoPartnerData"] = sorghum["gene_ID"].isin(sb_ortho_no_partner)
maize["Ortho_NoPartnerData"] = maize["gene_ID"].isin(zm_ortho_no_partner)

# ---- Build wide summary ----
all_families = sorted(set(sorghum["family"].unique()) | set(maize["family"].unique()))
results = []
gene_lists = {}  # {family: {category: [gene_ids]}}

for family in all_families:
    sb = sorghum[sorghum["family"] == family]
    zm = maize[maize["family"] == family]

    sb_total = len(sb)
    zm_total = len(zm)

    sb_no_ortho = (~sb["Has_Ortholog"]).sum()
    zm_no_ortho = (~zm["Has_Ortholog"]).sum()

    sb_active = (sb["Action"] == "Activator").sum()
    zm_active = (zm["Action"] == "Activator").sum()

    sb_no_ortho_active = ((~sb["Has_Ortholog"]) & (sb["Action"] == "Activator")).sum()
    zm_no_ortho_active = ((~zm["Has_Ortholog"]) & (zm["Action"] == "Activator")).sum()

    sb_ortho_nodata = sb["Ortho_NoPartnerData"].sum()
    zm_ortho_nodata = zm["Ortho_NoPartnerData"].sum()

    # Orthologs with data available (has ortholog AND partner has data)
    sb_ortho_with_data = (sb["Has_Ortholog"] & ~sb["Ortho_NoPartnerData"])
    zm_ortho_with_data = (zm["Has_Ortholog"] & ~zm["Ortho_NoPartnerData"])
    sb_ortho_data_count = sb_ortho_with_data.sum()
    zm_ortho_data_count = zm_ortho_with_data.sum()
    sb_ortho_active = (sb_ortho_with_data & (sb["Action"] == "Activator")).sum()
    zm_ortho_active = (zm_ortho_with_data & (zm["Action"] == "Activator")).sum()

    # Of active genes, proportion that are orthologs vs non-orthologs
    # (uses Has_Ortholog regardless of partner data availability)
    sb_ortho_active_any = (sb["Has_Ortholog"] & (sb["Action"] == "Activator")).sum()
    zm_ortho_active_any = (zm["Has_Ortholog"] & (zm["Action"] == "Activator")).sum()

    # Collect gene ID lists per category
    gene_lists[family] = {
        "Sb_All": sb["gene_ID"].tolist(),
        "Zm_All": zm["gene_ID"].tolist(),
        "Sb_No_Ortholog": sb.loc[~sb["Has_Ortholog"], "gene_ID"].tolist(),
        "Zm_No_Ortholog": zm.loc[~zm["Has_Ortholog"], "gene_ID"].tolist(),
        "Sb_Active": sb.loc[sb["Action"] == "Activator", "gene_ID"].tolist(),
        "Zm_Active": zm.loc[zm["Action"] == "Activator", "gene_ID"].tolist(),
        "Sb_NoOrtho_Active": sb.loc[(~sb["Has_Ortholog"]) & (sb["Action"] == "Activator"), "gene_ID"].tolist(),
        "Zm_NoOrtho_Active": zm.loc[(~zm["Has_Ortholog"]) & (zm["Action"] == "Activator"), "gene_ID"].tolist(),
        "Sb_Ortho_NoPartnerData": sb.loc[sb["Ortho_NoPartnerData"], "gene_ID"].tolist(),
        "Zm_Ortho_NoPartnerData": zm.loc[zm["Ortho_NoPartnerData"], "gene_ID"].tolist(),
        "Sb_Ortho_Active": sb.loc[sb_ortho_with_data & (sb["Action"] == "Activator"), "gene_ID"].tolist(),
        "Zm_Ortho_Active": zm.loc[zm_ortho_with_data & (zm["Action"] == "Activator"), "gene_ID"].tolist(),
    }

    results.append({
        "Family": family,
        "Sb_Total": sb_total,
        "Zm_Total": zm_total,
        "Zm_Sb_Ratio": round(zm_total / sb_total, 4) if sb_total > 0 else None,
        "Total_Orthologs": ortho_per_family.get(family, 0),
        "Sb_Prop_No_Ortholog": round(sb_no_ortho / sb_total, 4) if sb_total > 0 else None,
        "Zm_Prop_No_Ortholog": round(zm_no_ortho / zm_total, 4) if zm_total > 0 else None,
        "Sb_Prop_Active": round(sb_active / sb_total, 4) if sb_total > 0 else None,
        "Zm_Prop_Active": round(zm_active / zm_total, 4) if zm_total > 0 else None,
        "Sb_Prop_NoOrtho_Active": round(sb_no_ortho_active / sb_no_ortho, 4) if sb_no_ortho > 0 else None,
        "Zm_Prop_NoOrtho_Active": round(zm_no_ortho_active / zm_no_ortho, 4) if zm_no_ortho > 0 else None,
        "Sb_N_NoOrtho_Active": sb_no_ortho_active,
        "Zm_N_NoOrtho_Active": zm_no_ortho_active,
        "Sb_Ortho_NoPartnerData": sb_ortho_nodata,
        "Zm_Ortho_NoPartnerData": zm_ortho_nodata,
        "Sb_N_Ortho_Active": sb_ortho_active,
        "Zm_N_Ortho_Active": zm_ortho_active,
        "Sb_Prop_Ortho_Active": round(sb_ortho_active / sb_ortho_data_count, 4) if sb_ortho_data_count > 0 else None,
        "Zm_Prop_Ortho_Active": round(zm_ortho_active / zm_ortho_data_count, 4) if zm_ortho_data_count > 0 else None,
        # Of active genes: proportion that are orthologs vs non-orthologs
        "Sb_Active_Prop_Ortholog": round(sb_ortho_active_any / sb_active, 4) if sb_active > 0 else None,
        "Zm_Active_Prop_Ortholog": round(zm_ortho_active_any / zm_active, 4) if zm_active > 0 else None,
        "Sb_Active_Prop_NoOrtholog": round(sb_no_ortho_active / sb_active, 4) if sb_active > 0 else None,
        "Zm_Active_Prop_NoOrtholog": round(zm_no_ortho_active / zm_active, 4) if zm_active > 0 else None,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

# ---- Save counts-only version ----
OUTPUT_COUNTS = os.path.join(BASE_DIR, "results", "gene_family_summary_wide_counts_sorghumversion3_action.csv")
count_cols = ["Family", "Sb_Total", "Zm_Total", "Total_Orthologs",
              "Sb_No_Ortholog", "Zm_No_Ortholog",
              "Sb_Ortho_NoPartnerData", "Zm_Ortho_NoPartnerData",
              "Sb_Active", "Zm_Active",
              "Sb_N_NoOrtho_Active", "Zm_N_NoOrtho_Active",
              "Sb_N_Ortho_Active", "Zm_N_Ortho_Active",
              "Sb_Active_Prop_Ortholog", "Zm_Active_Prop_Ortholog",
              "Sb_Active_Prop_NoOrtholog", "Zm_Active_Prop_NoOrtholog"]

# Add the missing count columns to results_df
for family_row in results:
    family_row_name = family_row["Family"]
    sb = sorghum[sorghum["family"] == family_row_name]
    zm = maize[maize["family"] == family_row_name]
    family_row["Sb_No_Ortholog"] = (~sb["Has_Ortholog"]).sum()
    family_row["Zm_No_Ortholog"] = (~zm["Has_Ortholog"]).sum()
    family_row["Sb_Active"] = (sb["Action"] == "Activator").sum()
    family_row["Zm_Active"] = (zm["Action"] == "Activator").sum()

counts_df = pd.DataFrame(results)[count_cols]
counts_df.to_csv(OUTPUT_COUNTS, index=False)

# ---- Save gene lists per category per family ----
OUTPUT_GENE_LISTS = os.path.join(BASE_DIR, "results", "gene_family_gene_lists_sorghumversion3_action.pkl")
with open(OUTPUT_GENE_LISTS, "wb") as f:
    pickle.dump(gene_lists, f)

print(f"Proportions saved to {OUTPUT_FILE}")
print(f"Counts saved to {OUTPUT_COUNTS}")
print(f"Gene lists saved to {OUTPUT_GENE_LISTS}\n")
print(results_df.to_string(index=False))
