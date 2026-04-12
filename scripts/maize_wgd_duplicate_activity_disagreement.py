#!/usr/bin/env python3
"""
Maize WGD Duplicate Activity Disagreement Analysis
===================================================

Focus: Cases where MULTIPLE maize genes share the SAME sorghum ortholog
       (i.e., maize paralogs arising from Whole Genome Duplication).

Steps:
  1. Load ortholog table: sorghumversion3_maizeversion3.csv
     Columns: SorghumGene, Ortholog (GRMZM-style maize v3 IDs)

  2. Load maize and sorghum activity annotation files.
     Match maize via old_version3_gene_ID (GRMZM IDs).

  3. Identify sorghum genes that map to ≥2 maize genes
     (true WGD duplicates in maize).

  4. For each such sorghum gene, build all pairwise comparisons of
     its maize duplicates and flag pairs that DISAGREE on activity:
       - One maize gene classified as "Activator",
         the other NOT "Activator" (No activity / None / NaN).
     Checked for both Action and RecategorizeAction columns.

  5. Also note the sorghum gene's own activity classification.

Outputs:
  results/maize_wgd_duplicates_all.csv          – all duplicate groups
  results/maize_wgd_disagreement_Action.csv      – disagreeing pairs (Action)
  results/maize_wgd_disagreement_RecatAction.csv – disagreeing pairs (RecategorizeAction)
"""

import os
import itertools
import pandas as pd

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SORGHUM_FILE  = os.path.join(BASE_DIR, "sorghumdata", "Sorghum_ActivityAnnotated.csv")
MAIZE_FILE    = os.path.join(BASE_DIR, "maizedata",   "maize_ActivityAnnotated.csv")
ORTHOLOG_FILE = os.path.join(BASE_DIR, "input",     "sorghumversion3_maizeversion3.csv")

OUT_ALL       = os.path.join(BASE_DIR, "results", "maize_wgd_duplicates_all.csv")
OUT_DIS_ACT   = os.path.join(BASE_DIR, "results", "maize_wgd_disagreement_Action.csv")
OUT_DIS_RECAT = os.path.join(BASE_DIR, "results", "maize_wgd_disagreement_RecatAction.csv")

# ──────────────────────────────────────────────────────────────────
# 1.  Load data
# ──────────────────────────────────────────────────────────────────
print("Loading data...")
ortho   = pd.read_csv(ORTHOLOG_FILE)
sorghum = pd.read_csv(SORGHUM_FILE,  usecols=["gene_ID", "family", "Action", "RecategorizeAction"])
maize   = pd.read_csv(MAIZE_FILE,    usecols=["gene_ID", "old_version3_gene_ID",
                                               "family", "Action", "RecategorizeAction"])

print(f"  Ortholog pairs:  {len(ortho):,}")
print(f"  Sorghum genes:   {len(sorghum):,}")
print(f"  Maize genes:     {len(maize):,}\n")

# ──────────────────────────────────────────────────────────────────
# 2.  Merge maize activity onto ortholog table via old_version3_gene_ID
# ──────────────────────────────────────────────────────────────────
maize_clean = maize.dropna(subset=["old_version3_gene_ID"]).copy()
maize_clean["old_version3_gene_ID"] = maize_clean["old_version3_gene_ID"].astype(str).str.strip()

ortho_maize = ortho.merge(
    maize_clean.rename(columns={"old_version3_gene_ID": "Ortholog"}),
    on="Ortholog",
    how="inner"
).rename(columns={
    "gene_ID":           "Maize_gene_ID",
    "family":            "Maize_family",
    "Action":            "Maize_Action",
    "RecategorizeAction":"Maize_RecatAction",
})

print(f"Ortholog pairs with maize activity data: {len(ortho_maize):,}")

# ──────────────────────────────────────────────────────────────────
# 3.  Merge sorghum activity onto the pairs
# ──────────────────────────────────────────────────────────────────
ortho_full = ortho_maize.merge(
    sorghum.rename(columns={
        "gene_ID":           "SorghumGene",
        "family":            "Sorghum_family",
        "Action":            "Sorghum_Action",
        "RecategorizeAction":"Sorghum_RecatAction",
    }),
    on="SorghumGene",
    how="left"
)

# ──────────────────────────────────────────────────────────────────
# 4.  Identify WGD duplicate groups: sorghum genes with ≥2 maize hits
# ──────────────────────────────────────────────────────────────────
maize_per_sorghum = ortho_full.groupby("SorghumGene")["Ortholog"].nunique()
dup_sorghum_genes = maize_per_sorghum[maize_per_sorghum >= 2].index

dup_df = ortho_full[ortho_full["SorghumGene"].isin(dup_sorghum_genes)].copy()

print(f"\nSorghum genes with ≥2 maize orthologs (WGD duplicates): {len(dup_sorghum_genes):,}")
print(f"Maize gene entries in those groups:                      {len(dup_df):,}")

# ──────────────────────────────────────────────────────────────────
# 5.  Build pairwise comparisons and detect disagreements
# ──────────────────────────────────────────────────────────────────

def is_activator(val):
    """Return True if the activity value indicates an activator."""
    if pd.isna(val):
        return False
    return str(val).strip() == "Activator"


def has_valid_classification(val):
    """Return True if the value is non-NaN and non-empty."""
    if pd.isna(val):
        return False
    return str(val).strip() not in ("", "nan")


def build_pairwise_disagreements(dup_dataframe, action_col, maize_col):
    """
    For each sorghum gene group, make all pairwise combinations of
    maize duplicates and return those where activity disagrees:
    one is Activator, the other is not.

    action_col  : sorghum column name (e.g. 'Sorghum_Action')
    maize_col   : maize column name   (e.g. 'Maize_Action')
    """
    rows = []
    for sorghum_gene, grp in dup_dataframe.groupby("SorghumGene"):
        grp = grp.drop_duplicates(subset="Ortholog").reset_index(drop=True)
        if len(grp) < 2:
            continue

        # All pairwise combinations of maize duplicates within this group
        for i, j in itertools.combinations(grp.index, 2):
            row_i = grp.loc[i]
            row_j = grp.loc[j]

            # Skip pairs where either gene lacks a valid classification
            if not has_valid_classification(row_i[maize_col]) or \
               not has_valid_classification(row_j[maize_col]):
                continue

            act_i = is_activator(row_i[maize_col])
            act_j = is_activator(row_j[maize_col])

            # Disagreement: exactly one is Activator
            if act_i == act_j:
                continue

            # Determine which one is activator
            if act_i:
                activator_row, inactive_row = row_i, row_j
            else:
                activator_row, inactive_row = row_j, row_i

            rows.append({
                "SorghumGene":         sorghum_gene,
                "Sorghum_family":      row_i["Sorghum_family"],
                "Sorghum_Action_used": row_i[action_col],          # sorghum activity
                # Activator maize gene
                "Maize_Activator_v3ID":    activator_row["Ortholog"],
                "Maize_Activator_geneID":  activator_row["Maize_gene_ID"],
                f"Maize_Activator_{maize_col}": activator_row[maize_col],
                # Non-activator maize gene
                "Maize_NonActivator_v3ID":   inactive_row["Ortholog"],
                "Maize_NonActivator_geneID": inactive_row["Maize_gene_ID"],
                f"Maize_NonActivator_{maize_col}": inactive_row[maize_col],
                # Total duplicates in this group
                "N_Maize_Duplicates": len(grp),
            })
    return pd.DataFrame(rows)


print("\n── Action column ──────────────────────────────────────────")
dis_action = build_pairwise_disagreements(dup_df, "Sorghum_Action", "Maize_Action")
print(f"Disagreeing pairs (Action):            {len(dis_action):,}")
if len(dis_action) > 0:
    print(f"Unique sorghum genes with disagreement: "
          f"{dis_action['SorghumGene'].nunique():,}")
    # Summary by sorghum activity
    sb_act_summary = dis_action["Sorghum_Action_used"].value_counts(dropna=False)
    print("  Sorghum gene activity breakdown:")
    for val, cnt in sb_act_summary.items():
        print(f"    {val}: {cnt}")
dis_action.to_csv(OUT_DIS_ACT, index=False)
print(f"  → Saved: {OUT_DIS_ACT}")

print("\n── RecategorizeAction column ──────────────────────────────")
dis_recat = build_pairwise_disagreements(dup_df, "Sorghum_RecatAction", "Maize_RecatAction")
print(f"Disagreeing pairs (RecategorizeAction):{len(dis_recat):,}")
if len(dis_recat) > 0:
    print(f"Unique sorghum genes with disagreement: "
          f"{dis_recat['SorghumGene'].nunique():,}")
    sb_act_summary2 = dis_recat["Sorghum_Action_used"].value_counts(dropna=False)
    print("  Sorghum gene activity breakdown:")
    for val, cnt in sb_act_summary2.items():
        print(f"    {val}: {cnt}")
dis_recat.to_csv(OUT_DIS_RECAT, index=False)
print(f"  → Saved: {OUT_DIS_RECAT}")

# ──────────────────────────────────────────────────────────────────
# Save full group table filtered to only disagreeing sorghum genes
# ──────────────────────────────────────────────────────────────────
disagreeing_sorghum = set(dis_action["SorghumGene"].tolist() + dis_recat["SorghumGene"].tolist())
cols_out = ["SorghumGene", "Sorghum_family", "Sorghum_Action", "Sorghum_RecatAction",
            "Ortholog", "Maize_gene_ID", "Maize_Action", "Maize_RecatAction"]
dup_filtered = dup_df[dup_df["SorghumGene"].isin(disagreeing_sorghum)]
dup_filtered[cols_out].sort_values(["SorghumGene", "Ortholog"]).to_csv(OUT_ALL, index=False)
print(f"\nSorghum genes with at least one disagreeing pair: {len(disagreeing_sorghum):,}")
print(f"  → Saved: {OUT_ALL}")

# ──────────────────────────────────────────────────────────────────
# 6.  Quick family-level summary for Action disagreements
# ──────────────────────────────────────────────────────────────────
print("\n── Family breakdown (Action disagreements) ─────────────────")
if len(dis_action) > 0:
    fam_summary = (dis_action.groupby("Sorghum_family")
                              .size()
                              .reset_index(name="N_Disagreeing_Pairs")
                              .sort_values("N_Disagreeing_Pairs", ascending=False))
    print(fam_summary.to_string(index=False))

print("\n── Family breakdown (RecatAction disagreements) ────────────")
if len(dis_recat) > 0:
    fam_summary2 = (dis_recat.groupby("Sorghum_family")
                               .size()
                               .reset_index(name="N_Disagreeing_Pairs")
                               .sort_values("N_Disagreeing_Pairs", ascending=False))
    print(fam_summary2.to_string(index=False))

print("\nDone.")
