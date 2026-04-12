# Maize & Sorghum Transcription Factor Activity Analysis

This repository contains scripts for analyzing transcription factor (TF) activity predictions in maize and sorghum, including variant overlap analysis, ortholog comparisons, heritability tests, and whole-genome duplication (WGD) analyses.

---

## Directory Structure

```
RelevantFiles/
├── input/                  # Ortholog mapping files
├── maizedata/              # Maize-specific input data
├── sorghumdata/            # Sorghum-specific input data
├── results/                # All generated outputs
└── scripts/                # Analysis scripts (described below)
```

---

## Scripts

### 1. `Categorize_maize_TFs.py`

**Description:** Categorizes maize transcription factors into activity classes based on predicted activity scores.

- **Action** — Activator (any score ≥ 1) or No activity (all scores ≤ 1)
- **RecategorizeAction** — Activator (any score > 3) or No Activity (all scores between −1 and 0.5)

| | Input | Path |
|---|---|---|
| 1 | Maize TF predictions (pickle) | `maizedata/zma_all_preds.pkl` |

| | Output | Path |
|---|---|---|
| 1 | Activity-annotated maize TFs | `maizedata/maize_ActivityAnnotated.csv` |

---

### 2. `Categorize_sorghum_TFs.py`

**Description:** Categorizes sorghum transcription factors into activity classes using the same thresholds as the maize script.

| | Input | Path |
|---|---|---|
| 1 | Sorghum TF predictions (pickle) | `sorghumdata/sbi_all_preds.pkl` |

| | Output | Path |
|---|---|---|
| 1 | Activity-annotated sorghum TFs | `sorghumdata/Sorghum_ActivityAnnotated.csv` |

---

### 3. `Variants_overlap_maize.py`

**Description:** Computes variant density (variants per kb) in activation domains (AD), DNA-binding domains (DD), and non-AD/non-DD regions for maize activator TFs. Runs the analysis at two thresholds:

- **Less stringent:** AD > 1, non-AD ≤ 1
- **More stringent:** AD > 3, non-AD between −1 and 0.5

| | Input | Path |
|---|---|---|
| 1 | Maize InterProScan annotations | `maizedata/zma_proteins.iprscan` |
| 2 | Maize activity annotations | `maizedata/maize_ActivityAnnotated.csv` |
| 3 | Maize variant effects | `maizedata/annotated_maize_TFs_variant_effects.csv` |
| 4 | Maize TF predictions (pickle) | `maizedata/zma_all_preds.pkl` |

| | Output | Path |
|---|---|---|
| 1 | Printed variant density table (AD vs DD vs non-AD-non-DD × missense/synonymous) | stdout |

---

### 4. `Variants_overlap_sorghum.py`

**Description:** Same analysis as `Variants_overlap_maize.py` but for sorghum. Computes variant density in AD, DD, and non-AD/non-DD regions at two thresholds.

| | Input | Path |
|---|---|---|
| 1 | Sorghum InterProScan annotations | `sorghumdata/sbi_proteins.iprscan` |
| 2 | Sorghum activity annotations | `sorghumdata/Sorghum_ActivityAnnotated.csv` |
| 3 | Sorghum variant effects | `sorghumdata/annotated_sorghum_TFs_variant_effects_chrom_renamed.csv` |
| 4 | Sorghum TF predictions (pickle) | `sorghumdata/sbi_all_preds.pkl` |

| | Output | Path |
|---|---|---|
| 1 | Printed variant density table (AD vs DD vs non-AD-non-DD × missense/synonymous) | stdout |

---

### 5. `maize_AD_DD_overlap.py`

**Description:** Identifies overlap between activation domains (ADs) and DNA-binding domains (DBDs) in maize activator proteins. For each protein, computes the number of overlapping regions and their lengths in base pairs.

| | Input | Path |
|---|---|---|
| 1 | Maize InterProScan annotations | `maizedata/zma_proteins.iprscan` |
| 2 | Maize activity annotations | `maizedata/maize_ActivityAnnotated.csv` |
| 3 | Maize TF predictions (pickle) | `maizedata/zma_all_preds.pkl` |

| | Output | Path |
|---|---|---|
| 1 | Proteins with AD–DD overlap | `results/maize_DD_AD_overlap_proteins.csv` |

---

### 6. `sorghum_AD_DD_overlap.py`

**Description:** Same analysis as `maize_AD_DD_overlap.py` but for sorghum. Identifies overlap between ADs and DBDs in sorghum activator proteins.

| | Input | Path |
|---|---|---|
| 1 | Sorghum InterProScan annotations | `sorghumdata/sbi_proteins.iprscan` |
| 2 | Sorghum activity annotations | `sorghumdata/Sorghum_ActivityAnnotated.csv` |
| 3 | Sorghum TF predictions (pickle) | `sorghumdata/sbi_all_preds.pkl` |

| | Output | Path |
|---|---|---|
| 1 | Proteins with AD–DD overlap | `results/sorghum_DD_AD_overlap_proteins.csv` |

---

### 7. `ortholog_pair_activity.py`

**Description:** For each sorghum–maize ortholog pair, classifies the pair into one of four activity categories:

1. **Both Active** — both species are Activator
2. **Both Inactive** — neither is Activator
3. **Sb Active / Zm Inactive** — sorghum active, maize inactive
4. **Zm Active / Sb Inactive** — maize active, sorghum inactive

Produces per-family stacked bar charts and proportion tables for both `Action` and `RecategorizeAction` columns.

| | Input | Path |
|---|---|---|
| 1 | Sorghum activity annotations | `sorghumdata/Sorghum_ActivityAnnotated.csv` |
| 2 | Maize activity annotations | `maizedata/maize_ActivityAnnotated.csv` |
| 3 | Ortholog mapping | `input/sorghumversion3_maizeversion3.csv` |

| | Output | Path |
|---|---|---|
| 1 | Proportions table (Action) | `results/ortholog_pair_activity_proportions.csv` |
| 2 | Stacked bar chart (Action) | `results/ortholog_pair_activity_stacked.png` |
| 3 | Proportions table (RecategorizeAction) | `results/ortholog_pair_activity_recategorized_proportions.csv` |
| 4 | Stacked bar chart (RecategorizeAction) | `results/ortholog_pair_activity_recategorized_stacked.png` |

---

### 8. `ortholog_activity_association_test.py`

**Description:** Runs two Fisher's exact tests for both `Action` and `RecategorizeAction`:

- **Test 1:** Does having an ortholog make a gene more likely to be an activator? (per species × gene family)
- **Test 2:** Is sorghum activity correlated with maize ortholog activity? (co-activity test on ortholog pairs)

Includes Benjamini–Hochberg FDR correction.

| | Input | Path |
|---|---|---|
| 1 | Sorghum activity annotations | `sorghumdata/Sorghum_ActivityAnnotated.csv` |
| 2 | Maize activity annotations | `maizedata/maize_ActivityAnnotated.csv` |
| 3 | Ortholog mapping | `input/sorghumversion3_maizeversion3.csv` |

| | Output | Path |
|---|---|---|
| 1 | Association test results (Action) | `results/ortholog_activity_association_test_action.csv` |
| 2 | Association test results (RecategorizeAction) | `results/ortholog_activity_association_test_recategorized.csv` |
| 3 | Co-activity test results (Action) | `results/ortholog_pair_coactivity_test_action.csv` |
| 4 | Co-activity test results (RecategorizeAction) | `results/ortholog_pair_coactivity_test_recategorized.csv` |

---

### 9. `heritability_activity_test.py`

**Description:** Tests whether activator genes have significantly different heritability compared to non-activator genes using the Mann–Whitney U test, for both maize and sorghum and both activity columns.

| | Input | Path |
|---|---|---|
| 1 | Sorghum activity annotations | `sorghumdata/Sorghum_ActivityAnnotated.csv` |
| 2 | Maize activity annotations | `maizedata/maize_ActivityAnnotated.csv` |
| 3 | Sorghum gene variance/heritability | `sorghumdata/sorghumgene_variance.csv` |
| 4 | Maize gene variance/heritability | `maizedata/maize_693_genemodels_variance.csv` |

| | Output | Path |
|---|---|---|
| 1 | Mann–Whitney test statistics | `results/heritability_activity_mannwhitney.csv` |
| 2 | Boxplot figure | `results/heritability_activity_boxplot.png` |

---

### 10. `maize_wgd_duplicate_activity_disagreement.py`

**Description:** Identifies maize WGD (whole-genome duplication) duplicate pairs where the two maize paralogs disagree on activity status — one is an Activator and the other is not. Uses sorghum genes that map to ≥ 2 maize orthologs to define WGD groups.

| | Input | Path |
|---|---|---|
| 1 | Sorghum activity annotations | `sorghumdata/Sorghum_ActivityAnnotated.csv` |
| 2 | Maize activity annotations | `maizedata/maize_ActivityAnnotated.csv` |
| 3 | Ortholog mapping | `input/sorghumversion3_maizeversion3.csv` |

| | Output | Path |
|---|---|---|
| 1 | All WGD duplicate groups with disagreement | `results/maize_wgd_duplicates_all.csv` |
| 2 | Disagreeing pairs (Action) | `results/maize_wgd_disagreement_Action.csv` |
| 3 | Disagreeing pairs (RecategorizeAction) | `results/maize_wgd_disagreement_RecatAction.csv` |

---

## Input Data Summary

| File | Description |
|---|---|
| `input/sorghumversion3_maizeversion3.csv` | Sorghum–maize ortholog mapping (columns: `SorghumGene`, `Ortholog`) |
| `maizedata/zma_all_preds.pkl` | Maize TF activity predictions (pickle with `activity_avg` arrays) |
| `maizedata/zma_proteins.iprscan` | Maize InterProScan protein domain annotations |
| `maizedata/maize_ActivityAnnotated.csv` | Maize TFs with `Action` and `RecategorizeAction` columns |
| `maizedata/annotated_maize_TFs_variant_effects.csv` | Maize variant effect annotations |
| `maizedata/maize_693_genemodels_variance.csv` | Maize gene heritability estimates |
| `maizedata/maize_gene_count_2pcs_0.01.csv` | Maize gene counts |
| `sorghumdata/sbi_all_preds.pkl` | Sorghum TF activity predictions (pickle) |
| `sorghumdata/sbi_proteins.iprscan` | Sorghum InterProScan protein domain annotations |
| `sorghumdata/Sorghum_ActivityAnnotated.csv` | Sorghum TFs with `Action` and `RecategorizeAction` columns |
| `sorghumdata/annotated_sorghum_TFs_variant_effects_chrom_renamed_hetfilter.csv` | Sorghum variant effect annotations |
| `sorghumdata/sorghumgene_variance.csv` | Sorghum gene heritability estimates |
| `sorghumdata/sorghum_gene_count_4pcs_0.01.csv` | Sorghum gene counts |

---

## Suggested Execution Order

Scripts should be run from the `scripts/` directory:

```bash
cd scripts/

# Step 1: Categorize TFs (produces ActivityAnnotated files used by all other scripts)
python Categorize_maize_TFs.py
python Categorize_sorghum_TFs.py

# Step 2: Domain & variant analyses
python maize_AD_DD_overlap.py
python sorghum_AD_DD_overlap.py
python Variants_overlap_maize.py
python Variants_overlap_sorghum.py

# Step 3: Ortholog analyses
python ortholog_pair_activity.py
python ortholog_activity_association_test.py

# Step 4: Additional analyses
python heritability_activity_test.py
python maize_wgd_duplicate_activity_disagreement.py
```
