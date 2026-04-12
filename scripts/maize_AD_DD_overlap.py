import pandas as pd
import numpy as np
import pickle
import csv


'''
This script identifies the overlap between DNA-binding domains (DBDs) 
and activation domains (ADs) in maize proteins. It reads in protein data, 
filters for DNA-binding annotations, computes continuous regions of activation,
and calculates the overlap between these regions and the DBDs. 
The results are saved to a CSV file for further analysis.
'''

def continuous_ranges_above_one(arr: pd.Series) -> list[tuple[int, int]]:
    """
    Given a pandas Series (or any 1‑D array‑like) of numeric values,
    return a list of (start, end) positions (1‑based) for each
    contiguous region where the value is > 1.
    Example
    -------
    arr = [0.8, 1.2, 1.5, 0.9, 2.0, 2.3, 0.7]
    → [(2, 3), (5, 6)]
    """
    # Boolean mask: True where value > 1
    mask = arr > 1
    # Find the indices where the mask changes (False→True or True→False)
    # `diff()` gives NaN at the first element, we fill it with 0.
    changes = mask.astype(int).diff().fillna(0)
    # Start of a region: change from 0→1 (value == 1)
    starts = changes[changes == 1].index
    # End of a region: change from 1→0 (value == -1)
    ends   = changes[changes == -1].index
    # Edge cases: region may start at the very first element
    if mask.iloc[0]:
        starts = pd.Index([mask.index[0]]).append(starts)
    # Region may end at the very last element
    if mask.iloc[-1]:
        ends = ends.append(pd.Index([mask.index[-1]]))
    # Convert to 1‑based positions (add 1 because pandas index is 0‑based)
    ranges = [(int(s) + 1, int(e) + 1) for s, e in zip(starts, ends)]
    return ranges
def overlap(r1, r2):
    return r1[0] <= r2[1] and r2[0] <= r1[1]

def overlap_length(r1, r2):
    start = max(r1[0], r2[0])
    end = min(r1[1], r2[1])
    
    if start <= end:
        return end - start + 1   # +1 if positions are inclusive (common in genomics/proteins)
    else:
        return 0
    
columns = [ "protein_ID", "sequence_md5_digest", "sequence_length",
           "analysis", "signature_accession", "signature_description",
           "start_location", "stop_location", "score", "status", "date",
           "interpro_annotations_accession", "interpro_annotations_description",
           "go_annotations", "pathway_annotations"]
regex_pattern = r'\bDNA[- ]binding\b'

# unique_sbi_proteins = pd.read_csv("../sorghumdata/sbi_proteins.iprscan", 
#                                   sep='\t',names=columns, engine='python', quoting=csv.QUOTE_NONE)

unique_zma_proteins = pd.read_csv("../maizedata/zma_proteins.iprscan", sep='\t',
                                  names=columns, engine='python',quoting=csv.QUOTE_NONE)
# dbd_df = pd.concat([unique_sbi_proteins,unique_zma_proteins]).reset_index(drop=True)

dbd_df=unique_zma_proteins

dna_binding_mask = (dbd_df['signature_description'].str.contains(regex_pattern, case=False, na=False) |
                    dbd_df['interpro_annotations_description'].str.contains(regex_pattern, case=False, na=False))
dbd_df = dbd_df[dna_binding_mask]

dbd_df = dbd_df[['protein_ID', 'analysis','signature_description', 'interpro_annotations_accession','start_location','stop_location']].drop_duplicates()
dbd_df = dbd_df[dbd_df['signature_description'] != '-']
dbd_df = dbd_df[dbd_df['analysis'] == 'Pfam']

dbd_df['range'] = dbd_df.apply(lambda row: (row['start_location'],row['stop_location']),axis=1)
dbd_df = dbd_df.groupby('protein_ID')['range'].agg(lambda x: list(x)).reset_index()
print(f"{dbd_df['protein_ID'].nunique()} unique proteins.")

def get_overall_range(ranges):
    min_start = min(r[0] for r in ranges)
    max_end = max(r[1] for r in ranges)
    return (min_start, max_end)

dbd_df['overall_range'] = dbd_df['range'].apply(get_overall_range)

# dbd_df['protein_ID']=[x.replace('.p','') for x in dbd_df['protein_ID']]

dbd_protein = (
    dbd_df
    .groupby('protein_ID')['range']
    .sum()
    .to_dict()
)

file_path = '../maizedata/zma_all_preds.pkl' # Replace with your file's actual path
with open(file_path, 'rb') as file:
    data = pickle.load(file)
data=data[['gene_ID','protein_ID','family','seq', 'activity_avg']]

TFs=pd.read_csv('../maizedata/maize_ActivityAnnotated.csv')
transcripts_activator=set(TFs[TFs['Action']=='Activator']['protein_ID'])

activator_AD = {}                     # result container
for ta in transcripts_activator:                 # each activator name
    # Grab the *first* activity_avg series for this transcript
    # (you used `.iloc[0]` in the original code, so we keep that logic)
    arr = data.loc[data['protein_ID'] == ta, 'activity_avg'].iloc[0]
    # Ensure we are working with a pandas Series (in case it is a list/ndarray)
    if not isinstance(arr, pd.Series):
        arr = pd.Series(arr)
    # Compute the continuous >1 ranges
    activator_AD[ta] = continuous_ranges_above_one(arr)

dd_ad_overlap={}

dd_ad_bp_overlap={}
countdd=0
for ad in activator_AD:
     
    dd=False
    
    if ad in dbd_protein:
        countdd+=1
        count=0
        dd_domain=dbd_protein[ad]
        dd=True
    else: 
        continue
    
    for i in activator_AD[ad]:
        if dd:
            for j in dd_domain:
                if overlap(i, j):
                    count+=1
                    bp_overlap=overlap_length(i,j)
                    
                    if ad not in dd_ad_bp_overlap:
                        dd_ad_bp_overlap[ad]=[]
                    dd_ad_bp_overlap[ad].append(bp_overlap)
    dd_ad_overlap[ad]=count
            
dd_ad_count=pd.DataFrame([dd_ad_overlap]).T

# print(dd_ad_count)

dd_ad_count[0].value_counts()

overlapdf=pd.DataFrame(list(dd_ad_bp_overlap.items()), columns=['protein_ID', 'AD_DD_overlapbp'])

overlapdf.to_csv('../results/maize_DD_AD_overlap_proteins.csv', index=False)

print(f"Total proteins with DD and AD overlap: {len(overlapdf)}")

print('file saved to ../results/maize_DD_AD_overlap_proteins.csv')