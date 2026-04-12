import pandas as pd
import numpy as np
import pickle
import csv

columns = [ "protein_ID", "sequence_md5_digest", "sequence_length",
           "analysis", "signature_accession", "signature_description",
           "start_location", "stop_location", "score", "status", "date",
           "interpro_annotations_accession", "interpro_annotations_description",
           "go_annotations", "pathway_annotations"]
regex_pattern = r'\bDNA[- ]binding\b'

unique_sbi_proteins = pd.read_csv("../sorghumdata/sbi_proteins.iprscan", 
                                  sep='\t',names=columns, engine='python', quoting=csv.QUOTE_NONE)

# unique_zma_proteins = pd.read_csv("../maizedata/zma_proteins.iprscan", sep='\t',
#                                   names=columns, engine='python',quoting=csv.QUOTE_NONE)
# dbd_df = pd.concat([unique_sbi_proteins,unique_zma_proteins]).reset_index(drop=True)

dbd_df=unique_sbi_proteins

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
dbd_df['old_protein_ID']=dbd_df['protein_ID']

dbd_df['protein_ID']=[x.replace('.p','') for x in dbd_df['protein_ID']]

dbd_protein = (
    dbd_df
    .groupby('protein_ID')['range']
    .sum()
    .to_dict()
)

variantdata=pd.read_csv('../sorghumdata/annotated_sorghum_TFs_variant_effects_chrom_renamed.csv')
TFs=pd.read_csv('../sorghumdata/Sorghum_ActivityAnnotated.csv')
TFs['protein_ID']=TFs['protein_ID'].str.replace('.p', '')
TFs['trancript']=TFs['protein_ID']

activators=set(TFs[TFs['Action']=='Activator']['trancript'])
variantdata.rename(columns={'gene': 'gene_ID'}, inplace=True)

file_path = '../sorghumdata/sbi_all_preds.pkl' # Replace with your file's actual path
with open(file_path, 'rb') as file:
    data = pickle.load(file)

data=data[['gene_ID','protein_ID','family','seq', 'activity_avg']]
data['protein_ID']=data['protein_ID'].str.replace('.p', '')
data['trancript']=data['protein_ID']

transcript_nomatch=set()

transcripts_activator=set()

transcript_nomatch_length=set()
for _, row in variantdata.iterrows():
    
    transcript=row['transcript']

    if not transcript in activators: continue

    arr=data.loc[data['trancript'] == transcript, 'activity_avg'].iloc[0]

    variantpos=int(row['protein_position'])-1

    if (len(arr)!=int(row['protein_length'])):
        # variantdata.at[_, 'activity']='None'
        transcript_nomatch_length.add(transcript)
        continue
    if int(row['protein_position'])>len(arr):
        # print(transcript, int(row['protein_position']), len(arr)) ##because of stop codon not included
        continue

    transcripts_activator.add(transcript)
    
    is_dd = False
    
    if transcript in dbd_protein:
        dnabindingdomain = dbd_protein[transcript]

        for dbd in dnabindingdomain:
            if dbd[0] <= variantpos <= dbd[1]:
                variantdata.at[_, 'activity'] = 'DD'
                is_dd = True
                break
            
    if not is_dd:
        if arr[variantpos]>1:
            variantdata.at[_, 'activity']='AD'
        else:
            variantdata.at[_, 'activity']='non-AD-non-DD'
        
variantdata = variantdata[variantdata['activity'].isin(['AD','DD', 'non-AD-non-DD'])]
activatorlen=0
nonactivatorlen=0
domainbindlen=0
transcript_activator_len=[]
count_dd=set()

for ta in transcripts_activator:
    arr=data.loc[data['trancript'] == ta, 'activity_avg'].iloc[0]
    transcript_activator_len.append(len(arr))
    
    for arrpos in range(0,len(arr)):
        
        is_dd=False
    
        if ta in dbd_protein:
            count_dd.add(ta)
            domainposition=dbd_protein[ta]
            for dbd in domainposition:
                if dbd[0] <= arrpos+1 <= dbd[1]:
                    domainbindlen+=1
                    is_dd = True
                    break
            
        if not is_dd:
            if arr[arrpos]>1:
                activatorlen+=1
            else:
                nonactivatorlen+=1

print(activatorlen,domainbindlen,nonactivatorlen)

# counts = variantdata.groupby(['activity', 'effect']).size()
# # countsdf=pd.DataFrame(counts)

# # countsdf.columns=['effect', 'total']
countsdf = (
    variantdata
      .groupby(['activity', 'effect'])
      .size()
      .reset_index(name='total')
)
countsdf['length'] = np.where(
    countsdf['activity'] == 'AD',
    activatorlen,
    np.where(
        countsdf['activity'] == 'DD',
        domainbindlen,
        nonactivatorlen
    )
)
countsdf['variantsperkb']=(countsdf['total']/countsdf['length'])*1000
df_small = countsdf[['activity', 'effect', 'variantsperkb']]

# Pivot
df_pivot = df_small.pivot(index='effect',
                          columns='activity',
                          values='variantsperkb')

print('threhsold used: AD >1, non-AD-non-DD <=1')
print(df_pivot)

## more stringent threshold
variantdata=pd.read_csv('../sorghumdata/annotated_sorghum_TFs_variant_effects_chrom_renamed.csv')
TFs=pd.read_csv('../sorghumdata/Sorghum_ActivityAnnotated.csv')
TFs['protein_ID']=TFs['protein_ID'].str.replace('.p', '')
TFs['trancript']=TFs['protein_ID']

activators=set(TFs[TFs['Action']=='Activator']['trancript'])
variantdata.rename(columns={'gene': 'gene_ID'}, inplace=True)

file_path = '../sorghumdata/sbi_all_preds.pkl' # Replace with your file's actual path
with open(file_path, 'rb') as file:
    data = pickle.load(file)

data=data[['gene_ID','protein_ID','family','seq', 'activity_avg']]
data['protein_ID']=data['protein_ID'].str.replace('.p', '')
data['trancript']=data['protein_ID']

LOW=-1
HIGH=0.5

transcript_nomatch=set()


transcripts_activator=set()
for _, row in variantdata.iterrows():
    
    transcript=row['transcript']

    if not transcript in activators: continue

    arr=data.loc[data['trancript'] == transcript, 'activity_avg'].iloc[0]

    variantpos=int(row['protein_position'])-1

    if (len(arr)!=int(row['protein_length'])):
        # variantdata.at[_, 'activity']='None'
        # print(transcript, int(row['protein_length']), len(arr))
        transcript_nomatch.add(transcript)
        continue
    if int(row['protein_position'])>len(arr):
        # print(transcript, int(row['protein_position']), len(arr))
        continue

    transcripts_activator.add(transcript)
    
    is_dd = False
    
    if transcript in dbd_protein:
        dnabindingdomain = dbd_protein[transcript]

        for dbd in dnabindingdomain:
            if dbd[0] <= variantpos <= dbd[1]:
                variantdata.at[_, 'activity'] = 'DD'
                is_dd = True
                break
    if not is_dd:
        if arr[variantpos]>3:
            variantdata.at[_, 'activity']='AD'
        elif (arr[variantpos]>=LOW) & (arr[variantpos]<=HIGH):
            variantdata.at[_, 'activity']='non-AD-non-DD'

        
variantdata = variantdata[variantdata['activity'].isin(['AD', 'DD','non-AD-non-DD'])]

activatorlen=0
nonactivatorlen=0
domainbindlen=0

count_dd=set()

for ta in transcripts_activator:
    arr=data.loc[data['trancript'] == ta, 'activity_avg'].iloc[0]
    
    for arrpos in range(0,len(arr)):
        
        is_dd=False
    
        if ta in dbd_protein:
            domainposition=dbd_protein[ta]
            count_dd.add(ta)
            for dbd in domainposition:
                if dbd[0] <= arrpos+1 <= dbd[1]:
                    domainbindlen+=1
                    is_dd = True
                    break
            
        if not is_dd:
            if arr[arrpos]>3:
                activatorlen+=1
            elif (arr[arrpos] >= LOW) & (arr[arrpos] <= HIGH):
                nonactivatorlen+=1

print(activatorlen,domainbindlen,nonactivatorlen)

countsdf = (
    variantdata
      .groupby(['activity', 'effect'])
      .size()
      .reset_index(name='total')
)
countsdf['length'] = np.where(
    countsdf['activity'] == 'AD',
    activatorlen,
    np.where(
        countsdf['activity'] == 'DD',
        domainbindlen,
        nonactivatorlen
    )
)

countsdf['variantsperkb']=(countsdf['total']/countsdf['length'])*1000
df_small = countsdf[['activity', 'effect', 'variantsperkb']]

# Pivot
df_pivot = df_small.pivot(index='effect',
                          columns='activity',
                          values='variantsperkb')

print('threshold used: AD >3, non-AD-non-DD >=-1 and <=0.5')
print(df_pivot)