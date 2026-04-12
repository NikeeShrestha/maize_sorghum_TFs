import pickle
import pandas as pd
import numpy as np

file_path = '../sorghumdata/sbi_all_preds.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)
    
data=data[['gene_ID','protein_ID','family','seq', 'activity_avg']]


Activator=np.array([], dtype=float)
noactivity=np.array([], dtype=float)

allactivity=np.array([], dtype=float)

##categoriuze based on threhold of 1 for activator and -1 for no activity.
for i in range(len(data["gene_ID"])):
    arr = np.asarray(data.iloc[i, 4], dtype=float)

    # ignore NaNs explicitly (optional but clean)
    arr = arr[np.isfinite(arr)]
    
    

    if arr.size == 0:
        data.at[i, "Action"] = "None"
        continue
        
    allactivity=np.concatenate((allactivity,arr))

    has_pos = np.any(arr >= 1)
    # has_neg = np.any(arr < -1)
    
    has_neg = np.all(arr <= 1)

    if has_pos:
        # print(temp.iloc[i, 0], "Activator")
        data.at[i, "Action"] = "Activator"
        Activator=np.concatenate((Activator,arr))

    elif has_neg:
        data.at[i, "Action"] = "No activity"
        noactivity=np.concatenate((noactivity,arr))
        

    else:
        data.at[i, "Action"] = "None"

#categorize based on threhold of 3 for activator and -1 and 0.5 for no activity.

LOW  = -1
HIGH =  0.5

for i in range(len(data["gene_ID"])):
    arr = np.asarray(data.iloc[i, 4], dtype=float)

    # ignore NaNs explicitly (optional but clean)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        data.at[i, "RecategorizeAction"] = "None"
        continue

    has_pos = np.any(arr > 3)
    # has_neg = np.any(arr < -1)

    if has_pos:
        # print(temp.iloc[i, 0], "Activator")
        data.at[i, "RecategorizeAction"] = "Activator"
        Activator=np.concatenate((Activator,arr))

    # elif has_neg and not has_pos:
    #     # print(temp.iloc[i, 0], "Repressor")
    #     data.at[i, "Action"] = "Repressor"

    # elif has_pos and has_neg:
    #     # print(temp.iloc[i, 0], "Both")
    #     data.at[i, "Action"] = "Dual Action"

    if np.all((arr >= LOW) & (arr <= HIGH)):
        data.at[i, "RecategorizeAction"] = "No Activity"

data.to_csv('../sorghumdata/Sorghum_ActivityAnnotated.csv', index=False)