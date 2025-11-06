import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

from clean_basic import clean_basic, clean_nb15

#Paths
BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / "processed_data" / "merged_shared.csv"

basic_df = clean_basic()
nb15_df = clean_nb15()

merged_df = pd.concat([basic_df, nb15_df], ignore_index=True).drop_duplicates()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
merged_df.to_csv(OUT_PATH, index=False)

#ENCODE THE LABELS AND CATEGORIES
num_cols = ['duration', 'src_bytes', 'dst_bytes']
cat_cols = ['protocol_type', 'service', 'flag', 'source']
encoders = {}
for c in cat_cols:
    merged_df["encoded_"+c] = LabelEncoder().fit_transform(merged_df[c].astype(str))

merged_df['encoded_category'] = LabelEncoder().fit_transform(merged_df['category'].astype(str))
 
X = merged_df[num_cols + ["encoded_"+c for c in cat_cols]]
y = merged_df['encoded_category']
 
# THE DATASET SPLIT DATASET. DATASET IS 20% OF THE TOTAL DATASET AND THE OVERSAMPLING IS WORK.
#REST DATASET IS 80%!!!!
#VALIDATION DATASET EXIST BECAUSE WE NEED TO VALIDATE THAT THE DATASET IS WORKNG/TRAINING WORKS
xtr, xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=50)
 
#NORMALIZES THE DATASET WITH STANDARD SCALER (Standardization)
NORMALIZED = xtr.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
xtr[NORMALIZED] = scaler.fit_transform(xtr[NORMALIZED])
xte[NORMALIZED] = scaler.transform(xte[NORMALIZED])
 
rus = RandomUnderSampler(random_state=50)
xtr_bal, ytr_bal = rus.fit_resample(xtr, ytr)
 
#BALANCED DATASET BUILD
trainingDataset = pd.DataFrame(xtr_bal, columns=xtr.columns)
trainingDataset['encodedCategory'] = ytr_bal
validationDataset = pd.DataFrame(xte, columns=xte.columns)
validationDataset['encodedCategory'] = yte
 
trainingDataset.to_csv(r"processed_data/merged/trainingDataset.csv", index=False)
validationDataset.to_csv(r"processed_data/merged/validationDataset.csv", index=False)