import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
 
#READ THE CSV/DATASET
df = pd.read_csv(r"raw_data/basic_data_4.csv").drop_duplicates()
 
#FILL MISSING VALUES, AVARES THM MEIDAN AND MODE
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)
 
#READ THE LABEL CATEGORY
label_category_map = pd.read_csv(r"raw_data/label_category_map.csv")
 
#MAP THE LABELS TO CATEGORIES
df['category'] = df['label'].map(dict(zip(label_category_map['label'], label_category_map['category'])))
 
#ENCODE THE LABELS AND CATEGORIES
b = ['category', 'label', 'protocol_type', 'service', 'flag']
encoders = {}
for a in b:
    df[a] = LabelEncoder().fit_transform(df[a].astype(str))
 
X = df.drop(['label', 'category', 'protocol_type', 'service', 'flag'], axis=1)
y = df['category']
 
# THE DATASET SPLIT DATASET. DATASET IS 20% OF THE TOTAL DATASET AND THE OVERSAMPLING IS WORK.
#REST DATASET IS 80%!!!!
#VALIDATION DATASET EXIST BECAUSE WE NEED TO VALIDATE THAT THE DATASET IS WORKNG/TRAINING WORKS
xtr, xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=50)
 
#NORMALIZES THE DATASET WITH STANDARD SCALER (Standardization)
NORMALIZED = xtr.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
xtr[NORMALIZED] = scaler.fit_transform(xtr[NORMALIZED])
xte[NORMALIZED] = scaler.transform(xte[NORMALIZED])
 
ros = RandomOverSampler(random_state=50)
xtr_bal, ytr_bal = ros.fit_resample(xtr, ytr)
 
#BALANCED DATASET BUILD
trainingDataset = pd.DataFrame(xtr_bal, columns=xtr.columns)
trainingDataset['encodedCategory'] = ytr_bal
validationDataset = pd.DataFrame(xte, columns=xte.columns)
validationDataset['encodedCategory'] = yte
 
trainingDataset.to_csv(r"processed_data/basic/trainingDataset.csv", index=False)
validationDataset.to_csv(r"processed_data/basic/validationDataset.csv", index=False)
