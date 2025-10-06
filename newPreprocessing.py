import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

#READ THE CSV/DATASET
df = pd.read_csv(r"raw_data/basic_data_4.csv").drop_duplicates()
df2 = pd.read_csv(r"raw_data/UNSW-NB15_1.csv").drop_duplicates()

#FILL MISSING VALUES, AVERAGES THEM TO BOTH WORK FOR BOTH DATASETS
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)
df2.fillna(df2.median(numeric_only=True), inplace=True)
df2.fillna(df2.mode().iloc[0], inplace=True)

#READ THE LABEL CATEGORY
label_category_map = pd.read_csv(r"raw_data/label_category_map.csv")
label_category_map_unsw = pd.read_csv(r"raw_data/unsw_label_category_map.csv")

#MAP THE LABELS TO CATEGORIES
df['category'] = df['label'].map(dict(zip(label_category_map['label'], label_category_map['category'])))
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

#WITH THE ADDITION OF THE NEW DATASET
xtr, xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
xtr2, xte2, ytr2, yte2 = train_test_split(X2, y2, stratify=y2, test_size=0.2, random_state=42)

#NORMALIZES THE DATASET WITH STANDARD SCALER (Standardization)
NORMALIZED = xtr.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
xtr[NORMALIZED] = scaler.fit_transform(xtr[NORMALIZED])
xte[NORMALIZED] = scaler.transform(xte[NORMALIZED])

NORMALIZED2 = xtr2.select_dtypes(include=[np.number]).columns
scaler2 = StandardScaler()
xtr2[NORMALIZED] = scaler2.fit_transform(xtr2[NORMALIZED2])
xte2[NORMALIZED] = scaler2.transform(xte2[NORMALIZED2])

ros = RandomOverSampler(random_state=50)
xtr_bal, ytr_bal = ros.fit_resample(xtr, ytr)
xtr2_bal, ytr2_bal = ros.fit_resample(xtr2, ytr2)

#BALANCED DATASET BUILD FOR BOTH DATASETS 
testingDataset = pd.DataFrame(xtr_bal, columns=xtr.columns)
testingDataset['encodedCategory'] = ytr_bal
validationDataset = pd.DataFrame(xte, columns=xte.columns)
validationDataset['encodedCategory'] = yte

testingDataset2 = pd.DataFrame(xtr2_bal, columns=xtr2.columns)
testingDataset2['encodedCategory'] = ytr2_bal
validationDataset2 = pd.DataFrame(xte2, columns=xte2.columns)
validationDataset2['encodedCategory'] = yte2

testingDataset.to_csv(r"processed_data/testingDataset.csv", index=False)
validationDataset.to_csv(r"processed_data/validationDataset.csv", index=False)
testingDataset2.to_csv(r"processed_data/testingDataset2.csv", index=False)
validationDataset2.to_csv(r"processed_data/validationDataset2.csv", index=False)
