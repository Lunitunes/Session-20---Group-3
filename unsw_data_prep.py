#!/usr/bin/env python3
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

print("=== UNSW Pre-Processing Script ===")
print("Pre-Processing Setting Up...")

# Paths
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA = BASE_DIR / "raw_data"
OUT_DATA = BASE_DIR / "processed_data" / "nb15"

# Check folders exist
if not RAW_DATA.exists() or not OUT_DATA.exists():
    print("Environment not set up correctly.")
    print("Missing directories:")
    if not RAW_DATA.exists(): print(f" - {RAW_DATA}")
    if not OUT_DATA.exists(): print(f" - {OUT_DATA}")
    print(" Please retry unzipping folder")
    sys.exit(1)
else:
    print(f"Found data folders:\n  Raw: {RAW_DATA}\n  Out: {OUT_DATA}")

# Filenames
FEATURES_CSV = "UNSW-NB15_features.csv"
LABEL_MAP_CSV = "label_category_map_unsw.csv"
RAW_FILE     = "UNSW-NB15_4.csv"

# Load feature names
try:
    features_df = pd.read_csv(RAW_DATA / FEATURES_CSV, encoding="cp1252", skipinitialspace=True)
    features_df.columns = features_df.columns.astype(str).str.strip()
    if "Name" not in features_df.columns:
        print("Feature file headers detected:", features_df.columns.tolist())
        sys.exit(1)
    feature_names = features_df["Name"].astype(str).str.strip().tolist()
    print("Loaded feature names:", len(feature_names))
except FileNotFoundError:
    print(f"Could not find {FEATURES_CSV} in {RAW_DATA}")
    sys.exit(1)

# Load raw data and assign names
try:
    df = pd.read_csv(RAW_DATA / RAW_FILE, header=None, low_memory=False)
    df.columns = feature_names[: df.shape[1]]
    print(f"{RAW_FILE} successfully read and columns linked.")
except FileNotFoundError:
    print(f"Could not find {RAW_FILE} in {RAW_DATA}")
    sys.exit(1)
except pd.errors.EmptyDataError:
    print(f"{RAW_FILE} is empty.")
    sys.exit(1)
except Exception as e:
    print("Unexpected error while loading raw data:\n", e)
    sys.exit(1)

# Normalize column names to be safe (fixes Spkts/Dpkts, ct_src_ ltm, etc.)
df.columns = (
    pd.Series(df.columns)
      .astype(str)
      .str.strip()
      .str.replace(r"\s+", "", regex=True)
      .str.lower()
)

# Fill attack_cat for normal rows (Label=0)
if "label" not in df.columns or "attack_cat" not in df.columns:
    print("Expected 'label' and 'attack_cat' columns not found. Last columns were:",
          df.columns[-8:].tolist())
    sys.exit(1)

normal_before = df["attack_cat"].isna().sum()
df.loc[df["label"] == 0, "attack_cat"] = "Normal"
print(f"Filled {normal_before} rows: Label=0 => attack_cat='Normal'.")

# Load label map & build mapping (case/space-insensitive)
try:
    label_map = pd.read_csv(RAW_DATA / LABEL_MAP_CSV, encoding="cp1252", skipinitialspace=True)
except FileNotFoundError:
    print(f"Could not find {LABEL_MAP_CSV} in {RAW_DATA}")
    sys.exit(1)

label_map.columns = label_map.columns.astype(str).str.strip()
# Expect columns: encodedCategory, categoryName, description
possible_cat_cols = ["categoryName", "CategoryName", "category", "Category"]
cat_col = next((c for c in possible_cat_cols if c in label_map.columns), None)
if cat_col is None or "encodedCategory" not in label_map.columns:
    print("Label map missing expected columns. Found:", label_map.columns.tolist())
    sys.exit(1)

label_map[cat_col] = label_map[cat_col].astype(str).str.strip().str.lower()
label_map["encodedCategory"] = label_map["encodedCategory"].astype(int)

# Encode attack_cat to encodedCategory
df["attack_cat"] = df["attack_cat"].astype(str).str.strip().str.lower()
map_cat = dict(zip(label_map[cat_col], label_map["encodedCategory"]))
df["encodedcategory"] = df["attack_cat"].map(map_cat)

unmapped = df.loc[df["encodedcategory"].isna(), "attack_cat"].value_counts()
if len(unmapped):
    print("Unmapped attack_cat values (showing top 10):")
    print(unmapped.head(10))
else:
    print("All categories mapped successfully.")

# Select features
cols = [
    "proto", "service", "state", "dur",
    "sbytes", "dbytes", "spkts", "dpkts",
    "sttl", "dttl", "swin", "dwin",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
]

missing = [c for c in cols if c not in df.columns]
if missing:
    print("Missing expected feature columns:", missing)
    sys.exit(1)

X = df[cols].copy()
y = df["encodedcategory"].astype(int)

# Encode feature categoricals
for c in ["proto", "service", "state"]:
    if c in X.columns:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))

# Train/Validation split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale numeric columns fit on train only
num_cols = X_train.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

# Undersample

counts = Counter(y_train)
rus = RandomUnderSampler(random_state=50)
X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)

print("\nClass counts BEFORE undersample:", counts)
print("Class counts AFTER  undersample:", Counter(y_train_bal))

# Build output DataFrames 
train_df = X_train_bal.copy()
train_df["encodedCategory"] = y_train_bal

val_df = X_test.copy()
val_df["encodedCategory"] = y_test

print(f"\nTrain size: {len(train_df)} | Validation size: {len(val_df)}")

# Save CSV
train_csv = OUT_DATA / "trainingDataset.csv"
val_csv   = OUT_DATA / "validationDataset.csv"


train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)

print(f"Saved:\n - {train_csv}\n - {val_csv}\n")
print("Done.")
