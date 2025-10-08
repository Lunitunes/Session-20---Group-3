import pandas as pd
import numpy as np
import sys
from pathlib import Path

SHARED_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes"
]

ALL_COLUMNS = SHARED_COLUMNS + ["category", "source"]

# BASIC_TO_UNIFIED = {
#     "Normal": "Normal",
#     "DoS": "DoS",
#     "Probe": "Probe",
#     "R2L": "Exploit",
#     "U2R": "Exploit",
# }

def clean_basic():
    #READ THE CSV/DATASET
    basic_df = pd.read_csv(r"raw_data/basic_data_4.csv").drop_duplicates()

    #FILL MISSING VALUES, AVARES THM MEIDAN AND MODE
    basic_df.fillna(basic_df.median(numeric_only=True), inplace=True)
    basic_df.fillna(basic_df.mode().iloc[0], inplace=True)

    #STANDARDISE COLUMN NAMES
    basic_df.columns = (
        pd.Series(basic_df.columns)
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", "", regex=True)
            .str.lower())

    #READ THE LABEL CATEGORY
    label_category_map = pd.read_csv(r"raw_data/label_category_map.csv")
 
    #MAP THE LABELS TO CATEGORIES
    basic_df['category'] = basic_df['label'].map(dict(zip(label_category_map['label'], label_category_map['category'])))

    # basic_df["attack_category"] = (
    #     basic_df["category"].astype(str).str.strip().str.lower().map(BASIC_TO_UNIFIED)
    # )

    #CHECK ALL COLUMNS ARE THERE
    missing = [c for c in SHARED_COLUMNS if c not in basic_df.columns]
    if missing:
        raise ValueError(f"[basic] missing columns: {missing}")
    

    out = basic_df[SHARED_COLUMNS].copy()
    out["source"] = "basic"

    #STANDARDISE DATA
    out["service"] = out["service"].astype(str).str.strip().str.lower()
    out["protocol_type"] = out["protocol_type"].astype(str).str.strip().str.lower()
    out["flag"] = out["flag"].astype(str).str.strip().str.lower()

    # Drop rows with unknown class
    out["category"] = basic_df["category"]
    out = out.dropna(subset=["category"])
    return out[ALL_COLUMNS]
    


def clean_nb15():
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
      nb15_df = pd.read_csv(RAW_DATA / RAW_FILE, header=None, low_memory=False)
      nb15_df.columns = feature_names[: nb15_df.shape[1]]
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
  nb15_df.columns = (
      pd.Series(nb15_df.columns)
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.lower()
  )

  # Fill attack_cat for normal rows (Label=0)
  if "label" not in nb15_df.columns or "attack_cat" not in nb15_df.columns:
      print("Expected 'label' and 'attack_cat' columns not found. Last columns were:",
            nb15_df.columns[-8:].tolist())
      sys.exit(1)

  normal_before = nb15_df["attack_cat"].isna().sum()
  nb15_df.loc[nb15_df["label"] == 0, "attack_cat"] = "Normal"
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

  # after filling normal rows
  nb15_df.loc[nb15_df["label"] == 0, "attack_cat"] = "Normal"
  ac = nb15_df["attack_cat"].astype(str).str.strip()


  # rename to shared schema
  nb15_df = nb15_df.rename(columns={
      "dur": "duration",
      "proto": "protocol_type",
      "state": "flag",
      "sbytes": "src_bytes",
      "dbytes": "dst_bytes",
  })

  #CHECK ALL COLUMNS ARE THERE
  missing = [c for c in SHARED_COLUMNS if c not in nb15_df.columns]
  if missing:
      raise ValueError(f"[nb15] missing columns: {missing}")
  
  # value hygiene
  nb15_df["service"] = nb15_df["service"].astype(str).str.strip().str.lower().replace({"-": "unknown"})
  nb15_df["protocol_type"] = nb15_df["protocol_type"].astype(str).str.strip().str.lower()
  nb15_df["flag"] = nb15_df["flag"].astype(str).str.strip().str.upper()

  # select shared columns + original category
  out = nb15_df[SHARED_COLUMNS].copy()
  out["category"] = ac
  out["source"] = "nb15"

  # drop rows with unknown class
  out = out.dropna(subset=["category"])
  return out[ALL_COLUMNS]


