import pandas as pd


SHARED_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "attack_category", "source"
]

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
    out = out.dropna(subset=["attack_category"])

    return out
    


def clean_nb15():
    nb15_df = 