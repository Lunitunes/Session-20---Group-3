import pandas as pd
from pathlib import Path

from clean_basic import clean_basic, clean_nb15

#Paths
BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / "processed_data" / "merged_shared.csv"

basic_df = clean_basic()
nb15_df = clean_nb15()

merged_df = pd.concat([basic_df, nb15_df], ignore_index=True).drop_duplicates()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
merged_df.to_csv(OUT_PATH, index=False)