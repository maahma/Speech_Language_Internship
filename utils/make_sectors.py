import pandas as pd
import os

NASDAQ_CSV = os.path.join("/users/ac4ma/Speech_Language_Internship/feature_extraction/nasdaq_screener_1754902542929.csv")
OUTPUT_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/Sectors")

nasdaq_df = pd.read_csv(NASDAQ_CSV)

os.makedirs(OUTPUT_DIR, exist_ok=True)

sectors = nasdaq_df["Sector"].dropna().unique()

for sector in sectors:
    folder_path = os.path.join(OUTPUT_DIR, sector.strip())
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created folder: {folder_path}")

