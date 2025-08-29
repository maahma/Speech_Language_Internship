import pandas as pd
import os

SECTOR_DIR = "/users/ac4ma/Speech_Language_Internship/Sectors"

for sector_name in os.listdir(SECTOR_DIR):
    sector_path = os.path.join(SECTOR_DIR, sector_name)

    data_rows = []

    for filename in os.listdir(sector_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(sector_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            data_rows.append({
                "sector": sector_name,
                "filename": filename,
                "text": text
            })

    df = pd.DataFrame(data_rows)
    csv_path = os.path.join(sector_path, f"{sector_name}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")