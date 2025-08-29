import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

COMPANY_EMO_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/model/normalized_and_fused")
OUTPUT_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/model/threshold_result_25_75")
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_valences = []
all_arousals = []

for company in sorted(os.listdir(COMPANY_EMO_DIR)):
    emo_info_csv = os.path.join(COMPANY_EMO_DIR, company)
    df = pd.read_csv(emo_info_csv)

    all_valences.extend(df["fused_valence"].tolist())
    all_arousals.extend(df["fused_arousal"].tolist())

all_valences = np.array(all_valences)
all_arousals = np.array(all_arousals)

# ========================
# THRESHOLD CALCULATION (25, 75)
# ========================
low_thresh_v = np.percentile(all_valences, 25)
high_thresh_v = np.percentile(all_valences, 75)
low_thresh_a = np.percentile(all_arousals, 25)
high_thresh_a = np.percentile(all_arousals, 75)

thresholds = (low_thresh_v, high_thresh_v, low_thresh_a, high_thresh_a)

logging.info(f"Valence thresholds: low={low_thresh_v:.3f}, high={high_thresh_v:.3f}")
logging.info(f"Arousal thresholds: low={low_thresh_a:.3f}, high={high_thresh_a:.3f}")

# ========================
# HISTOGRAMS
# ========================
# --- Valence ---
plt.hist(all_valences, bins=30, alpha=0.7, color="#FFB6C1") 
plt.axvline(low_thresh_v, color='red', linestyle='--', label='Low threshold (25th %)')
plt.axvline(high_thresh_v, color='green', linestyle='--', label='High threshold (75th %)')
plt.legend(fontsize=10)
plt.title("Valence Distribution", fontsize=16)
plt.xlabel("Valence Scale", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.locator_params(axis='x', nbins=5)
plt.savefig(os.path.join(OUTPUT_DIR, "valence_hist.png"))
plt.close()

# --- Arousal ---
plt.hist(all_arousals, bins=30, alpha=0.7, color="#B4E7B0")  
plt.axvline(low_thresh_a, color='red', linestyle='--', label='Low threshold (25th %)')
plt.axvline(high_thresh_a, color='green', linestyle='--', label='High threshold (75th %)')
plt.legend(fontsize=12)
plt.title("Arousal Distribution", fontsize=16)
plt.xlabel("Arousal Scale", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.xticks(fontsize=12)  
plt.yticks(fontsize=12)
plt.savefig(os.path.join(OUTPUT_DIR, "arousal_hist.png"))
plt.close()