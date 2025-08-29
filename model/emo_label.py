import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

COMPANY_EMO_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/model/normalized_and_fused")
OUTPUT_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/model/emo_label")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# Precomputed thresholds
# =====================
# With 10,90 dist
# valence_low = -0.658
# valence_high = 0.677
# arousal_low = 0.443
# arousal_high = 0.843

# With 25, 75 dist
valence_low = -0.352
valence_high = 0.344
arousal_low = 0.555
arousal_high = 0.753

logging.info(f"Using thresholds: V=({valence_low}, {valence_high}), A=({arousal_low}, {arousal_high})")


# =====================
# Functions for emotion detection according to the circumplex model of affect
# =====================
def va_bin(v, a):
    """Return valence bin and arousal bin strings (neg/neu/pos and low/mid/high)."""
    if v <= valence_low:
        v_bin = "neg"
    elif v >= valence_high:
        v_bin = "pos"
    else:
        v_bin = "neu"

    if a <= arousal_low:
        a_bin = "low"
    elif a >= arousal_high:
        a_bin = "high"
    else:
        a_bin = "mid"

    return v_bin, a_bin

def classify_emotion_from_fused(v, a, speech_v=None, speech_a=None, text_v=None, text_a=None): #, require_modal_agreement=False
    """
    Classify into nervous, excited, neutral, calm, sad, other.
    """
    v_bin, a_bin = va_bin(v, a)


    if v_bin == "pos" and a_bin == "high":
        return "Excited (Positive, High arousal)"
    if v_bin == "neg" and a_bin == "high":
        return "Nervous (Negative, High arousal)"
    if v_bin == "pos" and a_bin != "high":
        return "Positive (Low/Mid arousal)"
    if v_bin == "neg" and a_bin != "high":
        return "Negative (Low/Mid arousal)"
    return "Neutral"

# =====================
# Process all companies
# =====================
all_rows = []  # (company, filename, fused_valence, fused_arousal, label)
for company in sorted(os.listdir(COMPANY_EMO_DIR)):
    emo_info_csv = os.path.join(COMPANY_EMO_DIR, company)
    df = pd.read_csv(emo_info_csv)

    df["emotion_label"] = df.apply(
        lambda r: classify_emotion_from_fused(r["fused_valence"], r["fused_arousal"]),
        axis=1
    )

    # Save per-company file
    out_path = os.path.join(OUTPUT_DIR, f"{company}_emotion_results_with_labels.csv")
    df.to_csv(out_path, index=False)
    logging.info(f"Saved labeled results for {company}: {out_path}")

    # Collect for master file
    for _, r in df.iterrows():
        all_rows.append((company, r["filename"], float(r["fused_valence"]), float(r["fused_arousal"]), r["emotion_label"]))

# =====================
# Save master file
# =====================
master = pd.DataFrame(all_rows, columns=["company","filename","fused_valence","fused_arousal","emotion_label"])
master_out = os.path.join(OUTPUT_DIR, "master_emotions.csv")
master.to_csv(master_out, index=False)
logging.info(f"Saved master: {master_out}")
