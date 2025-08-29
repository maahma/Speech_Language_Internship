import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

COMPANY_EMO_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/model/emo_classifier_result")
OUTPUT_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/model/normalized_and_fused")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_valence(v):
    """Robust z-score + tanh squash → [-1, 1]."""
    v_med = np.median(v)
    v_mad = np.median(np.abs(v - v_med))
    v_mad = v_mad if v_mad > 1e-8 else 1e-8
    return np.tanh((v - v_med) / (1.4826 * v_mad))


def normalize_arousal(a):
    """Min-max normalize → [0, 1]."""
    a_min, a_max = float(np.min(a)), float(np.max(a))
    span = (a_max - a_min) if (a_max - a_min) > 1e-8 else 1e-8
    return (a - a_min) / span


for company in sorted(os.listdir(COMPANY_EMO_DIR)):
    emo_info_csv = os.path.join(COMPANY_EMO_DIR, company, f"{company}_emotion_results.csv")
    df = pd.read_csv(emo_info_csv)

    logging.info(f"Processing {company} with {len(df)} rows")

    # --- Normalize Speech ---
    df["speech_valence_norm"] = normalize_valence(df["speech_valence"].values)
    df["speech_arousal_norm"] = normalize_arousal(df["speech_arousal"].values)

    # --- Normalize Text ---
    df["text_valence_norm"] = normalize_valence(df["text_valence"].values)
    df["text_arousal_norm"] = normalize_arousal(df["text_arousal"].values)

    # --- Fuse ---
    speech_w = 0.5   # 50% weight to speech
    text_w = 0.5     # 50% weight to text

    df["fused_valence"] = (speech_w * df["speech_valence_norm"] + text_w * df["text_valence_norm"]) / (speech_w + text_w)
    df["fused_arousal"] = (speech_w * df["speech_arousal_norm"] + text_w * df["text_arousal_norm"]) / (speech_w + text_w)


    # Save normalized + fused results
    out_csv = os.path.join(OUTPUT_DIR, f"{company}_emotion_results_normalized.csv")
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved normalized + fused results → {out_csv}")
