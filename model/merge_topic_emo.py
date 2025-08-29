import os
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

COMPANY_EMO_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/model/emo_label")
COMPANY_TOPIC_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/feature_extraction/company_topics")
OUTPUT_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/model/merge_topic_emo")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for company in sorted(os.listdir(COMPANY_TOPIC_DIR)):
    emo_info_file = os.path.join(COMPANY_EMO_DIR, f"{company}_emotion_results_with_labels.csv")
    topic_info_file = os.path.join(COMPANY_TOPIC_DIR, company, "segment_topics.csv")

    topic_df = pd.read_csv(topic_info_file)
    emo_df = pd.read_csv(emo_info_file)

    topic_df["file_id"] = topic_df["filename"].str.replace(".txt", "", regex=False)
    emo_df["file_id"] = emo_df["filename"].str.replace(".wav", "", regex=False)

    merged_df = pd.merge(topic_df, emo_df, on="file_id", how="inner")

    # Group by topic and emotion
    topic_emotions = (
        merged_df.groupby(["topic_id", "topic_keywords", "emotion_label"])
        .size()
        .reset_index(name="count")
    )

    merged_out = os.path.join(OUTPUT_DIR, f"{company}_topic_emotion_merged.csv")
    grouped_out = os.path.join(OUTPUT_DIR, f"{company}_topic_emotion_summary.csv")

    merged_df.to_csv(merged_out, index=False)
    topic_emotions.to_csv(grouped_out, index=False)

    logging.info(f"Saved merged file: {merged_out}")
    logging.info(f"Saved summary file: {grouped_out}")
