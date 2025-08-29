import os
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

TOPIC_EMO = os.path.join("/users/ac4ma/Speech_Language_Internship/model/merge_topic_emo")
OUTPUT_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/model/topic_emo_label")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in sorted(os.listdir(TOPIC_EMO)):
    topic_emo_file = os.path.join(TOPIC_EMO, fname)
    df = pd.read_csv(topic_emo_file)

    df["topic_keywords"] = df["topic_keywords"].fillna("").apply(lambda x: [kw.strip() for kw in x.split(",") if kw.strip()])

    df_exploded = df.explode("topic_keywords")

    df_keywords = df_exploded[["topic_keywords", "emotion_label"]]

    keyword_emotions = df_keywords.groupby("topic_keywords")["emotion_label"].value_counts().unstack(fill_value=0)

    out_file = os.path.join(OUTPUT_DIR, fname.replace("_topic_emotion_merged.csv", ""))
    keyword_emotions.to_csv(out_file)
    logging.info(f"Saved keyword-emotion counts to {out_file}")
