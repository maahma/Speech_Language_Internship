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
OUTPUT_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/results/topic_emo_dist")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in sorted(os.listdir(TOPIC_EMO)):
    if not fname.endswith("_topic_emotion_merged.csv"):
        continue 
    
    topic_emo_file = os.path.join(TOPIC_EMO, fname)
    df = pd.read_csv(topic_emo_file)

    topic_emotions = (
        df.groupby(["topic_id", "topic_keywords", "emotion_label"])
        .size()
        .reset_index(name="count")
    )

    for topic_id, group in topic_emotions.groupby("topic_id"):
        keywords = group["topic_keywords"].iloc[0][:80] 
        emotion_counts = group.set_index("emotion_label")["count"]

        plt.figure(figsize=(6,4))
        emotion_counts.plot(kind="bar")
        plt.title(f"Topic {topic_id}\n{keywords}", pad=30)
        plt.ylabel("Count of Emotion Labels")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        out_file = os.path.join(
            OUTPUT_DIR,
            f"{os.path.splitext(fname)[0]}_topic{topic_id}_emo_dist.png"
        )
        plt.savefig(out_file, bbox_inches="tight")
        plt.close()

    logging.info(f"Saved topic-level emotion charts for {fname}")
