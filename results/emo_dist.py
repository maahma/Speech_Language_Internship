import os
import pandas as pd
import logging
import matplotlib.pyplot as plt
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

TOPIC_EMO = os.path.join("/users/ac4ma/Speech_Language_Internship/model/merge_topic_emo")
OUTPUT_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/results/2_emo_dist")
os.makedirs(OUTPUT_DIR, exist_ok=True)

custom_colors = {
    "Neutral": "lightgrey",
    "Excited (Positive, High arousal)": "salmon",
    "Positive (Low/Mid arousal)": "lightsalmon",
    "Negative (Low/Med arousal)": "lightblue",
    "Nervous (Negative, High arousal)": "steelblue"
}

for fname in sorted(os.listdir(TOPIC_EMO)):
    if not fname.endswith("_topic_emotion_merged.csv"):
        continue 

    topic_emo_file = os.path.join(TOPIC_EMO, fname)
    df = pd.read_csv(topic_emo_file)

    emotion_counts = df["emotion_label"].value_counts()
    emotion_pct = emotion_counts / emotion_counts.sum() * 100

    colors = [custom_colors.get(e, "lightgrey") for e in emotion_pct.index]

    used_keywords = set()  
    emotion_keywords = {}
    for emo in emotion_counts.index:
        emo_rows = df[df["emotion_label"] == emo]

        all_kw = []
        for kws in emo_rows["topic_keywords"].dropna():
            if kws.lower() == "outlier":
                continue
            parts = [k.strip() for k in kws.split(",") if k.strip()]
            all_kw.extend(parts)

        if all_kw:
            counter = Counter(all_kw)
            unique_kw = [(w, c) for w, c in counter.most_common() if w not in used_keywords]
            top2 = [w for w, _ in unique_kw[:2]]
            emotion_keywords[emo] = top2
            used_keywords.update(top2)
        else:
            emotion_keywords[emo] = []

    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        emotion_pct,
        labels=None,
        autopct="%.1f%%",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 14}  
    )

    name_no_ext = os.path.splitext(fname)[0]
    base = name_no_ext.split("_")[0]
    plt.title(f"Overall Emotion Distribution for {base}", fontsize=18)  

    legend_labels = []
    for emo in emotion_pct.index:
        kw_str = ", ".join(emotion_keywords[emo]) if emotion_keywords[emo] else "No keywords"
        legend_labels.append(f"{emo}: {kw_str}")
    plt.legend(
        wedges, 
        legend_labels, 
        title="Emotions & Top Keywords", 
        title_fontsize=14,
        fontsize=12,  
        bbox_to_anchor=(1.05, 1), 
        loc="upper left"
    )

    out_file = os.path.join(OUTPUT_DIR, fname.replace("_topic_emotion_merged.csv", "_emo_dist.png"))
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()

    logging.info(f"Saved emotion distribution chart to {out_file}")

