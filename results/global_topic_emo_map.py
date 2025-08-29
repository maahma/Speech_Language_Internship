import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TOPIC_EMO = os.path.join("/users/ac4ma/Speech_Language_Internship/model/merge_topic_emo")
OUTPUT_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/results/2_global_topic_emo")
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_dfs = []
for fname in sorted(os.listdir(TOPIC_EMO)):
    if fname.endswith(".csv") and not fname.endswith("_topic_emotion_summary.csv"):
        df = pd.read_csv(os.path.join(TOPIC_EMO, fname))
        df["topic_label"] = df["topic_keywords"].str.split(",").str[0].str.strip()
        df = df[df["emotion_label"] != "Neutral"]        
        all_dfs.append(df[["topic_label", "emotion_label"]])

global_df = pd.concat(all_dfs, ignore_index=True)
top_topics = global_df["topic_label"].value_counts().head(50).index
filtered = global_df[global_df["topic_label"].isin(top_topics)]

topic_emo_counts = (
    filtered.groupby(["topic_label", "emotion_label"])
    .size()
    .unstack(fill_value=0)
)

topic_emo_pct = topic_emo_counts.div(topic_emo_counts.sum(axis=1), axis=0) * 100


top_topics_per_emotion = []
for emo in topic_emo_pct.columns:
    top_topics_per_emotion.extend(topic_emo_pct[emo].nlargest(30).index.tolist())

unique_topics = list(set(top_topics_per_emotion))
filtered_pct = topic_emo_pct.loc[unique_topics]

filtered_pct = filtered_pct.sort_index()

col_normed = filtered_pct.copy()
for col in col_normed.columns:
    min_val = col_normed[col].min()
    max_val = col_normed[col].max()
    if max_val > min_val:  
        col_normed[col] = (col_normed[col] - min_val) / (max_val - min_val)

original_labels = col_normed.columns

short_labels = [label.split()[0] for label in original_labels]

plt.figure(figsize=(12, 8))
sns.heatmap(
    col_normed,
    annot=filtered_pct,
    fmt=".1f",
    cmap="coolwarm",
    cbar=False,
    annot_kws={"size": 10, "color": "black"}  
)
plt.title("Top Topic-Level Emotion Distribution", fontsize=20)
plt.xlabel("Emotion", fontsize=14)
plt.ylabel("Topic", fontsize=14)
plt.xticks(ticks=[i + 0.5 for i in range(len(short_labels))], labels=short_labels, fontsize=12, rotation=0)

plt.yticks(fontsize=12, rotation=0)
plt.tight_layout()
out_file = os.path.join(OUTPUT_DIR, "global_topic_emotion_heatmap_colnorm.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved global topic-emotion heatmap to {os.path.join(OUTPUT_DIR, 'global_topic_emotion_heatmap.png')}")
print(f"Saved column-normalized heatmap to {out_file}")