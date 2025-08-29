import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
import ast

TOPICS_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/feature_extraction/company_topics")
OUTPUT_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/feature_extraction/plot_company_topics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

for company in sorted(os.listdir(TOPICS_DIR)):
    topic_info_csv = os.path.join(TOPICS_DIR, company, "topic_info.csv")
    if not os.path.isfile(topic_info_csv):
        logging.warning(f"Missing topic_info.csv for {company}, skipping...")
        continue

    logging.info(f"Processing company: {company}")

    df = pd.read_csv(topic_info_csv)
    df = df[df['Topic'] != -1]

    df = df.sort_values("Topic")

    df['Keywords'] = df['Representation'].apply(lambda x: ast.literal_eval(x)[:3])
    df['Label'] = df['Keywords'].apply(lambda kws: ", ".join(kws))

    company_output_dir = os.path.join(OUTPUT_DIR, company)
    os.makedirs(company_output_dir, exist_ok=True)
    output_plot = os.path.join(company_output_dir, "topics_bar_chart_horizontal.png")

    plt.figure(figsize=(16, 6))
    bars = plt.barh(df['Topic'].astype(str), df['Count'], height=0.5, color='forestgreen', edgecolor='black')

    plt.xlabel("Number of sentences")
    plt.ylabel("Topic ID")
    plt.title("Distribution of Topics with Top Keywords", fontsize=14)
    plt.gca().invert_yaxis()  # optional: largest topic at top

    for bar, keywords in zip(bars, df['Label']):
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, keywords, 
                 va='center', ha='left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()
    logging.info(f"Saved plot to {output_plot}")