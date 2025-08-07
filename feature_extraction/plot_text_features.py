from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

features_path = "/users/ac4ma/Speech_Language_Internship/feature_extraction/features_text_sbert"
plots_folder = "/users/ac4ma/Speech_Language_Internship/feature_extraction/plots_sbert_embeddings"
os.makedirs(plots_folder, exist_ok=True)

for company in os.listdir(features_path):
    company_path = os.path.join(features_path, company)
    logging.info(f"Processing {company}")

    embeddings_file = os.path.join(company_path, "sbert_embeddings.csv")
    embeddings = pd.read_csv(embeddings_file).values

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Optional clustering for coloring
    kmeans = KMeans(n_clusters=10, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='tab10', s=15)
    plt.title(f"t-SNE for {company}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, f"{company}_tsne.png"))
    plt.close()
