from sentence_transformers import SentenceTransformer
import os
import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

model = SentenceTransformer('all-MiniLM-L6-v2')

RECO_DIR = os.path.join("/mnt/parscratch/users/ac4ma/All_Recordings")
OUTPUT_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/feature_extraction", "features_text_sbert")
os.makedirs(OUTPUT_DIR, exist_ok=True)


for company in sorted(os.listdir(RECO_DIR)):
    text_dir = os.path.join(RECO_DIR, company, "sent_token")

    if not os.path.exists(text_dir):
        continue

    logging.info(f"Processing {company}")

    company_out_dir = os.path.join(OUTPUT_DIR, company)
    os.makedirs(company_out_dir, exist_ok=True)

    sentences = []
    for text_file in sorted(os.listdir(text_dir)):
        input_path = os.path.join(text_dir, text_file)

        with open(input_path, 'r', encoding='utf-8') as f:
            sentence = f.read().strip()
            if sentence:
                sentences.append(sentence)

    if not sentences:
        continue
    
    sentence_embeddings = model.encode(sentences)

    pd.DataFrame(sentence_embeddings).to_csv(os.path.join(company_out_dir, 'sbert_embeddings.csv'), index=False)
    pd.DataFrame({'sentence': sentences}).to_csv(os.path.join(company_out_dir, 'sentences.csv'), index=False)