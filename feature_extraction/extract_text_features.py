import os
import numpy as np
import pandas as pd
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

RECO_DIR = os.path.join("/mnt/parscratch/users/ac4ma/All_Recordings")
OUTPUT_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/feature_extraction", "features_text_glove")
GLOVE_PATH = "../glove/glove.6B.100d.txt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

glove_embeddings = {}
with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        vector = np.array(parts[1:], dtype=np.float32)
        glove_embeddings[word] = vector
logging.info(f"Loaded {len(glove_embeddings)} word vectors.")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) 
    return text.strip().split()


for company in os.listdir(RECO_DIR):
    text_dir = os.path.join(RECO_DIR, company, "sent_token")
    
    if not os.path.exists(text_dir):
        continue 
        
    company_out_dir = os.path.join(OUTPUT_DIR, company)   
    os.makedirs(company_out_dir, exist_ok=True)

    for text_file in os.listdir(text_dir):
        input_path = os.path.join(text_dir, text_file)
        output_csv = os.path.join(company_out_dir, text_file.replace(".txt", ".csv"))
        
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()
            tokens = preprocess(text)

            vectors = [glove_embeddings[word] for word in tokens if word in glove_embeddings]
            if vectors:
                sentence_embedding = np.mean(vectors, axis=0)  
            else:
                sentence_embedding = np.zeros(100)  

            df = pd.DataFrame([sentence_embedding])
            df.to_csv(output_csv, index=False)
        except Exception as e:
            logging.info(f"Failed to process {text_file}: {e}")