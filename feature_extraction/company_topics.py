import os
import re
import logging
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from cleantext import clean
from bertopic.representation import PartOfSpeech
from bertopic import BERTopic
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import umap.umap_ as umap
from umap import UMAP
from sentence_transformers import SentenceTransformer
import spacy
import hdbscan

# ---------- config ----------
RECO_DIR = os.path.join("/mnt/parscratch/users/ac4ma/All_Recordings")
OUTPUT_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/feature_extraction/company_topics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm") 

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

# ---------- helpers ----------
def cleaning(text):
    cleaned_text = clean(
        str(text),
        lower=True,
        no_numbers=True,
        no_digits=True,
        no_currency_symbols=True,
        no_punct=True,
        replace_with_punct="",
        replace_with_number="",
        replace_with_digit="",
        replace_with_currency_symbol=""
    )

    # cleaned_text = " ".join(line.strip() for line in cleaned_text.split("\n") if line.strip())
    return cleaned_text

def return_vectorizer_model():
    standard = list(stopwords.words('english'))
    additional = [
        'let', 'us','like','say','would','also','th','need','afternoon','morning','evening',
        'ladies','gentleman','foremost','colleagues','friends','years','team','ago','last',
        'year','thanks','thank you','appreciate','good afternoon','good morning',
        'good evening','thanks everyone','welcome','pleased to','nice to','operator',
        'conference','call','participants','let me','we will','i will','i would',
        'you know','kind of','ahead','youre','weve','yeah','hi','hey','im','youve',
        'theres','thats','theyre','youll','please','actually',
        'officer', 'executive', 'vice', 'president', 'securities', 'agricole', 'guy',
        'ph', 'everyone', 'quarter', 'first', 'second', 'third', 'fourth', 'bit', 'currency',
        'little bit'
    ]
    full = standard + additional
    return CountVectorizer(ngram_range=(2,2), stop_words=full)

def format_topic_words(topic_words):
    return ", ".join([w for w, _ in topic_words])

# NEW ------------------------------------
def get_topic_keywords(topic_id):
    if topic_id == -1:
        return "Outlier"
    words = topic_model.get_topic(topic_id)
    return ", ".join([w for w, _ in words])
# NEW ------------------------------------


# ---------- representation (POS) ----------
pos_patterns = [
    [{'POS': 'ADJ'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN'}],
    [{'POS': 'ADJ'}]
]

representation_model = PartOfSpeech("en_core_web_sm", pos_patterns=pos_patterns)
vectorizer_model = return_vectorizer_model()

for company in sorted(os.listdir(RECO_DIR)):
    text_dir = os.path.join(RECO_DIR, company, "sent_token")

    if not os.path.isdir(text_dir):
        logging.warning(f"Missing folders for {company}, skipping...")
        continue

    logging.info(f"Processing company: {company}")

    # Collect all text segments
    segment_texts = []
    segment_filenames = []

    for fname in sorted(os.listdir(text_dir)):
        with open(os.path.join(text_dir, fname), "r", encoding="utf-8") as f:
            segment_texts.append(f.read().strip())
        segment_filenames.append(fname)

    if len(segment_texts) < 5:
        logging.warning(f"Too few segments for {company}, skipping BERTopic.")
        continue
    
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        representation_model=representation_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        calculate_probabilities=True
    )

    # Run topic modeling on all segments for this company
    topics, probs = topic_model.fit_transform(segment_texts)
    topic_info = topic_model.get_topic_info()

    # NEW ------------------------------------
    df_topics = pd.DataFrame({
        'filename': segment_filenames,
        'topic_id': topics
    })

    df_topics['topic_keywords'] = df_topics['topic_id'].apply(get_topic_keywords)
    # NEW ------------------------------------

    # Create output folder for this sector
    output_company = os.path.join(OUTPUT_DIR, company)
    os.makedirs(output_company, exist_ok=True)

    # Save overview CSV
    overview_path = os.path.join(output_company, "topic_info.csv")
    topic_info.to_csv(overview_path, index=False, encoding="utf-8")
    logging.info(f"Topics overview saved to {overview_path}")


    # NEW ------------------------------------
    # Save topic assignment per segment CSV
    segment_topic_path = os.path.join(output_company, "segment_topics.csv")
    df_topics.to_csv(segment_topic_path, index=False, encoding="utf-8")
    logging.info(f"Segment topic assignments saved to {segment_topic_path}")
    # NEW ------------------------------------

    # Save words for each topic separately
    for topic_id in topic_info['Topic']:
        words_df = pd.DataFrame(topic_model.get_topic(topic_id), columns=["Word", "Probability"])
        words_path = os.path.join(output_company, f"topic_{topic_id}_words.csv")
        words_df.to_csv(words_path, index=False, encoding="utf-8")
        logging.info(f"Saved words for topic {topic_id} to {words_path}")
