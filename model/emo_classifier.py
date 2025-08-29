import os
import librosa
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

# ========================
# SPEECH EMOTION VA VALUES
# ========================
# From the research poster:
# Unsupervised Arousal Valence Estimation from Speech and Corresponding Discrete Emotion by Enting Zhou 
# Show discrete emotion mapping
SPEECH_EMOTION_DISTRIBUTIONS = {
    "angry":     {"valence": -0.43, "arousal": 0.67},
    "boredom":   {"valence": -0.65, "arousal": -0.62},
    "contempt":  {"valence": -0.80, "arousal": 0.20},
    "disgust":   {"valence": -0.60, "arousal": 0.35},
    "fear":      {"valence": -0.64, "arousal": 0.35},
    "happy":     {"valence": 0.76, "arousal": 0.48},
    "neutral":   {"valence": 0.00, "arousal": 0.00},
    "sad":       {"valence": -0.63, "arousal": -0.27},
    "surprised": {"valence": 0.00, "arousal": 0.60},
}

# ========================
# TEXT EMOTION VA RANGES
# ========================
# From the research paper:
# Emotional valence and arousal affect reading in an interactive way: Neuroimaging evidence for an approach-withdrawal framework
# These represent the statistical ranges for different VA quadrants
TEXT_EMOTION_DISTRIBUTIONS = {
    "positive_high_arousal": {
        "valence_range": (1.01, 2.52),
        "arousal_range": (4.00, 5.35),
        "mean_valence": 1.92,
        "mean_arousal": 4.45,
    },
    "negative_high_arousal": {
        "valence_range": (-2.61, -1.17),
        "arousal_range": (4.06, 5.41),
        "mean_valence": -1.77,
        "mean_arousal": 4.60,
    },
    "positive_low_arousal": {
        "valence_range": (1.04, 1.90),
        "arousal_range": (2.59, 3.88),
        "mean_valence": 1.46,
        "mean_arousal": 3.41,
    },
    "negative_low_arousal": {
        "valence_range": (-2.02, -0.89),
        "arousal_range": (2.24, 4.42),
        "mean_valence": -1.33,
        "mean_arousal": 3.52,
    },
    "neutral_low_arousal": {
        "valence_range": (-0.85, 0.85),
        "arousal_range": (2.79, 4.15),
        "mean_valence": 0.19,
        "mean_arousal": 3.32,
    }
}

# From the Circumplex Model of Affect
TEXT_EMOTION_CATEGORY_MAP = {
    **{emo: "positive_high_arousal" for emo in [
        "excitement", "joy", "amusement", "pride", "optimism", "approval", "gratitude", "admiration"
    ]},
    **{emo: "negative_high_arousal" for emo in [
        "anger", "disgust", "fear", "nervousness", "embarrassment"
    ]},
    **{emo: "positive_low_arousal" for emo in [
        "love", "relief", "caring"
    ]},
    **{emo: "negative_low_arousal" for emo in [
        "sadness", "disappointment", "grief", "remorse", "disapproval"
    ]},
    **{emo: "neutral_low_arousal" for emo in [
        "neutral", "confusion", "curiosity", "desire", "realization"
    ]}
}

# ========================
# MODELS
# ========================
speech_model = pipeline(
    task="audio-classification",
    model="superb/hubert-large-superb-er",
    top_k=None
)

text_model_name = "bhadresh-savani/bert-base-go-emotion"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModelForSequenceClassification.from_pretrained(text_model_name)
text_labels = text_model.config.id2label

# ========================
# FUNCTIONS
# ========================
# To get valence, arousal pair of a text emotion label
def get_text_va(emotion):
    category = TEXT_EMOTION_CATEGORY_MAP.get(emotion.lower())
    if not category:
        return {"valence": 0.0, "arousal": 0.0}
    dist = TEXT_EMOTION_DISTRIBUTIONS[category]
    return {"valence": dist["mean_valence"], "arousal": dist["mean_arousal"]}

# Thisjust computes the probability-weighted average of speech arousal and valence
# Uses the expectation formula to do this : sum(probability from the speech model X valence/arousal)
def expected_va_speech(speech_probs):
    v_sum, a_sum = 0.0, 0.0
    for item in speech_probs:
        emo = item["label"].lower()
        prob = item["score"]
        va = SPEECH_EMOTION_DISTRIBUTIONS.get(emo, {"valence": 0.0, "arousal": 0.0})
        v_sum += prob * va["valence"]
        a_sum += prob * va["arousal"]
    return v_sum, a_sum

# Same probability-weighted average computation
# gets probability i.e. logits from the text model 
# multiplies by VA
def expected_va_text(text):
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = text_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    v_sum, a_sum = 0.0, 0.0
    for idx, prob in enumerate(probs):
        emo = text_labels[idx].lower()
        va = get_text_va(emo)
        v_sum += prob * va["valence"]
        a_sum += prob * va["arousal"]
    return v_sum, a_sum


# ========================
# MAIN PROCESSING
# ========================
RECO_DIR = os.path.join("/mnt/parscratch/users/ac4ma/All_Recordings")
OUTPUT_MODEL = os.path.join("/users/ac4ma/Speech_Language_Internship/model/emo_classifier_result")
# os.makedirs(OUTPUT_MODEL, exist_ok=True)


for company in sorted(os.listdir(RECO_DIR)):
    text_dir = os.path.join(RECO_DIR, company, "sent_token")
    audio_dir = os.path.join(RECO_DIR, company, "audio_wav")
    company_output_dir = os.path.join(OUTPUT_MODEL, company)
    output_csv = os.path.join(company_output_dir, f"{company}_emotion_results.csv")

    if os.path.exists(company_output_dir) and os.path.exists(output_csv): 
        logging.info(f"Results already exist for {company}, skipping...")
        continue

    if not os.path.isdir(text_dir) or not os.path.isdir(audio_dir):
        logging.warning(f"Missing folders for {company}, skipping...")
        continue

    logging.info(f"Processing company: {company}")
    results = []

    for filename in sorted(os.listdir(audio_dir)):
        if not filename.endswith(".wav"):
            continue

        audio_path = os.path.join(audio_dir, filename)
        text_path = os.path.join(text_dir, filename.replace(".wav", ".txt"))
        
        if not os.path.exists(text_path):
            logging.warning(f"Missing transcript for {filename}, skipping...")
            continue

        with open(text_path, "r", encoding="utf-8") as f:
            text_content = f.read().strip()
            
        try:
            speech_probs = speech_model(audio_path)

            # Speech VA
            v_s, a_s = expected_va_speech(speech_probs)

        except ValueError as e:
            logging.error(f"Skipping corrupt or unreadable file {filename}: {e}")
            continue

        # Text VA
        v_t, a_t = expected_va_text(text_content)

        # Fuse VA
        # gets the expectation for text and speech
        # multiplies by weight which I will experiment with after I see the result - REMEMBER TO DO THIS PLEASE
        # gets the fused valence and arousal of text and speech - AGAIN CHECK RESULTS AND MAKE CHANGES 
        # saves fused values in the CSV
        # fused_valence = 0.5 * v_s + 0.5 * v_t
        # fused_arousal = 0.5 * a_s + 0.5 * a_t

        results.append({
            "filename": filename,
            "speech_valence": round(v_s, 4),
            "speech_arousal": round(a_s, 4),
            "text_valence": round(v_t, 4),
            "text_arousal": round(a_t, 4),
            # "fused_valence": round(fused_valence, 4),
            # "fused_arousal": round(fused_arousal, 4),
        })

    os.makedirs(company_output_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False)
    logging.info(f"Saved raw VA results to {output_csv}")