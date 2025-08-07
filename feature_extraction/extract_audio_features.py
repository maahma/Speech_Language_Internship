import os
import opensmile
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

RECO_DIR = os.path.join("/mnt/parscratch/users/ac4ma/All_Recordings")
OUTPUT_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/feature_extraction", "features_speech_emobase")
os.makedirs(OUTPUT_DIR, exist_ok=True)

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# all_features = []

for company in os.listdir(RECO_DIR):
    speech_dir = os.path.join(RECO_DIR, company, "audio_wav")
    company_out_dir = os.path.join(OUTPUT_DIR, company)
    os.makedirs(company_out_dir, exist_ok=True)

    for wav_recording in os.listdir(speech_dir):
        wav_path = os.path.join(speech_dir, wav_recording)
        try:
            logging.info(f"Extracting features from: {wav_recording}")
            features = smile.process_file(wav_path)

            features["recording"] = wav_recording
            features["company"] = company

            # all_features.append(features)
            out_csv_path = os.path.join(company_out_dir, wav_recording.replace(".wav", ".csv"))
            features.to_csv(out_csv_path, index=False)

        except Exception as e:
            logging.info(f"Failed to process {wav_recording}: {e}")

        # if all_features:
        #     df = pd.concat(all_features)
        #     output_csv_path = os.path.join(company_out_dir, "speech_features.csv")
        #     df.to_csv(output_csv_path, index=False)