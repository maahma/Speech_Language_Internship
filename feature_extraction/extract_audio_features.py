import os
import opensmile
import pandas as pd

reco_dir = os.path.join("..", "dummy_recording_delete")

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,
    feature_level=opensmile.FeatureLevel.Functionals,
)

all_features = []

for company in os.listdir(reco_dir):
    speech_dir = os.path.join(reco_dir, company, "audio_wav")
    output_dir = os.path.join(reco_dir, company, "features_speech_emobase")
    os.makedirs(output_dir, exist_ok=True)

    for wav_recording in os.listdir(speech_dir):
        wav_path = os.path.join(speech_dir, wav_recording)
        try:
            print(f"Extracting features from: {wav_recording}")
            features = smile.process_file(wav_path)

            features["recording"] = wav_recording
            features["company"] = company

            all_features.append(features)

        except Exception as e:
            print(f"Failed to process {wav_recording}: {e}")

    if all_features:
        df = pd.concat(all_features)
        output_csv_path = os.path.join(output_dir, "speech_features.csv")
        df.to_csv(output_csv_path, index=False)
