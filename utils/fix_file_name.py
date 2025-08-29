import os

OUTPUT_DIR = os.path.join("/users/ac4ma/Speech_Language_Internship/model/emo_label")

for fname in os.listdir(OUTPUT_DIR):
    if fname.endswith("_emotion_results_normalized.csv_emotion_results_with_labels.csv"):
        old_path = os.path.join(OUTPUT_DIR, fname)
        new_name = fname.replace("_emotion_results_normalized.csv", "")
        new_path = os.path.join(OUTPUT_DIR, new_name)
        
        os.rename(old_path, new_path)
        print(f"Renamed: {fname} -> {new_name}")
