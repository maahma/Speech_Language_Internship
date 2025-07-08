import os
import csv

reco_dir = os.path.join("..", "All_Recordings")


for company in os.listdir(reco_dir):
    if os.path.isdir(company):
        reco_dict = {}
        text_dict = {}
        company_path = os.path.join(reco_dir, company)
        speech_dir = os.path.join(reco_dir, company, "CEO")
        text_dir = os.path.join(reco_dir, company, "sent_token")

        if os.path.isdir(speech_dir) and os.path.isdir(text_dir):
            for recording in os.listdir(speech_dir):
                if recording.endswith(".mp3"):
                    reco_num = recording.split("-")[0]
                    relative_path = os.path.relpath(os.path.join(speech_dir, recording), start=reco_dir)
                    reco_dict[reco_num] = relative_path



            for transcript in os.listdir(text_dir):
                if transcript.endswith(".txt"):
                    text_num = transcript.split(".")[0]
                    relative_path = os.path.relpath(os.path.join(text_dir, transcript), start=reco_dir)
                    text_dict[text_num] = relative_path

        mapping_csv_path = os.path.join(company_path, "mapping.csv")
        with open(mapping_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Index", "Audio Path", "Transcript Path"])

            for key in sorted(reco_dict.keys(), key=lambda x: int(x)):
                if key in text_dict:
                    writer.writerow([key, reco_dict[key], text_dict[key]])



