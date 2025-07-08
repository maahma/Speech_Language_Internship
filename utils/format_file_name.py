import os

reco_dir = os.path.join("..", "All_Recordings")

last_recording_number = 0

new_filenames = []

for company_dir in os.listdir(reco_dir):
    company_path = os.path.join(reco_dir, company_dir, "CEO")
    last_recording_number = 0

    if os.path.isdir(company_path):
        mp3_files = []

        for recording_file in os.listdir(company_path):
            if recording_file.endswith("mp3"):
                mp3_files.append(recording_file)

        mp3_files.sort()

        for f in mp3_files:
            file_parts = f.split("-")
            reco_num_part = file_parts[0]
            speaker_name_part = file_parts[1].replace(".mp3", "")

            last_recording_number += 1

            new_filename = f"{str(last_recording_number).zfill(3)}-{speaker_name_part}.mp3"

            old_path = os.path.join(company_path, f)
            new_path = os.path.join(company_path, new_filename)

            os.rename(old_path, new_path)

