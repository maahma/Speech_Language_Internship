import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

RECO_DIR = os.path.join("/mnt/parscratch/users/ac4ma/All_Recordings")

count = 0
for company in sorted(os.listdir(RECO_DIR)):
    audio_dir = os.path.join(RECO_DIR, company, "audio_wav")

    if not os.path.isdir(audio_dir):
        logging.warning(f"NOT A FOLDER: {company}, skipping...")
        continue

    for filename in sorted(os.listdir(audio_dir)):

        count+=1


logging.info(f"Total files: {count}")
