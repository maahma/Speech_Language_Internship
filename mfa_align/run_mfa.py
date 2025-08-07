import os
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

reco_dir = os.path.join("/mnt/parscratch/users/ac4ma/All_Recordings")
dictionary_path = "/users/ac4ma/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict"
acoustic_model_path = "/users/ac4ma/Documents/MFA/pretrained_models/acoustic/english_us_arpa.zip"
output_root = os.path.join(".", "aligned_audio_txt")
os.makedirs(output_root, exist_ok=True)

skip_company = "Amazon.com Inc._20170202"

for company_dir in os.listdir(reco_dir):
    if company_dir == skip_company:
        logging.info(f"Skipping {company_dir}")
        continue

    corpus_path = os.path.join(reco_dir, company_dir, "corpus_dir")
    output_path = os.path.join(output_root, company_dir)

    os.makedirs(output_path, exist_ok=True)


    if os.path.isdir(corpus_path):
        logging.info(f"Running MFA for {company_dir}")

        command = [
            "/users/ac4ma/.conda/envs/aligner/bin/mfa", "align",
            corpus_path,
            dictionary_path,
            acoustic_model_path,
            output_path,
            "--clean", "--overwrite"
        ]

        try:
            subprocess.run(command, check=True)
            logging.info(f"Alignment completed for {company_dir}")
        except subprocess.CalledProcessError as e:
            logging.info(f"MFA failed for {company_dir}: {e}")
    else:
        logging.info(f"No corpus_dir found for {company_dir}, skipping.")
