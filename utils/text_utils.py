import os
import shutil


# PATH TO THE ALL_RECORDINGS DIRECTORY
reco_dir = os.path.join("..", "All_Recordings")

for company_dir in os.listdir(reco_dir):
    print(f"Processing {company_dir}")
    company_path = os.path.join(reco_dir, company_dir)
    
    if os.path.isdir(company_path):
        text_path = os.path.join(company_path, "TextSequence.txt")

        if os.path.exists(text_path):
            with open(text_path, "r") as f:
                lines = f.read().splitlines()

            sentences = [line.strip() for line in lines if line.strip()]

            sent_token_dir = os.path.join(company_path, 'sent_token')

            if os.path.exists(sent_token_dir):
                shutil.rmtree(sent_token_dir)

            os.makedirs(sent_token_dir, exist_ok=True)

            for i, sent in enumerate(sentences):
                padded_num = str(i + 1).zfill(3)
                sentence_file = os.path.join(sent_token_dir, f"{padded_num}.txt")
                with open(sentence_file, "w") as out:
                    out.write(sent)
        else:
            print(f"Skipping {company_path} - no TextSequence.txt found")