import os
import shutil


# PATH TO THE ALL_RECORDINGS DIRECTORY
reco_dir = os.path.join("..", "All_Recordings")

# missing_transcripts = []

for company_dir in os.listdir(reco_dir):
    company_path = os.path.join(reco_dir, company_dir)

    if os.path.isdir(company_path):

        text_path = os.path.join(company_path, "TextSequence.txt")

        if os.path.exists(text_path):
            with open(text_path, "r") as f:
                lines = f.read().splitlines()

            sentences = [line.strip() for line in lines if line.strip()]

            sent_token_dir = os.path.join(company_path, 'sent_token')
            os.makedirs(sent_token_dir, exist_ok=True)

            total = len(sentences)

            for i, sent in enumerate(sentences):
                padded_num = str(i + 1).zfill(len(str(total)))
                sentence_file = os.path.join(sent_token_dir, f"{padded_num}.txt")
                with open(sentence_file, "w") as out:
                    out.write(sent)

    
# print(missing_transcripts)
