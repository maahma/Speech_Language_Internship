import os
import shutil

# PATH TO THE ALL_RECORDINGS DIRECTORY
reco_dir = os.path.join("..", "All_Recordings")

for company_dir in os.listdir(reco_dir):
    company_path = os.path.join(reco_dir, company_dir)

    if os.path.isdir(company_path):
        sent_token_dir = os.path.join(company_path, "sent_token")

        if os.path.exists(sent_token_dir):
            shutil.rmtree(sent_token_dir)