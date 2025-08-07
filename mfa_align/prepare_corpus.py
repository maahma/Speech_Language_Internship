import os
import shutil

RECO_DIR = "/users/ac4ma/Speech_Language_Internship/All_Recordings"
print("Script started")


for company in os.listdir(RECO_DIR):
    company_dir = os.path.join(RECO_DIR, company)

    print(f"Processing {company}")

    if os.path.isdir(company_dir):
        wav_dir = os.path.join(company_dir, "audio_wav")       
        text_dir = os.path.join(company_dir, "sent_token") 
        corpus_dir = os.path.join(company_dir, "corpus_dir") 


        if os.path.isdir(wav_dir) and os.path.isdir(text_dir):

            # if os.path.exists(corpus_dir):
            #     shutil.rmtree(corpus_dir)
            
            os.makedirs(corpus_dir, exist_ok=True)

            for f_name in os.listdir(wav_dir):
                if f_name.endswith(".wav"):
                    base_name = os.path.splitext(f_name)[0]
                    wav_path = os.path.join(wav_dir, f_name)
                    txt_path = os.path.join(text_dir, base_name + ".txt") 

                    if os.path.exists(txt_path):
                        shutil.copy(wav_path, os.path.join(corpus_dir, f_name))
                        shutil.copy(txt_path, os.path.join(corpus_dir, base_name + ".txt"))
                    else:
                        print(f"Missing .txt for {f_name}")
