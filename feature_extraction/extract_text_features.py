import os
 

RECO_DIR = os.path.join("..", "dummy_recording_delete")
OUTPUT_DIR = "./features_text_glove"
os.makedirs(OUTPUT_DIR, exist_ok=True)


for company in os.listdir(RECO_DIR):
    text_dir = os.path.join(RECO_DIR, company, "sent_token")
    company_out_dir = os.path.join(OUTPUT_DIR, company)   
    os.makedirs(company_out_dir, exist_ok=True)

    for text_file in os.listdir(text_dir):
        
        output_csv = os.path.join(company_out_dir, text_file.replace(".txt", ".csv"))
        try:
            
        except Exception as e:
            print(f"Failed to process {text_file}: {e}")