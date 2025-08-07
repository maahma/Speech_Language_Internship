import ffmpeg
import os
import shutil

def convert_mp3_to_wav(mp3_path, wav_path, ffmpeg_bin="/users/ac4ma/tools/ffmpeg-7.0.2-amd64-static/ffmpeg"):
    try:
        (
            ffmpeg
            .input(mp3_path)
            .output(wav_path)
            .run(cmd=ffmpeg_bin)
        ) 
    except ffmpeg.Error as e:
        print(f"❌ Error converting {mp3_path}:\n{e.stderr.decode() if e.stderr else e}")

reco_dir = "/users/ac4ma/Speech_Language_Internship/All_Recordings"

for company in os.listdir(reco_dir):
    company_path = os.path.join(reco_dir, company)

    if os.path.isdir(company_path):
        company_mp3_dir = os.path.join(company_path, "CEO")
        company_wav_dir = os.path.join(company_path, "audio_wav")
        
        print(f"Currently processing : {company}")

        if os.path.exists(company_wav_dir):
            # shutil.rmtree(company_wav_dir)
            continue
        
        os.makedirs(company_wav_dir, exist_ok=True)

        for audio_file in os.listdir(company_mp3_dir):
            if audio_file.endswith(".mp3"):
                mp3_path = os.path.join(company_mp3_dir, audio_file)
                prefix = audio_file.split("-")[0]
                wav_file_name = f"{prefix}.wav"
                wav_path = os.path.join(company_wav_dir, wav_file_name)
                convert_mp3_to_wav(mp3_path, wav_path) 