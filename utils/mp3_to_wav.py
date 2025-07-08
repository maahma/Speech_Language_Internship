import ffmpeg
import os

def convert_mp3_to_wav(mp3_path, wav_path, ffmpeg_bin="/users/acu23ma/tools/ffmpeg-7.0.2-amd64-static/ffmpeg"):
    (
        ffmpeg
        .input(mp3_path)
        .output(wav_path)
        .run(cmd=ffmpeg_bin)
    )

    # return wav_path


reco_dir = os.path.join("..", "dummy_recording_delete")

for company in os.listdir(reco_dir):
    company_mp3_dir = os.path.join(reco_dir, company, "CEO")
    company_wav_dir = os.path.join(reco_dir, company, "audio_wav")
    os.makedirs(company_wav_dir, exist_ok=True)

    for audio_file in os.listdir(company_mp3_dir):
        if audio_file.endswith(".mp3"):
            print(f"audio_file: {audio_file}")
            mp3_path = os.path.join(company_mp3_dir, audio_file)
            wav_file_name = os.path.splitext(audio_file)[0] + ".wav"
            wav_path = os.path.join(company_wav_dir, wav_file_name)
            convert_mp3_to_wav(mp3_path, wav_path) 