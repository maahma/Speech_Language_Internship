import os

reco_dir = os.path.join(os.getcwd(), "All_Recordings")

for comp_dir in os.listdir(reco_dir):
    comp_path = os.path.join(reco_dir, comp_dir)

    if os.path.isdir(comp_path):
        ceo_path = os.path.join(comp_path, "CEO")
        if os.path.exists(ceo_path) and os.path.isdir(ceo_path):
            for reco_file in os.listdir(ceo_path):
                old_path = os.path.join(ceo_path, reco_file)
                f_name, f_ext = os.path.splitext(reco_file)
                f_speaker, f_section, f_recording = f_name.split('_')
                f_speaker = f_speaker.replace(" ", "")
                f_section = f_section.zfill(2)
                f_recording = f_recording.zfill(3)
                # print('{}-{}-{}{}'.format(f_section, f_recording, f_speaker, f_ext))
                new_f_name = '{}-{}-{}{}'.format(f_section, f_recording, f_speaker, f_ext)
                new_path = os.path.join(ceo_path, new_f_name)
                os.rename(old_path, new_path)
