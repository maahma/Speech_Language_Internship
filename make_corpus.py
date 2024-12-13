import os

recordings_folder = "./Recordings/ACL19_Release"
transcript_list = []

for root, _, folders in os.walk(recordings_folder):
    for file in folders:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                transcript_list.append(f.read())

print(f"Number of transcripts loaded: {len(transcript_list)}")
# print(transcript_list[0])
