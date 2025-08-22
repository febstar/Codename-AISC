import os

# Path to the main folder containing subfolders with videos
main_folder = r"C:\Users\User\Documents\AISC\data_collection\wrong_detections"

# Video file extensions to count
video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv"}

total_videos = 0

for root, dirs, files in os.walk(main_folder):
    for file in files:
        if os.path.splitext(file)[1].lower() in video_extensions:
            total_videos += 1

print(f"Total number of videos: {total_videos}")
