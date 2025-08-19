import os
import shutil

# Source folder that contains subfolders with videos
source_folder = r"C:\Users\User\Documents\AISC\dataset\shot_clips"

# Destination folder where all videos will be moved
destination_folder = r"C:\Users\User\Documents\AISC\data_collection\videos"

# Video file extensions to move
video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv"}

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

counter = 1  # For renaming duplicates if needed

for root, dirs, files in os.walk(source_folder):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in video_extensions:
            source_path = os.path.join(root, file)

            # Ensure unique filename in destination
            dest_path = os.path.join(destination_folder, file)
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(file)
                dest_path = os.path.join(destination_folder, f"{name}_{counter}{ext}")
                counter += 1

            # Move file
            shutil.move(source_path, dest_path)
            print(f"Moved: {source_path} → {dest_path}")

print("✅ All videos moved successfully!")
