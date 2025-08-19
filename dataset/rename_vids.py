import os

# Path to the main folder containing subfolders with videos
main_folder = r"C:\Users\User\Documents\AISC\dataset\shot_clips"

# Video file extensions to rename
video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv"}

for root, dirs, files in os.walk(main_folder):
    # Get the subfolder name
    folder_name = os.path.basename(root)
    if folder_name.strip() == "":
        continue  # Skip if it's the main folder without a name

    # Extract first and last word from the subfolder name
    words = folder_name.split()
    if len(words) >= 2:
        prefix = f"{words[0]}_{words[-1]}"
    else:
        prefix = words[0]  # Only one word in folder name

    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in video_extensions:
            old_path = os.path.join(root, file)
            new_filename = f"{prefix}_{file}"
            new_path = os.path.join(root, new_filename)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} → {new_path}")

print("✅ Renaming completed!")
