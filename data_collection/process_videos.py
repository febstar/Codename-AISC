import os
import subprocess

VIDEO_FOLDER = "videos"
PROCESSED_FOLDER = "processed_videos"

def safe_move(video_path, new_folder):
    """Moves video to the processed folder, ensuring no overwrite."""
    filename = os.path.basename(video_path)
    new_path = os.path.join(new_folder, filename)

    # If file already exists, add _processed to filename
    if os.path.exists(new_path):
        name, ext = os.path.splitext(filename)
        new_path = os.path.join(new_folder, f"{name}_processed{ext}")

    os.rename(video_path, new_path)
    print(f"✅ Moved {filename} to {new_folder}")

def process_all_videos():
    """Loops through all videos and runs data collection scripts."""
    if not os.path.exists(PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER)

    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]

    if not video_files:
        print("❌ No videos found in the videos folder!")
        return

    for video in video_files:
        video_path = os.path.join(VIDEO_FOLDER, video)
        print(f"🔄 Processing: {video}")

        try:
            result = subprocess.run(["python", "collect_frames.py", video_path], check=True)
            
            if result.returncode == 0:  # Only move video if script runs successfully
                safe_move(video_path, PROCESSED_FOLDER)
            else:
                print(f"⚠️ Processing failed for {video}. Skipping move.")
        
        except subprocess.CalledProcessError as e:
            print(f"❌ Error processing {video}: {e}. Skipping this file.")

if __name__ == "__main__":
    process_all_videos()
