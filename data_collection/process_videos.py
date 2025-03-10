import os
import subprocess

VIDEO_FOLDER = "videos"
PROCESSED_FOLDER = "processed_videos"

def process_all_videos():
    """Loops through all videos and runs data collection scripts."""
    if not os.path.exists(PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER)

    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]

    for video in video_files:
        video_path = os.path.join(VIDEO_FOLDER, video)
        print(f"🔄 Processing: {video}")

        # Run keypoint extraction
        subprocess.run(["python", "collect_frames.py", video_path])

        # Move processed videos
        new_path = os.path.join(PROCESSED_FOLDER, video)
        os.rename(video_path, new_path)
        print(f"✅ Moved {video} to processed_videos folder")

if __name__ == "__main__":
    process_all_videos()
