import os
import shutil
import tkinter as tk
from tkinter import filedialog

UPLOAD_FOLDER = "videos"

def upload_videos():
    """Opens a file dialog to select multiple videos and upload them to the videos folder."""
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Open file dialog to select multiple video files
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    video_paths = filedialog.askopenfilenames(title="Select Videos to Upload", 
                                              filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])

    if not video_paths:
        print("❌ No videos selected!")
        return

    for video_path in video_paths:
        filename = os.path.basename(video_path)
        dest_path = os.path.join(UPLOAD_FOLDER, filename)

        shutil.copy(video_path, dest_path)
        print(f"✅ Video uploaded: {filename}")

# Example Usage:
if __name__ == "__main__":
    upload_videos()
