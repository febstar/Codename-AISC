import os
import subprocess

# Upload videos
# print("📂 Uploading videos to the 'videos' directory...")
# subprocess.run(["python", "upload_videos.py"])

# Step 1: Manually label shots
print("\n🏀 Step 1: Launching manual shot labeling...")
subprocess.run(["python", "label_shots.py"])

# Step 2: Process all videos (This runs collect_frames.py inside it)
print("🔄 Step 2: Processing all videos...")
subprocess.run(["python", "process_videos.py"])

# Step 3: Extract shot release frames
print("\n🔍 Step 3: Extracting shot release frames...")
subprocess.run(["python", "extract_shot_release.py"])

# Step 4: Extract shot sequence
print("\n🔍 Step 4: Extracting shot sequences...")
subprocess.run(["python", "extract_shot_sequence.py"])


print("\n✅ All steps completed! Data is ready for preprocessing.")
