import os
import subprocess

# Step 1: Process all videos (This runs collect_frames.py inside it)
print("🔄 Step 1: Processing all videos...")
subprocess.run(["python", "process_videos.py"])

# Step 2: Extract shot release frames
print("\n🔍 Step 2: Extracting shot release frames...")
subprocess.run(["python", "extract_shot_release.py"])

# Step 2: Extract shot sequence
print("\n🔍 Step 2: Extracting shot sequences...")
subprocess.run(["python", "extract_shot_sequence.py"])

# Step 3: Manually label shots
print("\n🏀 Step 3: Launching manual shot labeling...")
subprocess.run(["python", "label_shots.py"])

print("\n✅ All steps completed! Data is ready for preprocessing.")
