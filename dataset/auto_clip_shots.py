# dataset/auto_clip_shots.py (IMPROVED)

import os
import cv2
import mediapipe as mp
import numpy as np

# === PARAMETERS ===
INPUT_VIDEO = "raw_videos/example.mp4"   # Change to your video
OUTPUT_FOLDER = "shot_clips"
PRE_FRAMES = 12       # Frames before detected release
POST_FRAMES = 35      # Frames after release
MIN_FRAME_GAP = 15    # Minimum gap between shots (to avoid duplicates)
DISPLAY = False       # Set True for debugging

# === INITIAL SETUP ===
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def detect_shots_from_wrist(wrist_ys):
    """
    Detect shot release frames by finding wrist Y minima (highest point).
    Returns a list of release frame indices.
    """
    release_frames = []
    wrist_ys = np.array(wrist_ys)
    
    for i in range(1, len(wrist_ys)-1):
        if wrist_ys[i] < wrist_ys[i-1] and wrist_ys[i] < wrist_ys[i+1]:  # local min
            if np.isnan(wrist_ys[i]):
                continue
            if not release_frames or i - release_frames[-1] > MIN_FRAME_GAP:
                release_frames.append(i)
    return release_frames

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🔄 Processing: {video_path}")
    print(f"FPS: {fps}, Total frames: {total_frames}")

    frames = []
    wrist_ys = []

    # === STEP 1: Collect all frames and wrist Y positions ===
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        wrist_y = np.nan
        if results.pose_landmarks:
            wrist_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y

        wrist_ys.append(wrist_y)
        frames.append(frame)

    cap.release()

    # === STEP 2: Detect shot releases ===
    release_frames = detect_shots_from_wrist(wrist_ys)
    print(f"🏀 Detected {len(release_frames)} possible shots.")

    # === STEP 3: Clip videos ===
    shot_count = 0
    for release_idx in release_frames:
        start_idx = max(0, release_idx - PRE_FRAMES)
        end_idx = min(len(frames), release_idx + POST_FRAMES)

        clip_frames = frames[start_idx:end_idx]
        if clip_frames:
            shot_count += 1
            save_clip(clip_frames, shot_count, fps)

    print(f"✅ Finished: {shot_count} shots saved.")

def save_clip(frames, shot_num, fps):
    filename = os.path.join(OUTPUT_FOLDER, f"shot_{shot_num}.mp4")
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    print(f"💾 Saved: {filename}")

if __name__ == "__main__":
    process_video(INPUT_VIDEO)
