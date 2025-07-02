# data_collection/extract_shot_sequence.py

import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

INPUT_FOLDER = "processed_videos"
OUTPUT_FOLDER = "processed_sequences"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    """Calculate angle between three 2D points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

def extract_sequence(video_path, margin_before=10, margin_after=5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    all_frames = []
    wrist_ys = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
                elbow = (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y)
                wrist = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y)
                hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)
                knee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y)
                ankle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y)

                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                shoulder_angle = calculate_angle(elbow, shoulder, hip)
                wrist_angle = calculate_angle(elbow, wrist, (wrist[0], wrist[1] - 0.1))
                hip_angle = calculate_angle(shoulder, hip, knee)
                knee_angle = calculate_angle(hip, knee, ankle)

                frame_data = {
                    "Elbow_Angle": elbow_angle,
                    "Shoulder_Angle": shoulder_angle,
                    "Wrist_Angle": wrist_angle,
                    "Hip_Angle": hip_angle,
                    "Knee_Angle": knee_angle,
                    "Wrist_Y": wrist[1]
                }
                all_frames.append(frame_data)
                wrist_ys.append(wrist[1])

            except IndexError:
                all_frames.append(None)
                wrist_ys.append(None)

    cap.release()

    # Release detection: Wrist Y hits min (top), then goes back down
    wrist_ys_np = np.array([y if y is not None else np.nan for y in wrist_ys])
    if np.all(np.isnan(wrist_ys_np)):
        print(f"⚠️ No valid wrist data in {video_name}")
        return

    peak_index = np.nanargmin(wrist_ys_np)

    # Detect jump start: first dip before rise
    jump_start = None
    for i in range(peak_index - 1, 0, -1):
        if wrist_ys_np[i] > wrist_ys_np[i - 1]:
            jump_start = i
            break
    if jump_start is None:
        jump_start = max(peak_index - margin_before, 0)

    start_idx = max(jump_start - margin_before, 0)
    end_idx = min(peak_index + margin_after, len(all_frames))

    sequence = []
    for idx in range(start_idx, end_idx):
        frame = all_frames[idx]
        if frame:
            sequence.append([
                idx,
                frame["Elbow_Angle"],
                frame["Shoulder_Angle"],
                frame["Wrist_Angle"],
                frame["Hip_Angle"],
                frame["Knee_Angle"]
            ])

    df = pd.DataFrame(sequence, columns=["Frame", "Elbow_Angle", "Shoulder_Angle", "Wrist_Angle", "Hip_Angle", "Knee_Angle"])
    csv_path = os.path.join(OUTPUT_FOLDER, f"{video_name}_sequence.csv")
    df.to_csv(csv_path, index=False)
    print(f"📊 Sequence saved: {csv_path}")

def process_all_videos():
    video_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
    for video in video_files:
        path = os.path.join(INPUT_FOLDER, video)
        print(f"🔄 Extracting sequence from: {video}")
        extract_sequence(path)

if __name__ == "__main__":
    process_all_videos()
