import os
import cv2
import mediapipe as mp
import numpy as np

PROCESSED_FOLDER = "processed_videos"
OUTPUT_FOLDER = "data_collection/release_frames"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def detect_shot_release(video_path):
    """Detects the exact moment the ball leaves the player's hand."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    prev_wrist_y = None
    peak_wrist_y = None
    release_frame = None
    release_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

            wrist_y = wrist.y  # Y-coordinate (higher = smaller value)

            if prev_wrist_y is None:
                prev_wrist_y = wrist_y
                peak_wrist_y = wrist_y

            # Update peak wrist height if the wrist moves upward
            if wrist_y < peak_wrist_y:
                peak_wrist_y = wrist_y

            # Detect release: when wrist starts dropping significantly
            if wrist_y > peak_wrist_y + 0.05 and not release_detected:
                release_detected = True
                release_frame = frame

        if release_detected:
            release_path = os.path.join(OUTPUT_FOLDER, f"{video_name}_release.jpg")
            cv2.imwrite(release_path, release_frame)
            print(f"📸 Shot release frame saved: {release_path}")
            break  # Stop after detecting the first release

        prev_wrist_y = wrist_y  # Store previous wrist position for next frame

    cap.release()

def process_all_videos():
    """Runs shot release detection on all processed videos."""
    video_files = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]

    if not video_files:
        print("❌ No processed videos found!")
        return

    for video in video_files:
        video_path = os.path.join(PROCESSED_FOLDER, video)
        print(f"🔍 Detecting shot release in: {video}")
        detect_shot_release(video_path)

if __name__ == "__main__":
    process_all_videos()
