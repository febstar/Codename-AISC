import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import sys

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    """Calculate the angle between three points in 2D space."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def process_video(video_path):
    """Extracts key shooting form angles from a video and saves them to a CSV file."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error: Unable to open video {video_path}")
        return
    
    data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
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

                # Calculate key angles
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                shoulder_angle = calculate_angle(elbow, shoulder, hip)
                wrist_angle = calculate_angle(elbow, wrist, (wrist[0], wrist[1] - 0.1))
                hip_angle = calculate_angle(shoulder, hip, knee)
                knee_angle = calculate_angle(hip, knee, ankle)

                data.append([frame_count, elbow_angle, shoulder_angle, wrist_angle, hip_angle, knee_angle])

                # Progress Update
                if frame_count % 50 == 0:
                    print(f"✅ Processed {frame_count} frames...")

            except IndexError:
                print(f"⚠️ Warning: Incomplete pose detected at frame {frame_count}. Skipping frame.")

    cap.release()

    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=["Frame", "Elbow_Angle", "Shoulder_Angle", "Wrist_Angle", "Hip_Angle", "Knee_Angle"])
    
    # Save CSV
    csv_name = os.path.splitext(os.path.basename(video_path))[0] + "_data.csv"
    output_path = os.path.join("processed_data", csv_name)

    if not os.path.exists("processed_data"):
        os.makedirs("processed_data")

    df.to_csv(output_path, index=False)
    print(f"📊 Data saved: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_video.py <video_file>")
    else:
        video_file = sys.argv[1]
        process_video(video_file)
