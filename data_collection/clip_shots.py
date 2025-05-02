import os
import cv2
import mediapipe as mp

VIDEO_FOLDER = "videos"  # Input folder with full session videos
CLIP_OUTPUT_FOLDER = "shot_clips"  # Where individual clips will be saved

if not os.path.exists(CLIP_OUTPUT_FOLDER):
    os.makedirs(CLIP_OUTPUT_FOLDER)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def clip_shots(video_path, buffer_frames=15):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    frames = []
    wrist_y_values = []
    shot_count = 0
    frame_index = 0

    prev_wrist_y = None
    peak_wrist_y = None
    jump_started = False
    start_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        frames.append(frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y

            if prev_wrist_y is None:
                prev_wrist_y = wrist_y
                peak_wrist_y = wrist_y
                continue

            # Detect rising wrist (jump up)
            if wrist_y < peak_wrist_y:
                peak_wrist_y = wrist_y
                if not jump_started:
                    jump_started = True
                    start_idx = max(0, frame_index - buffer_frames)

            # Detect falling wrist (ball released)
            elif jump_started and wrist_y > peak_wrist_y + 0.05:
                end_idx = min(len(frames), frame_index + buffer_frames)

                clip_frames = frames[start_idx:end_idx]
                clip_name = f"{video_name}_shot{shot_count+1}.mp4"
                clip_path = os.path.join(CLIP_OUTPUT_FOLDER, clip_name)

                out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
                for f in clip_frames:
                    out.write(f)
                out.release()

                print(f"🎥 Saved clip: {clip_path}")
                shot_count += 1

                # Reset detection
                jump_started = False
                peak_wrist_y = None
                prev_wrist_y = None
                frames = []  # Start buffer anew
                frame_index = 0  # Reset for frame tracking in clip

        prev_wrist_y = wrist_y

    cap.release()
    print(f"✅ Finished clipping {shot_count} shots from: {video_name}")

def process_all_videos():
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print("❌ No videos found in folder.")
        return

    for video in video_files:
        video_path = os.path.join(VIDEO_FOLDER, video)
        print(f"🔄 Clipping shots in: {video}")
        clip_shots(video_path)

if __name__ == "__main__":
    process_all_videos()
