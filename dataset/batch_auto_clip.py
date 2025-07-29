import os
import cv2
import numpy as np
import argparse
import mediapipe as mp
import subprocess
import json
from collections import deque
from datetime import datetime

# === CONFIG ===
RAW_FOLDER = "raw_videos"
OUTPUT_FOLDER = "shot_clips"
LINKS_FILE = "youtube_links.txt"
LOG_FILE = "shot_learning_log.json"

BASE_PRE_JUMP_BUFFER = 20
BASE_POST_RELEASE_SECONDS = 2.0
MIN_POST_RELEASE_SECONDS = 2.56
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
HISTORY_SIZE = 20

# Wrong detection heuristic thresholds
MIN_CLIP_LENGTH_SEC = 1.2
MAX_CLIP_LENGTH_SEC = 5.0
LOW_CONFIDENCE_REJECT = 0.45

# --- Inserted into the CONFIG section ---
MOTION_THRESHOLD = 0.2  # Initial motion threshold (auto-tuned per video)
FLOW_SAMPLING_AREA = (0.3, 0.7)  # Central vertical band
MOTION_DYNAMIC_RANGE = (0.15, 0.5)  # Min/Max allowed threshold
BASELINE_SAMPLE_FRAMES = 50  # Frames used to determine baseline motion

# --- New global variable per video ---
motion_baseline = None

# Learning parameters
learning_params = {
    "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
    "knee_factor": 1.0,
    "wrist_factor": 1.0,
    "confidence_history": []
}

DEBUG = False
SHOW_SKELETON = False
SHOW_FLOW = False
os.makedirs(RAW_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# === LOAD/UPDATE LEARNING LOG ===
def load_learning_log():
    global learning_params
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            try:
                data = json.load(f)
                learning_params.update(data.get("params", {}))
            except json.JSONDecodeError:
                pass
        print(f"📚 Loaded learning parameters: {learning_params}")
    else:
        print("⚠️ No learning log found. Starting fresh.")

def update_learning_log(video_name, shots_data):
    data = {"params": learning_params, "runs": []}
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pass
    data["runs"].append({
        "video": video_name,
        "timestamp": datetime.now().isoformat(),
        "shots": shots_data
    })
    data["params"] = learning_params
    with open(LOG_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print("📝 Learning log updated.")
    auto_tune()

# === AUTO-TUNING LOGIC ===
def auto_tune():
    confs = learning_params.get("confidence_history", [])
    if len(confs) < 5:
        return

    avg_conf = np.mean(confs[-10:])
    print(f"🤖 Auto-tuning... Last 10 avg confidence = {avg_conf:.2f}")

    if avg_conf < 0.5:
        learning_params["confidence_threshold"] = min(0.9, learning_params["confidence_threshold"] + 0.05)
        learning_params["knee_factor"] *= 1.05
        learning_params["wrist_factor"] *= 1.05
        print(f"⚡ Shots are weak. Increased confidence threshold to {learning_params['confidence_threshold']:.2f}")
    elif avg_conf > 0.75 and len(confs) > 15:
        learning_params["confidence_threshold"] = max(0.4, learning_params["confidence_threshold"] - 0.03)
        learning_params["knee_factor"] *= 0.97
        learning_params["wrist_factor"] *= 0.97
        print(f"⚡ Shots are strong. Lowered confidence threshold to {learning_params['confidence_threshold']:.2f}")

    if len(confs) > 100:
        learning_params["confidence_history"] = confs[-100:]

# === STEP 1: DOWNLOAD VIDEO ===
def download_video(link):
    cmd = ["yt-dlp", "-f", "mp4", "-o", os.path.join(RAW_FOLDER, "%(title)s.%(ext)s"), link]
    subprocess.run(cmd, check=True)
    print(f"✅ Downloaded: {link}")

# === STEP 2: OPTICAL FLOW MOTION ANALYSIS ===
def compute_vertical_motion(frames):
    if len(frames) < 3:
        return 0.0

    total_motion = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        h, w = flow.shape[:2]
        y_start, y_end = int(h * FLOW_SAMPLING_AREA[0]), int(h * FLOW_SAMPLING_AREA[1])
        x_start, x_end = int(w * 0.3), int(w * 0.7)
        vertical_flow = flow[y_start:y_end, x_start:x_end, 1]
        total_motion.append(np.mean(vertical_flow))
        prev_gray = gray

        if SHOW_FLOW:
            hsv = np.zeros((h, w, 3), dtype=np.uint8)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = np.clip(mag * 15, 0, 255)
            flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow("Optical Flow", flow_vis)
            cv2.waitKey(1)

    return abs(np.mean(total_motion))

def tune_motion_threshold(video_motion_values):
    """Adjust MOTION_THRESHOLD based on average motion from the first frames."""
    global MOTION_THRESHOLD, motion_baseline
    if len(video_motion_values) < 5:
        return
    motion_baseline = np.mean(video_motion_values)
    MOTION_THRESHOLD = np.clip(motion_baseline * 0.3, *MOTION_DYNAMIC_RANGE)
    print(f"⚙️ Auto-tuned motion threshold: {MOTION_THRESHOLD:.3f} (baseline={motion_baseline:.3f})")


def motion_confidence(frames):
    vertical_movement = compute_vertical_motion(frames)
    conf = min(1.0, vertical_movement / MOTION_THRESHOLD)
    return conf, vertical_movement

# === STEP 3: WRONG DETECTION HEURISTIC ===
def is_wrong_detection(frames, fps, confidence):
    duration_sec = len(frames) / fps
    if confidence < LOW_CONFIDENCE_REJECT:
        return True
    if duration_sec < MIN_CLIP_LENGTH_SEC:
        return True
    if duration_sec > MAX_CLIP_LENGTH_SEC:
        return True
    return False

# === STEP 4: SAVE CLIP ===
def save_clip(frames, out_folder, shot_id, fps, confidence):
    motion_conf, motion_val = motion_confidence(frames)
    print(f"DEBUG Motion value: {motion_val:.4f}")
    final_conf = (confidence + motion_conf) / 2.0

    if is_wrong_detection(frames, fps, final_conf) or motion_conf < 0.2 * (MOTION_THRESHOLD / 0.2):
        print(f"❌ Discarded shot (pose_conf={confidence:.2f}, motion_conf={motion_conf:.2f}, motion={motion_val:.2f})")
        return False


    os.makedirs(out_folder, exist_ok=True)
    fname = os.path.join(out_folder, f"shot_{shot_id:02d}_conf{final_conf:.2f}.mp4")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"💾 Saved {fname} (pose_conf={confidence:.2f}, motion_conf={motion_conf:.2f})")
    return True

# === STEP 5: CALIBRATE ===
def calibrate_thresholds(cap):
    print("⚙️ Calibrating shot detection thresholds...")
    wrist_y_vals, knee_y_vals = [], []
    frame_idx = 0
    while frame_idx < 300:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            wrist_y_vals.append(lm[mp_pose.PoseLandmark.RIGHT_WRIST].y)
            knee_y_vals.append(lm[mp_pose.PoseLandmark.RIGHT_KNEE].y)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if len(wrist_y_vals) < 5 or len(knee_y_vals) < 5:
        print("⚠️ Not enough pose data for calibration, using defaults.")
        return 0.05, 0.04

    knee_range = np.percentile(knee_y_vals, 90) - np.percentile(knee_y_vals, 10)
    wrist_range = np.percentile(wrist_y_vals, 90) - np.percentile(wrist_y_vals, 10)
    knee_thresh = max(0.04, min(0.08, knee_range * 0.4)) * learning_params["knee_factor"]
    wrist_thresh = max(0.03, min(0.07, wrist_range * 0.35)) * learning_params["wrist_factor"]

    print(f"🎯 Calibrated thresholds: KNEE_DIP={knee_thresh:.3f}, WRIST_RISE={wrist_thresh:.3f}")
    return knee_thresh, wrist_thresh

# === STEP 6: ADAPTIVE BUFFERS ===
def adaptive_buffers(fps, knee_hist, wrist_hist):
    knee_vel = np.abs(np.diff(knee_hist)).mean() if len(knee_hist) > 3 else 0
    wrist_vel = np.abs(np.diff(wrist_hist)).mean() if len(wrist_hist) > 3 else 0

    pre_buffer = BASE_PRE_JUMP_BUFFER
    if knee_vel > 0.02:
        pre_buffer = max(10, int(BASE_PRE_JUMP_BUFFER * 0.6))
    elif knee_vel < 0.01:
        pre_buffer = min(40, int(BASE_PRE_JUMP_BUFFER * 1.5))

    post_sec = BASE_POST_RELEASE_SECONDS
    if wrist_vel > 0.03:
        post_sec *= 0.8
    elif wrist_vel < 0.015:
        post_sec *= 1.4

    post_sec = max(post_sec, MIN_POST_RELEASE_SECONDS)
    return pre_buffer, int(post_sec * fps)

# === STEP 7: CONFIDENCE SCORE ===
def compute_shot_confidence(knee_dip, wrist_rise, knee_thresh, wrist_thresh):
    knee_score = min(1.0, knee_dip / (knee_thresh * 1.2))
    wrist_score = min(1.0, wrist_rise / (wrist_thresh * 1.2))
    return (knee_score * 0.5 + wrist_score * 0.5)

# === STEP 8: PROCESS VIDEO ===
def process_video(path):
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    basename = os.path.splitext(os.path.basename(path))[0]
    out_folder = os.path.join(OUTPUT_FOLDER, basename)
    os.makedirs(out_folder, exist_ok=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n🎬 Processing: {basename} | FPS: {fps} | Total frames: {total_frames}")

    knee_thresh, wrist_thresh = calibrate_thresholds(cap)

    frame_buf, shot_buf = deque(maxlen=60), []
    in_shot = False
    release_detected = False
    release_countdown = 0
    shot_count = 0
    wrong_count = 0
    shot_confidence = 0
    wrist_y_hist, knee_y_hist = [], []

    shots_log = []
    
    video_motion_samples = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            wrist_y = lm[mp_pose.PoseLandmark.RIGHT_WRIST].y
            knee_y = lm[mp_pose.PoseLandmark.RIGHT_KNEE].y

            wrist_y_hist.append(wrist_y)
            knee_y_hist.append(knee_y)
            if len(wrist_y_hist) > HISTORY_SIZE:
                wrist_y_hist.pop(0)
                knee_y_hist.pop(0)

            if not in_shot and len(wrist_y_hist) >= 5:
                knee_dip = max(knee_y_hist) - knee_y_hist[-1]
                wrist_rise = max(wrist_y_hist) - wrist_y_hist[-1]
                confidence = compute_shot_confidence(knee_dip, wrist_rise, knee_thresh, wrist_thresh)

                if confidence >= learning_params["confidence_threshold"]:
                    pre_buffer, post_frames = adaptive_buffers(fps, knee_y_hist, wrist_y_hist)
                    in_shot = True
                    shot_confidence = confidence
                    shot_buf = list(frame_buf)[-pre_buffer:]
                    release_detected = False
                    release_countdown = post_frames
                    print(f"🏀 Shot started (conf={confidence:.2f}, pre={pre_buffer}f)")
                    learning_params["confidence_history"].append(confidence)

            if in_shot:
                shot_buf.append(frame)

                if not release_detected and len(wrist_y_hist) >= 3:
                    if wrist_y_hist[-2] < wrist_y_hist[-3] and wrist_y_hist[-2] < wrist_y_hist[-1]:
                        release_detected = True
                        print(f"🎯 Release detected (conf={shot_confidence:.2f})")

                if release_detected:
                    release_countdown -= 1
                    if release_countdown <= 0:
                        if save_clip(shot_buf, out_folder, shot_count + 1, fps, shot_confidence):
                            shot_count += 1
                            shots_log.append({"id": shot_count, "confidence": shot_confidence, "frames": len(shot_buf)})
                        else:
                            wrong_count += 1
                        in_shot = False
                        shot_buf = []

        frame_buf.append(frame)
        
        if len(video_motion_samples) < BASELINE_SAMPLE_FRAMES:
            if len(frame_buf) >= 2:
                # Compute quick motion sample
                motion_sample = compute_vertical_motion([frame_buf[-2], frame_buf[-1]])
                video_motion_samples.append(motion_sample)

            if len(video_motion_samples) == BASELINE_SAMPLE_FRAMES:
                tune_motion_threshold(video_motion_samples)


        if DEBUG:
            if SHOW_SKELETON and res.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Debug View", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()

    if wrong_count > 0 and wrong_count >= 0.3 * (shot_count + wrong_count):
        print(f"⚠️ {wrong_count} wrong detections found ({wrong_count/(shot_count+wrong_count):.0%}). Tightening thresholds.")
        learning_params["confidence_threshold"] = min(0.9, learning_params["confidence_threshold"] + 0.05)

    update_learning_log(basename, shots_log)
    print(f"✅ Done: {shot_count} shots saved for {basename} ({wrong_count} wrong detections discarded)")

# === STEP 9: BATCH PROCESS ===
def batch_process():
    if os.path.exists(LINKS_FILE):
        with open(LINKS_FILE) as f:
            for link in f:
                download_video(link.strip())

    videos = [f for f in os.listdir(RAW_FOLDER) if f.lower().endswith(('.mp4', '.mkv', '.avi'))]
    print(f"🎬 Found {len(videos)} videos. Extracting shots...")
    for vid in videos:
        process_video(os.path.join(RAW_FOLDER, vid))

# === ENTRY POINT ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug visualization")
    parser.add_argument("--train", action="store_true", help="Run self-tuning from past data only")
    parser.add_argument("--show-skeleton", action="store_true", help="Show MediaPipe skeleton in debug mode")
    parser.add_argument("--show-flow", action="store_true", help="Visualize optical flow")
    args = parser.parse_args()

    DEBUG = args.debug
    SHOW_SKELETON = args.show_skeleton
    SHOW_FLOW = args.show_flow
    load_learning_log()

    if args.train:
        auto_tune()
        print("🔧 Self-tuning completed from logs.")
    else:
        batch_process()
