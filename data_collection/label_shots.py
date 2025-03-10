import os
import cv2
import pandas as pd

PROCESSED_FOLDER = "processed_videos"
OUTPUT_CSV = "labeled_shooting_data.csv"

def label_shots():
    """Manually labels shots as Made (M) or Missed (X) while watching the full video."""
    video_files = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]

    if not video_files:
        print("❌ No processed videos found!")
        return

    labels = []

    for video in video_files:
        video_path = os.path.join(PROCESSED_FOLDER, video)
        cap = cv2.VideoCapture(video_path)

        print(f"🎥 Now labeling: {video}")
        shot_count = 1
        paused = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Video has ended

            frame_resized = cv2.resize(frame, (800, 600))
            cv2.putText(frame_resized, "Press SPACE to pause, 'M' for Made, 'X' for Missed, 'Q' to Quit", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Shot Labeling", frame_resized)
            key = cv2.waitKey(30) & 0xFF  # 30ms delay for smooth playback

            if key == ord(' '):  # Pause/Resume with SPACE
                paused = not paused
                while paused:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord(' '):  # Resume playback
                        paused = False
                    elif key == ord('m'):  # Label as Made
                        labels.append([video, shot_count, "Made"])
                        print(f"✅ Shot {shot_count} labeled as Made")
                        shot_count += 1
                    elif key == ord('x'):  # Label as Missed
                        labels.append([video, shot_count, "Missed"])
                        print(f"❌ Shot {shot_count} labeled as Missed")
                        shot_count += 1
                    elif key == ord('q'):  # Quit labeling
                        paused = False
                        cap.release()
                        cv2.destroyAllWindows()
                        save_labels(labels)
                        return

            elif key == ord('q'):  # Quit immediately
                cap.release()
                cv2.destroyAllWindows()
                save_labels(labels)
                return

        cap.release()
        cv2.destroyAllWindows()

    save_labels(labels)

def save_labels(labels):
    """Save labeled data to CSV"""
    df = pd.DataFrame(labels, columns=["Video", "Shot_Number", "Label"])
    os.makedirs(os.path.dirname(f"data_collection/{OUTPUT_CSV}"), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"📊 Shot labels saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    label_shots()
