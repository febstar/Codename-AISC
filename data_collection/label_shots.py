import os
import cv2
import pandas as pd

PROCESSED_FOLDER = "processed_videos"
OUTPUT_CSV = "labeled_shooting_data.csv"

def label_shots():
    """Plays the video until the end, then pauses for manual labeling of shots."""
    video_files = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]

    if not video_files:
        print("❌ No processed videos found!")
        return

    # Load existing labels to prevent relabeling
    if os.path.exists(OUTPUT_CSV):
        existing_labels = pd.read_csv(OUTPUT_CSV)["Video"].unique()
    else:
        existing_labels = []

    for video in video_files:
        video_name = os.path.splitext(video)[0]  # Remove extension

        if video_name in existing_labels:
            print(f"⚠️ Skipping {video} (Already labeled)")
            continue  # Skip already labeled videos

        video_path = os.path.join(PROCESSED_FOLDER, video)
        cap = cv2.VideoCapture(video_path)

        print(f"🎥 Now playing: {video}")
        last_frame = None  # Store last valid frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Video has ended
            
            last_frame = frame.copy()  # Save last valid frame
            frame_resized = cv2.resize(frame, (800, 600))
            cv2.imshow("Shot Labeling", frame_resized)
            
            key = cv2.waitKey(30) & 0xFF  # Play video smoothly at ~30fps
            if key == ord('q'):  # Quit immediately
                cap.release()
                cv2.destroyAllWindows()
                return  

        # Video finished, pause on the last frame for labeling
        if last_frame is not None:
            print("⏸ Video finished. Please label the shot.")
            last_frame_resized = cv2.resize(last_frame, (800, 600))
            cv2.putText(last_frame_resized, "Press 'M' for Made, 'X' for Missed, 'Q' to Quit",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            while True:  # Ensure labeling before exiting
                cv2.imshow("Shot Labeling", last_frame_resized)
                key = cv2.waitKey(0) & 0xFF  # Wait indefinitely

                if key == ord('m'):  # Label as Made
                    save_label(video_name, "Made")
                    print(f"✅ {video_name} labeled as Made")
                    break  
                elif key == ord('x'):  # Label as Missed
                    save_label(video_name, "Missed")
                    print(f"❌ {video_name} labeled as Missed")
                    break  
                elif key == ord('q'):  # Quit labeling
                    cap.release()
                    cv2.destroyAllWindows()
                    return  

        cap.release()
        cv2.destroyAllWindows()
        print(f"🎬 Finished labeling {video}")

def save_label(video_name, label):
    """Save labeled data to CSV"""
    df = pd.DataFrame([[video_name, label]], columns=["Video", "Label"])

    if os.path.exists(OUTPUT_CSV):  # Append if file exists
        df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(OUTPUT_CSV, index=False)

    print(f"📊 Label saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    label_shots()
