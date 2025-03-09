import cv2
import pandas as pd

df = pd.read_csv("data_collection/labeled_shooting_data.csv")
df["Shot_Label"] = None  # Add label column

cap = cv2.VideoCapture("player_shooting.mp4")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Label Shots: Press 'M' for Make, 'X' for Miss", frame)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('m'):
        df.loc[frame_count, "Shot_Label"] = 1  # Made
    elif key == ord('x'):
        df.loc[frame_count, "Shot_Label"] = 0  # Missed
    else:
        continue  # Ignore other keys

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

df.to_csv("data_collection/labeled_shooting_data.csv", index=False)
print("Shots labeled and saved!")
