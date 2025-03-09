import pandas as pd
import numpy as np

df = pd.read_csv("data_collection/labeled_shooting_data.csv")
release_frames = []

for i in range(1, len(df)):
    wrist_speed = np.linalg.norm(df.iloc[i, 2:4] - df.iloc[i-1, 2:4])  # Speed of wrist movement

    if wrist_speed > 0.1:  # Threshold for shot release
        release_frames.append(i)

# Save shot release frames
release_df = df.iloc[release_frames]
release_df.to_csv("data_collection/release_frames.csv", index=False)
print("Shot release frames detected and saved!")
