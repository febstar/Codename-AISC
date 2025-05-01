import os
import pandas as pd

# File Paths
PROCESSED_DATA_FOLDER = "data_collection/processed_data"
RELEASE_CSV = "data_collection/release_data/shot_release.csv"
LABELS_CSV = "data_collection/labeled_shooting_data.csv"
OUTPUT_CSV = "final_shot_dataset.csv"

def load_data():
    """Loads all processed video CSVs, the shot release CSV, and shot labels CSV."""
    
    # Load processed shooting data
    processed_files = [f for f in os.listdir(PROCESSED_DATA_FOLDER) if f.endswith("_data.csv")]
    processed_data = {}

    if not processed_files:
        print("❌ No processed data files found!")
        return {}, None, None

    for file in processed_files:
        video_name = file.replace("_data.csv", "")
        df = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, file))
        processed_data[video_name] = df

    # Load shot release data
    if os.path.exists(RELEASE_CSV):
        release_df = pd.read_csv(RELEASE_CSV)
    else:
        print(f"⚠️ Missing {RELEASE_CSV}")
        return processed_data, None, None

    # Load labeled shots data
    if os.path.exists(LABELS_CSV):
        labels_df = pd.read_csv(LABELS_CSV)
    else:
        print(f"⚠️ Missing {LABELS_CSV}")
        return processed_data, release_df, None

    return processed_data, release_df, labels_df

def match_release_frames(processed_data, release_df):
    """Extracts key angles at the release frame for each shot."""
    release_features = []

    if release_df is None:
        print("❌ No release data found, skipping feature extraction!")
        return pd.DataFrame()

    for _, row in release_df.iterrows():
        video_name = row["Video"]
        frame_number = int(row["Frame Number"])  # Ensure it's an integer

        if video_name in processed_data:
            video_data = processed_data[video_name]

            # Find the frame closest to the release
            shot_data = video_data[video_data["Frame"] == frame_number]

            if not shot_data.empty:
                angles = shot_data.iloc[0][["Elbow_Angle", "Shoulder_Angle", "Wrist_Angle", "Hip_Angle", "Knee_Angle"]]
                release_features.append([video_name, frame_number] + list(angles))
            else:
                print(f"⚠️ No matching frame {frame_number} found in {video_name}")

    if not release_features:
        print("⚠️ No valid release features extracted!")
        return pd.DataFrame()

    return pd.DataFrame(release_features, columns=["Video", "Frame", "Elbow_Angle", "Shoulder_Angle", "Wrist_Angle", "Hip_Angle", "Knee_Angle"])

def merge_labels(release_features, labels_df):
    """Assigns 'Made' or 'Missed' labels to each shot."""
    if release_features.empty or labels_df is None:
        print("❌ Missing release features or labels, skipping merging!")
        return pd.DataFrame()

    final_data = release_features.merge(labels_df, on="Video", how="inner")

    if final_data.empty:
        print("⚠️ No matching labels found, check if videos are named correctly!")
    
    return final_data

def save_final_dataset(final_data):
    """Saves the cleaned dataset to a CSV file."""
    if final_data.empty:
        print("❌ No data to save!")
        return

    final_data.to_csv(OUTPUT_CSV, index=False)
    print(f"📊 Final dataset saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    print("🔄 Loading data...")
    processed_data, release_df, labels_df = load_data()

    if not processed_data:
        print("❌ No processed data available. Exiting.")
        exit()

    print("📌 Extracting release frame data...")
    release_features = match_release_frames(processed_data, release_df)

    print("🏀 Assigning shot labels...")
    final_data = merge_labels(release_features, labels_df)

    print("💾 Saving final dataset...")
    save_final_dataset(final_data)
    print("✅ Data preprocessing complete!")
