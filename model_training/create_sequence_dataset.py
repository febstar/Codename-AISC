# sequence_modeling/create_sequence_dataset.py

import os
import pandas as pd

SEQUENCE_FOLDER = "data_collection/processed_sequences"  # updated path
LABELS_CSV = "data_collection/labeled_shooting_data.csv"
OUTPUT_CSV = "final_shot_sequence_dataset.csv"
SEQUENCE_LENGTH = 10  # number of frames to include per sequence

def flatten_sequence(df, video_name):
    """
    Flattens the last SEQUENCE_LENGTH frames into one row of features.
    """
    if len(df) < SEQUENCE_LENGTH:
        print(f"⚠️ Skipping {video_name}: only {len(df)} frames available (need {SEQUENCE_LENGTH})")
        return None

    df = df.tail(SEQUENCE_LENGTH)
    flat_features = []

    for _, row in df.iterrows():
        flat_features.extend([
            row["Elbow_Angle"],
            row["Shoulder_Angle"],
            row["Wrist_Angle"],
            row["Hip_Angle"],
            row["Knee_Angle"]
        ])

    return flat_features

def main():
    if not os.path.exists(SEQUENCE_FOLDER):
        print("❌ Missing folder:", SEQUENCE_FOLDER)
        return

    if not os.path.exists(LABELS_CSV):
        print("❌ Missing label file:", LABELS_CSV)
        return

    labels_df = pd.read_csv(LABELS_CSV)
    all_features = []
    all_labels = []

    for file in os.listdir(SEQUENCE_FOLDER):
        if file.endswith(".csv"):
            video_name = os.path.splitext(file)[0].replace("_sequence", "")
            file_path = os.path.join(SEQUENCE_FOLDER, file)

            try:
                df = pd.read_csv(file_path)
                flat_sequence = flatten_sequence(df, video_name)

                if flat_sequence:
                    label_row = labels_df[labels_df["Video"] == video_name]
                    if not label_row.empty:
                        label = label_row.iloc[0]["Label"]
                        all_features.append(flat_sequence)
                        all_labels.append(label)
                    else:
                        print(f"⚠️ No label found for: {video_name}")
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")

    if not all_features:
        print("❌ No valid sequences found.")
        return

    # Build column headers
    columns = []
    for i in range(SEQUENCE_LENGTH):
        columns.extend([
            f"Elbow_{i}", f"Shoulder_{i}", f"Wrist_{i}", f"Hip_{i}", f"Knee_{i}"
        ])

    df_out = pd.DataFrame(all_features, columns=columns)
    df_out["Label"] = all_labels

    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"📊 Sequence dataset saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
