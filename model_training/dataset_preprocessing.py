import os
import pandas as pd
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import StandardScaler

# Paths to datasets
DATA_FOLDER = "data_collection/processed_data"
LABELS_FILE = "data_collection/labeled_shooting_data.csv"
RELEASE_FILE = "data_collection/shot_release_data.csv"
IMAGE_FOLDER = "data_collection/release_frames"  # Folder containing release frame images
OUTPUT_FILE = "final_shooting_dataset.csv"

# Load MobileNetV2 model (without top layers)
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_image_features(img_path):
    if not os.path.exists(img_path):
        return np.zeros((1280,))  # Return zero vector if image is missing
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Load processed CSV files
def load_processed_data():
    all_data = []
    for file in os.listdir(DATA_FOLDER):
        if file.endswith("_data.csv"):
            df = pd.read_csv(os.path.join(DATA_FOLDER, file))
            df["Video"] = file.replace("_data.csv", "")
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else None

# Load shot labels
def load_shot_labels():
    return pd.read_csv(LABELS_FILE) if os.path.exists(LABELS_FILE) else None

# Load shot release data
def load_shot_release():
    return pd.read_csv(RELEASE_FILE) if os.path.exists(RELEASE_FILE) else None

# Merge datasets
def merge_datasets(processed_df, labels_df, release_df):
    # Merge only if 'Shot_Number' exists
    if labels_df is not None and "Shot_Number" in labels_df.columns and "Shot_Number" in processed_df.columns:
        processed_df = processed_df.merge(labels_df, on=["Video", "Shot_Number"], how="left")
    else:
        print("⚠️ Warning: 'Shot_Number' column missing, skipping labels merge!")

    if release_df is not None and "Frame" in release_df.columns and "Frame" in processed_df.columns:
        processed_df = processed_df.merge(release_df, on=["Video", "Frame"], how="left")
    else:
        print("⚠️ Warning: 'Frame' column missing, skipping release data merge!")

    return processed_df


# Extract image features for each shot release frame
def add_image_features(df):
    print("🖼️ Extracting image features...")
    image_features = []
    for _, row in df.iterrows():
        img_filename = f"{row['Video']}_frame_{int(row['Frame'])}.jpg"
        img_path = os.path.join(IMAGE_FOLDER, img_filename)
        image_features.append(extract_image_features(img_path))
    
    image_features = np.array(image_features)
    feature_columns = [f"img_feature_{i}" for i in range(image_features.shape[1])]
    image_df = pd.DataFrame(image_features, columns=feature_columns)
    df = df.reset_index(drop=True)
    df = pd.concat([df, image_df], axis=1)
    return df

# Normalize features
def normalize_features(df):
    scaler = StandardScaler()
    angle_cols = ["Elbow_Angle", "Shoulder_Angle", "Wrist_Angle", "Hip_Angle", "Knee_Angle"]
    df[angle_cols] = scaler.fit_transform(df[angle_cols])
    return df

# Handle missing values
def clean_data(df):
    df.dropna(inplace=True)
    return df

# Main preprocessing function
def preprocess_data():
    print("🔍 Loading data...")
    processed_df = load_processed_data()
    labels_df = load_shot_labels()
    release_df = load_shot_release()
    
    if processed_df is None:
        print("❌ No processed data found!")
        return
    
    print("🔄 Merging datasets...")
    merged_df = merge_datasets(processed_df, labels_df, release_df)
    print("📸 Adding image features...")
    merged_df = add_image_features(merged_df)
    print("📏 Normalizing angles...")
    normalized_df = normalize_features(merged_df)
    print("🧹 Cleaning data...")
    final_df = clean_data(normalized_df)
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Preprocessing complete! Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess_data()
