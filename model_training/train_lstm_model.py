import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib
import os

# File paths
DATASET_PATH = "final_shot_sequence_dataset.csv"
MODEL_PATH = "shot_lstm_model.h5"
ENCODER_PATH = "lstm_label_encoder.pkl"

# Load dataset
df = pd.read_csv(DATASET_PATH)
print(f"📊 Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")

# Prepare features and labels
X = df.drop("Label", axis=1).values
y = df["Label"].values

# Label encoding (e.g. "Made" -> 1, "Missed" -> 0)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Save encoder
joblib.dump(encoder, ENCODER_PATH)
print(f"🔖 Label encoder saved to: {ENCODER_PATH}")

# Reshape X for LSTM [samples, time steps, features]
n_steps = int(X.shape[1] / 5)  # Each frame has 5 angles
X = X.reshape((X.shape[0], n_steps, 5))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, activation="tanh", input_shape=(n_steps, 5), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(32, activation="relu"))
model.add(Dense(y_categorical.shape[1], activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train
print("🚀 Training model...")
model.fit(X_train, y_train, epochs=25, batch_size=16, validation_data=(X_test, y_test))

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Model Evaluation:")
print(f"Accuracy: {acc:.4f}")
print(f"Loss: {loss:.4f}")

# Save model
model.save(MODEL_PATH)
print(f"📦 Model saved as {MODEL_PATH}")
