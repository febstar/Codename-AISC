import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("final_shot_dataset.csv")

# Features and label
X = df[["Elbow_Angle", "Shoulder_Angle", "Wrist_Angle", "Hip_Angle", "Knee_Angle"]]
y = df["Label"]

# Encode label: 'Made' -> 1, 'Missed' -> 0
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("✅ Model Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model and label encoder
joblib.dump(model, "shot_classifier.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("📦 Model saved as shot_classifier.pkl")
print("📦 Label encoder saved as label_encoder.pkl")

# Visualize feature importance
plt.figure(figsize=(8, 5))
importances = model.feature_importances_
features = X.columns
plt.barh(features, importances, color='teal')
plt.xlabel("Feature Importance")
plt.title("Which Joint Angles Influence the Shot Result?")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
