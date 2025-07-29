# analysis/feature_importance_plot.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load the flat dataset (not sequence)
DATA_PATH = "final_shot_dataset.csv"  # make sure this is the one with single frame data

# Load dataset
df = pd.read_csv(DATA_PATH)

# Features and labels
X = df.drop(["Video", "Frame", "Label"], axis=1)
y = df["Label"]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Fit Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Compute permutation importance
result = permutation_importance(clf, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)

# Map importances
importances = result.importances_mean
feature_names = X.columns

# Create DataFrame for plotting
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=True)

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm")
plt.title("Which Joint Angles Influence the Shot Result?")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
