# Codename‑AISC

> **A**utomated **I**nsights for **S**hooting **C**oaching — an end‑to‑end pipeline for analyzing basketball shooting form from video, extracting pose‑based kinematics, and training classifiers/LSTMs to predict shot outcomes and give actionable feedback.

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Repository Structure](#repository-structure)
* [Data & Schemas](#data--schemas)
* [Environment Setup](#environment-setup)
* [Quickstart](#quickstart)
* [Data Processing Pipeline](#data-processing-pipeline)
* [Training](#training)
* [Inference / Using the Models](#inference--using-the-models)
* [Visualization](#visualization)
* [Configuration](#configuration)
* [Troubleshooting](#troubleshooting)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [Acknowledgements](#acknowledgements)
* [License](#license)

---

## Overview

Codename‑AISC is a research/engineering project aimed at helping hoopers and coaches evaluate **shooting mechanics** from ordinary videos. The pipeline:

1. **Pose extraction** from raw videos (e.g., with MediaPipe/OpenCV).
2. **Kinematics derivation** (angles & y‑positions) per frame.
3. **Shot segmentation & release detection**.
4. **Per‑shot labeling** (Made / Missed or other custom labels).
5. **Dataset building** (tabular and sequential formats).
6. **Model training** (classic ML classifier and LSTM sequence model).
7. **Feedback generation**, feature importance, and simple plots.

This repo already includes **trained artifacts** (`shot_classifier.pkl`, `shot_lstm_model.h5`, label encoders) and **prepared datasets** to help you reproduce results quickly.

> **Note:** The name *AISC* here refers to the project codename and is unrelated to the structural‑steel organization.

---

## Features

* 🔁 **End‑to‑end pipeline**: from video → pose → features → datasets → models.
* 🧭 **Release frame detection**: uses wrist/elbow trajectories to tag the release frame.
* 🧩 **Two dataset modes**:

  * **Per‑shot tabular** (aggregated features)
  * **Per‑frame sequences** (for LSTM/RNN)
* 🤖 **Two model flavors**:

  * **Classical ML** (e.g., RandomForest/SVM) → `shot_classifier.pkl`
  * **LSTM** sequence model → `shot_lstm_model.h5`
* 🧪 **Ready‑to‑use datasets**: CSVs in the repo for fast experiments.
* 📈 **Explainability**: `feature_importance.png` for a quick sense of what matters.

---

## Repository Structure

```
Codename-AISC/
├─ analysis/                 # Notebooks / analysis scripts (exploration, EDA, eval)
├─ data_collection/          # Video → pose → features (collection & preprocessing)
├─ dataset/                  # Dataset builders/validators; sample data configs
├─ model_training/           # Training scripts for classic & LSTM models
├─ .gitignore
├─ README.md
├─ confirm-install.py        # Sanity check that deps installed & GPU/TF is visible
├─ feature_importance.png    # Example plot for classic model importance
├─ final_shooting_dataset.csv           # Aggregated per-shot features
├─ final_shot_dataset.csv               # (Alias/variant) per-shot features
├─ final_shot_sequence_dataset.csv      # Per-frame sequences (for LSTM)
├─ label_encoder.pkl
├─ lstm_label_encoder.pkl
├─ requirements.txt
├─ shot_classifier.pkl
└─ shot_lstm_model.h5
```

> Some folders may include additional scripts/notebooks for collection, labeling, and release detection.

---

## Data & Schemas

The project standardizes the following schemas across CSVs (feel free to extend):

### 1) **Per‑frame processed video CSVs**

Used during collection & sequence dataset building.

Required columns (one row per frame):

* `Elbow_Angle`, `Shoulder_Angle`, `Wrist_Angle`, `Hip_Angle`, `Knee_Angle`
* (Optional) `Frame`, `Timestamp_s`, `Wrist_Y`, `Elbow_Y`, `Video`

### 2) **Labeled shots CSV**

Global table aggregating all shots across videos.

Required columns:

* `Video`, `Shot_Number`, `Label`
  Where `Label` ∈ {`Made`, `Missed`} or your custom taxonomy.

### 3) **Shot release CSV**

Capture the detected release frame for each shot.

Required columns:

* `Video Name`, `Frame Number`, `Timestamp (s)`, `Wrist Y`, `Elbow Y`

### 4) **Final per‑shot dataset** (`final_shooting_dataset.csv` / `final_shot_dataset.csv`)

Aggregated features per shot (e.g., means, peaks, deltas around release).
Expected columns include angle summary stats and engineered features.

### 5) **Final sequence dataset** (`final_shot_sequence_dataset.csv`)

Each row encodes a sequence window across frames for a single shot, with an associated label.

---

## Environment Setup

> Conda users: the project commonly uses an env like `tensorflowAISC`.

```bash
# 1) Create & activate a fresh env (example)
conda create -n tensorflowAISC python=3.10 -y
conda activate tensorflowAISC

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) (Optional) Verify install
python confirm-install.py
```

**Typical dependencies** (see `requirements.txt`):

* numpy, pandas, scikit-learn, matplotlib
* opencv-python, mediapipe
* tensorflow/keras

> GPU support (optional): install CUDA/cuDNN per your TensorFlow version.

---

## Quickstart

1. **Prepare input videos** (practice clips; camera should see the whole shooter).
2. **Run data collection & processing**

   * Extract pose & compute angles per frame
   * Detect shot segments & the release frame
   * Generate per‑shot labels (Made/Missed)
3. **Build datasets** (tabular & sequence CSVs)
4. **Train models** (classic & LSTM)
5. **Run inference** on new clips and visualize feedback

If you just want to **play with the trained models**, skip to [Inference](#inference--using-the-models).

---

## Data Processing Pipeline

Below is a reference flow you can adapt to your scripts inside `data_collection/` and `dataset/`.

```text
Raw videos
  └─> Pose extraction (MediaPipe + OpenCV)
        └─> Per-frame angles & positions
              ├─> Shot segmentation
              ├─> Release detection (Wrist Y / Elbow Y dynamics)
              └─> Shot-level aggregation
                      ├─> final_shooting_dataset.csv (per-shot)
                      └─> final_shot_sequence_dataset.csv (per-frame sequences)
```

**Tips & heuristics** (tweak as needed):

* Smooth pose streams with a small moving average (3–5 frames).
* Use velocity/acceleration thresholds on `Wrist_Y` for potential release.
* Keep sequence windows centered on release (e.g., `[-15, +15]` frames).
* Normalize angles per subject (z‑score) before training sequence models.

---

## Training

> Training scripts typically live in `model_training/`.

### Classical ML (tabular)

* Input: `final_shooting_dataset.csv`
* Target: `Label` (encoded via `label_encoder.pkl`)
* Example model: RandomForest (saved to `shot_classifier.pkl`)

**Suggested CLI** (adapt to your script names):

```bash
python model_training/train_classic.py \
  --input final_shooting_dataset.csv \
  --model-out shot_classifier.pkl \
  --label-encoder-out label_encoder.pkl \
  --test-size 0.2 --random-state 42
```

### LSTM (sequence)

* Input: `final_shot_sequence_dataset.csv`
* Target: `Label` (encoded via `lstm_label_encoder.pkl`)
* Output: `shot_lstm_model.h5`

**Suggested CLI**:

```bash
python model_training/train_lstm.py \
  --input final_shot_sequence_dataset.csv \
  --model-out shot_lstm_model.h5 \
  --label-encoder-out lstm_label_encoder.pkl \
  --epochs 40 --batch-size 32 --lr 1e-3
```

> **Note:** If your scripts are named differently, keep the arguments but change the filenames accordingly.

---

## Inference / Using the Models

Here’s a minimal example to load and use the shipped models.

### 1) Classic classifier (`shot_classifier.pkl`)

```python
import joblib
import pandas as pd

# Load model & encoder
clf = joblib.load('shot_classifier.pkl')
le  = joblib.load('label_encoder.pkl')

# Load per-shot features to score
X = pd.read_csv('final_shooting_dataset.csv').drop(columns=['Label'])
y_pred = clf.predict(X)
labels = le.inverse_transform(y_pred)
print(labels[:10])
```

### 2) LSTM sequence model (`shot_lstm_model.h5`)

```python
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

model = load_model('shot_lstm_model.h5')
le = joblib.load('lstm_label_encoder.pkl')

# Suppose you’ve built a (N, T, F) sequence array from a new clip
sequences = np.load('my_new_sequences.npy')  # shape: [num_shots, timesteps, features]
probs = model.predict(sequences)
classes = probs.argmax(axis=1)
labels = le.inverse_transform(classes)
print(labels[:10])
```

---

## Visualization

* `feature_importance.png` — quick peek into which engineered features matter for the classic model.
* Consider adding:

  * Per‑frame angle plots around release
  * Confusion matrices & PR curves for both models

---

## Configuration

Create a project config (YAML/TOML/JSON) to centralize paths and hyper‑params, e.g.:

```yaml
paths:
  raw_videos: data/raw_videos
  processed:  data/processed
  datasets:   data/datasets
  models:     models

training:
  random_state: 42
  test_size: 0.2
  lstm:
    epochs: 40
    batch_size: 32
    learning_rate: 0.001
```

---

## Troubleshooting

* **Mediapipe not detecting pose:** ensure subject is fully visible; increase input resolution.
* **TensorFlow GPU not used:** check CUDA/cuDNN versions; run `confirm-install.py` to print device info.
* **Label mismatch errors:** always load the matching `label_encoder.pkl` saved alongside a model.
* **Sequence shapes wrong:** verify `(N, T, F)` order and consistent `T` across samples.

---

## Roadmap

* ✅ Ship baseline classic & LSTM models
* ✅ Provide ready‑made CSV datasets
* ⏳ Add CLI wrappers for full video→inference pipeline
* ⏳ Add richer visual feedback overlays on video
* ⏳ Expand labels beyond Made/Missed (e.g., *short*, *left*, *right*)

---

## Contributing

PRs welcome! Please:

1. Open an issue describing the change.
2. Use a feature branch.
3. Add/adjust docs and tests as needed.

---

## Acknowledgements

* Pose estimation via **MediaPipe** + **OpenCV**
* Deep learning via **TensorFlow/Keras**
* Classical ML via **scikit‑learn**

---

## License

No license specified yet. Consider adding an OSS license (e.g., MIT) at the repo root.
