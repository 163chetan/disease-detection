"""
Parkinson's Disease Prediction Model Training
==============================================
Dataset : Parkinson's Disease Dataset (UCI)
          Download: https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set
          Place at: data/parkinsons.csv
Features: 22 voice measurement features (MDVP, Jitter, Shimmer, NHR, HNR, RPDE, DFA, etc.)
Target  : status  (0 = Healthy, 1 = Parkinson's)
Algorithm: Support Vector Classifier (SVC) with linear kernel
Output  : models/parkinsons_model.sav
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
print("Loading dataset...")
for fname in ["data/parkinsons.csv", "data/parkinsons_data.csv", "data/parkinson.csv"]:
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        print(f"Loaded: {fname}")
        break
else:
    raise FileNotFoundError(
        "Parkinson's dataset not found. Place it at data/parkinsons.csv\n"
        "Download from: https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set"
    )

print(f"Dataset shape: {df.shape}")
print(df.head())

# ─── 2. FEATURES & TARGET ────────────────────────────────────────────────────
# Drop name column if present
if "name" in df.columns:
    df = df.drop(columns=["name"])

feature_cols = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]
target_col = "status"

X = df[feature_cols]
y = df[target_col]

print(f"\nClass distribution:\n{y.value_counts()}")

# ─── 3. TRAIN-TEST SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─── 4. FEATURE SCALING ──────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─── 5. MODEL TRAINING ───────────────────────────────────────────────────────
print("\nTraining SVC model...")
model = SVC(kernel="linear", C=1.0)
model.fit(X_train_scaled, y_train)

# ─── 6. EVALUATION ───────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson's"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─── 7. SAVE MODEL ───────────────────────────────────────────────────────────
joblib.dump(model, "models/parkinsons_model.sav")
print("\n✅ Model saved to models/parkinsons_model.sav")
