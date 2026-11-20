"""
Heart Disease Prediction Model Training
========================================
Dataset : Heart Disease Dataset (UCI / Kaggle)  → data/heart.csv
          (If your file is named differently, update the path below)
Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach,
          exang, oldpeak, slope, ca, thal
Target  : target  (0 = No Heart Disease, 1 = Heart Disease)
Algorithm: Logistic Regression
Output  : models/heart_disease_model.sav
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
print("Loading dataset...")
# Try common file names for the heart disease dataset
import os
for fname in ["data/heart.csv", "data/heart_disease.csv", "data/heart_dataset.csv"]:
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        print(f"Loaded: {fname}")
        break
else:
    raise FileNotFoundError(
        "Heart disease dataset not found. Place it at data/heart.csv\n"
        "Download from: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset"
    )

print(f"Dataset shape: {df.shape}")
print(df.head())

# ─── 2. FEATURES & TARGET ────────────────────────────────────────────────────
feature_cols = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
target_col = "target"

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
print("\nTraining Logistic Regression model...")
model = LogisticRegression(C=1.0, max_iter=100, solver="lbfgs", random_state=42)
model.fit(X_train_scaled, y_train)

# ─── 6. EVALUATION ───────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Heart Disease", "Heart Disease"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─── 7. SAVE MODEL ───────────────────────────────────────────────────────────
joblib.dump(model, "models/heart_disease_model.sav")
print("\n✅ Model saved to models/heart_disease_model.sav")
