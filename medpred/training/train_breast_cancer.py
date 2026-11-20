"""
Breast Cancer Prediction Model Training
=========================================
Dataset : Wisconsin Breast Cancer Dataset (sklearn built-in or CSV)
          Alternatively place your CSV at: data/breast_cancer.csv
Features: 30 tumor cell nucleus measurement features
          (radius, texture, perimeter, area, smoothness, compactness,
           concavity, concave_points, symmetry, fractal_dimension)
           — mean, SE, and worst values for each.
Target  : diagnosis  (0 = Benign, 1 = Malignant)
Algorithm: StackingClassifier
           - Base: RandomForest, GradientBoosting, AdaBoost, SVM (linear)
           - Meta: RandomForestClassifier
Output  : models/breast_cancer.sav
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
print("Loading dataset...")

if os.path.exists("data/breast_cancer.csv"):
    df = pd.read_csv("data/breast_cancer.csv")
    # Drop ID column if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    # Encode diagnosis: M=1, B=0
    if df["diagnosis"].dtype == object:
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    target_col = "diagnosis"
    X = df.drop(columns=[target_col])
    y = df[target_col]
else:
    # Fallback: use sklearn's built-in breast cancer dataset
    print("data/breast_cancer.csv not found. Using sklearn built-in dataset.")
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)   # 0=malignant, 1=benign (sklearn inverts labels)
    # Re-map so 1 = Malignant to match the app's convention
    y = 1 - y

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"\nClass distribution:\n{y.value_counts()}")

# ─── 2. TRAIN-TEST SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─── 3. SCALE FEATURES ───────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─── 4. BUILD STACKING CLASSIFIER ────────────────────────────────────────────
base_estimators = [
    ("Random Forest",      RandomForestClassifier(n_estimators=100, random_state=42)),
    ("Gradient Boosting",  GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                       max_depth=3, random_state=42)),
    ("AdaBoost",           AdaBoostClassifier(n_estimators=50, learning_rate=1.0,
                                               random_state=42)),
    ("SVM",                SVC(kernel="linear", C=1.0, random_state=42, probability=False)),
]

final_estimator = RandomForestClassifier(n_estimators=100, random_state=42)

model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=final_estimator,
    cv=None,
    passthrough=False,
    verbose=1,
)

# ─── 5. MODEL TRAINING ───────────────────────────────────────────────────────
print("\nTraining StackingClassifier (RF + GB + AdaBoost + SVM → RF)...")
model.fit(X_train_scaled, y_train)

# ─── 6. EVALUATION ───────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─── 7. SAVE MODEL ───────────────────────────────────────────────────────────
joblib.dump(model, "models/breast_cancer.sav")
print("\n✅ Model saved to models/breast_cancer.sav")
