"""
Chronic Kidney Disease Prediction Model Training
==================================================
Dataset : Chronic Kidney Disease Dataset (UCI)
          Download: https://www.kaggle.com/datasets/mansoordaku/ckdisease
          Place at: data/kidney_disease.csv
Features: age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc,
          sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane
Target  : classification  ('ckd' / 'notckd')
Algorithm: LogisticRegression (random_state=42)
Output  : models/chronic_model.sav
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
print("Loading dataset...")
for fname in ["data/kidney_disease.csv", "data/chronic_kidney.csv", "data/ckd.csv"]:
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        print(f"Loaded: {fname}")
        break
else:
    raise FileNotFoundError(
        "Chronic Kidney Disease dataset not found.\n"
        "Download from: https://www.kaggle.com/datasets/mansoordaku/ckdisease\n"
        "Place it at: data/kidney_disease.csv"
    )

print(f"Dataset shape: {df.shape}")

# ─── 2. PREPROCESSING ────────────────────────────────────────────────────────
# Strip whitespace from all string values
df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

# Replace '?' with NaN
df = df.replace("?", np.nan)

feature_cols = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane"
]
target_col = "classification"

# Encode categorical columns used in app
cat_map = {
    "rbc":   {"normal": 1, "abnormal": 0},
    "pc":    {"normal": 1, "abnormal": 0},
    "pcc":   {"present": 1, "notpresent": 0},
    "ba":    {"present": 1, "notpresent": 0},
    "htn":   {"yes": 1, "no": 0},
    "dm":    {"yes": 1, "no": 0},
    "cad":   {"yes": 1, "no": 0},
    "appet": {"good": 1, "poor": 0},
    "pe":    {"yes": 1, "no": 0},
    "ane":   {"yes": 1, "no": 0},
}
for col, mapping in cat_map.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# Convert remaining cols to numeric
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Encode target
df[target_col] = df[target_col].str.strip().map({"ckd": 1, "notckd": 0})

# Drop rows with missing target
df = df.dropna(subset=[target_col])

X = df[feature_cols]
y = df[target_col].astype(int)

# Fill remaining NaNs with column median
X = X.fillna(X.median())

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
print("\nTraining LogisticRegression model...")
model = LogisticRegression(C=1.0, max_iter=100, solver="lbfgs", random_state=42)
model.fit(X_train_scaled, y_train)

# ─── 6. EVALUATION ───────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No CKD", "CKD"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─── 7. SAVE MODEL ───────────────────────────────────────────────────────────
joblib.dump(model, "models/chronic_model.sav")
print("\n✅ Model saved to models/chronic_model.sav")
