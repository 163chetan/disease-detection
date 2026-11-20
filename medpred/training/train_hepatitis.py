"""
Hepatitis Prediction Model Training
=====================================
Dataset : HCV / Hepatitis Dataset (UCI)
          Download: https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset
          Place at: data/hepatitis.csv  (or data/HepatitisCdata.csv)
Features: Age, Sex, ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT
Target  : Category  (blood donors vs hepatitis patients → binary 0/1)
Algorithm: RandomForestClassifier (n_estimators=100)
Output  : models/hepititisc_model.sav
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
print("Loading dataset...")
for fname in ["data/hepatitis.csv", "data/HepatitisCdata.csv",
              "data/hepatitis_c.csv", "data/hcv_data.csv"]:
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        print(f"Loaded: {fname}")
        break
else:
    raise FileNotFoundError(
        "Hepatitis dataset not found.\n"
        "Download from: https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset\n"
        "Place it at: data/hepatitis.csv"
    )

print(f"Dataset shape: {df.shape}")
print(df.head())

# ─── 2. PREPROCESSING ────────────────────────────────────────────────────────
# Drop unnamed index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Encode Sex: male=1, female=2
if df["Sex"].dtype == object:
    df["Sex"] = df["Sex"].map({"m": 1, "f": 2, "M": 1, "F": 2}).fillna(1)

# Encode Category (target): 0=Blood Donor, 1=Hepatitis/Suspect
# Categories: '0=Blood Donor', '0s=suspect Blood Donor', '1=Hepatitis', '2=Fibrosis', '3=Cirrhosis'
if "Category" in df.columns:
    df["target"] = df["Category"].apply(
        lambda x: 0 if str(x).startswith("0") else 1
    )
    df = df.drop(columns=["Category"])
elif "class" in df.columns:
    df = df.rename(columns={"class": "target"})
else:
    raise ValueError("Could not find target column ('Category' or 'class') in dataset.")

feature_cols = ["Age", "Sex", "ALB", "ALP", "ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"]
# Keep only columns that exist
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols]
y = df["target"]

# Fill NaN with median
X = X.fillna(X.median())

print(f"\nClass distribution:\n{y.value_counts()}")

# ─── 3. TRAIN-TEST SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─── 4. MODEL TRAINING ───────────────────────────────────────────────────────
print("\nTraining RandomForestClassifier model...")
model = RandomForestClassifier(n_estimators=100, criterion="gini", random_state=42)
model.fit(X_train, y_train)

# ─── 5. EVALUATION ───────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Healthy/Donor", "Hepatitis"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─── 6. SAVE MODEL ───────────────────────────────────────────────────────────
joblib.dump(model, "models/hepititisc_model.sav")
print("\n✅ Model saved to models/hepititisc_model.sav")
