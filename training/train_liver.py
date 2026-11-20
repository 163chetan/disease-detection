"""
Liver Disease Prediction Model Training
=========================================
Dataset : Indian Liver Patient Dataset (ILPD)
          Download: https://www.kaggle.com/datasets/uciml/indian-liver-patient-records
          Place at: data/liver.csv
Features: Gender, Age, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
          Alamine_Aminotransferase, Aspartate_Aminotransferase,
          Total_Protiens, Albumin, Albumin_and_Globulin_Ratio
Target  : Dataset  (1 = Liver Disease, 2 = No Liver Disease → mapped to 1/0)
Algorithm: LogisticRegression
Output  : models/liver_model.sav
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
print("Loading dataset...")
for fname in ["data/liver.csv", "data/indian_liver_patient.csv",
              "data/liver_patient.csv", "data/ILPD.csv"]:
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        print(f"Loaded: {fname}")
        break
else:
    raise FileNotFoundError(
        "Liver disease dataset not found.\n"
        "Download from: https://www.kaggle.com/datasets/uciml/indian-liver-patient-records\n"
        "Place it at: data/liver.csv"
    )

print(f"Dataset shape: {df.shape}")
print(df.head())

# ─── 2. PREPROCESSING ────────────────────────────────────────────────────────
# Column rename support for different CSV formats
col_map = {
    "Gender": "Gender",
    "Age": "Age",
    "Total_Bilirubin": "Total_Bilirubin",
    "Direct_Bilirubin": "Direct_Bilirubin",
    "Alkaline_Phosphotase": "Alkaline_Phosphotase",
    "Alamine_Aminotransferase": "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase": "Aspartate_Aminotransferase",
    "Total_Protiens": "Total_Protiens",
    "Albumin": "Albumin",
    "Albumin_and_Globulin_Ratio": "Albumin_and_Globulin_Ratio",
    "Dataset": "Dataset",
}

# Encode Gender: Male=0, Female=1
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1, "M": 0, "F": 1}).fillna(0)

# Target: 1=Liver Disease → keep as 1; 2=No Disease → 0
if "Dataset" in df.columns:
    df["target"] = df["Dataset"].map({1: 1, 2: 0})
elif "is_patient" in df.columns:
    df["target"] = df["is_patient"].map({1: 1, 2: 0})
else:
    raise ValueError("Could not find target column in liver dataset.")

feature_cols = [
    "Gender", "Age", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase", "Total_Protiens",
    "Albumin", "Albumin_and_Globulin_Ratio"
]
# Only keep existing cols
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols]
y = df["target"]

# Fill NaN (Albumin_and_Globulin_Ratio sometimes has NaN)
X = X.fillna(X.median())
y = y.dropna()
X = X.loc[y.index]

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
model = LogisticRegression(C=1.0, max_iter=100, solver="lbfgs")
model.fit(X_train_scaled, y_train)

# ─── 6. EVALUATION ───────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Liver Disease", "Liver Disease"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─── 7. SAVE MODEL ───────────────────────────────────────────────────────────
joblib.dump(model, "models/liver_model.sav")
print("\n✅ Model saved to models/liver_model.sav")
