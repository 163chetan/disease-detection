"""
Lung Cancer Prediction Model Training
=======================================
Dataset : Lung Cancer Survey Dataset  →  data/lung_cancer.csv  (already present)
Features: GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE,
          CHRONICDISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOLCONSUMING,
          COUGHING, SHORTNESSOFBREATH, SWALLOWINGDIFFICULTY, CHESTPAIN
Target  : LUNG_CANCER  ('YES' / 'NO')
Algorithm: Pipeline → (StandardScaler + OneHotEncoder) → LogisticRegression
Output  : models/lung_cancer_model.sav
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("data/lung_cancer.csv")
print(f"Dataset shape: {df.shape}")
print(df.head())

# Standardise Gender values
df["GENDER"] = df["GENDER"].map({"M": "Male", "F": "Female"}).fillna(df["GENDER"])

# Replace YES/NO with numeric for binary symptom columns
binary_cols = [
    "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
    "CHRONICDISEASE", "FATIGUE", "ALLERGY", "WHEEZING",
    "ALCOHOLCONSUMING", "COUGHING", "SHORTNESSOFBREATH",
    "SWALLOWINGDIFFICULTY", "CHESTPAIN"
]
df[binary_cols] = df[binary_cols].replace({"NO": 1, "YES": 2})
df[binary_cols] = df[binary_cols].apply(pd.to_numeric, errors="coerce")

print(f"\nClass distribution:\n{df['LUNG_CANCER'].value_counts()}")

# ─── 2. FEATURES & TARGET ────────────────────────────────────────────────────
numeric_features = ["AGE"] + binary_cols
categorical_features = ["GENDER"]

X = df[numeric_features + categorical_features]
y = df["LUNG_CANCER"]

# ─── 3. TRAIN-TEST SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─── 4. BUILD PIPELINE ───────────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
])

# ─── 5. MODEL TRAINING ───────────────────────────────────────────────────────
print("\nTraining Pipeline (Preprocessor + LogisticRegression)...")
pipeline.fit(X_train, y_train)

# ─── 6. EVALUATION ───────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─── 7. SAVE MODEL ───────────────────────────────────────────────────────────
joblib.dump(pipeline, "models/lung_cancer_model.sav")
print("\n✅ Model saved to models/lung_cancer_model.sav")
