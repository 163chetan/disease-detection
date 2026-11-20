"""
Diabetes Disease Prediction Model Training
==========================================
Dataset : PIMA Indian Diabetes Dataset  (data/dataset.csv)
Features: Pregnancies, Glucose, BloodPressure, SkinThickness,
          Insulin, BMI, DiabetesPedigreeFunction, Age
Target  : Outcome  (0 = No Diabetes, 1 = Diabetic)
Algorithm: Support Vector Classifier (SVC) with linear kernel
Output  : models/diabetes_model.sav
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("data/dataset.csv")
print(f"Dataset shape: {df.shape}")
print(df.head())

# ─── 2. FEATURES & TARGET ────────────────────────────────────────────────────
feature_cols = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
X = df[feature_cols]
y = df["Outcome"]

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
model = SVC(kernel="linear", C=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

# ─── 6. EVALUATION ───────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetic"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─── 7. SAVE MODEL ───────────────────────────────────────────────────────────
joblib.dump(model, "models/diabetes_model.sav")
print("\n✅ Model saved to models/diabetes_model.sav")
