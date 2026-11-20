"""
General Disease Prediction Model Training (XGBoost)
=====================================================
Dataset : data/clean_dataset.tsv  (already present)
          Symptom-disease mapping across 132 diseases, 377 symptoms
Features: 377 binary symptom columns
Target  : prognosis (disease name, encoded as integer category)
Algorithm: XGBClassifier (XGBoost)
Output  : model/xgboost_model.json
Note    : This model is used via the DiseaseModel class in code/DiseaseModel.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from core.DiseaseModel import DiseaseModel

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("data/clean_dataset.tsv", sep="\t")
print(f"Dataset shape: {df.shape}")

# ─── 2. FEATURES & TARGET ────────────────────────────────────────────────────
X = df.iloc[:, :-1]   # All symptom columns
y = df.iloc[:, -1]    # 'prognosis' column

print(f"\nNumber of unique diseases: {y.nunique()}")
print(f"Number of symptom features: {X.shape[1]}")

# Encode target as integer categories
y_cat = y.astype("category")
diseases = y_cat.cat.categories
y_encoded = y_cat.cat.codes

# ─── 3. TRAIN-TEST SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─── 4. MODEL TRAINING ───────────────────────────────────────────────────────
print("\nTraining XGBClassifier...")
model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=len(diseases),
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    verbosity=1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=10,
)

# ─── 5. EVALUATION ───────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")

# Decode predictions back to disease names for a sample report
y_test_names = [diseases[i] for i in y_test]
y_pred_names = [diseases[i] for i in y_pred]
print("\nSample Predictions vs Actual (first 10):")
for actual, predicted in zip(y_test_names[:10], y_pred_names[:10]):
    status = "✅" if actual == predicted else "❌"
    print(f"  {status} Actual: {actual:40s} | Predicted: {predicted}")

# ─── 6. SAVE MODEL ───────────────────────────────────────────────────────────
model.save_model("model/xgboost_model.json")
print("\n✅ Model saved to model/xgboost_model.json")

# ─── 7. VERIFY via DiseaseModel ──────────────────────────────────────────────
print("\nVerifying via DiseaseModel.load_xgboost()...")
dm = DiseaseModel()
dm.load_xgboost("model/xgboost_model.json")
print("✅ DiseaseModel loaded successfully.")
