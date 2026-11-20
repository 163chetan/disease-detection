# MedPredict — Model Training Scripts

This folder contains the full Python training code for every ML model used in the app.
Run any script from the **project root** (`MedPredict-Clean/`) using the venv Python.

---

## Model Summary

| Script | Model File | Algorithm | Dataset Needed |
|---|---|---|---|
| `train_disease_xgboost.py` | `model/xgboost_model.json` | XGBClassifier | `data/clean_dataset.tsv` ✅ already present |
| `train_diabetes.py` | `models/diabetes_model.sav` | SVC (linear) | `data/dataset.csv` ✅ already present |
| `train_heart.py` | `models/heart_disease_model.sav` | LogisticRegression | `data/heart.csv` ⬇️ download needed |
| `train_parkinsons.py` | `models/parkinsons_model.sav` | SVC (linear) | `data/parkinsons.csv` ⬇️ download needed |
| `train_lung_cancer.py` | `models/lung_cancer_model.sav` | Pipeline + LR | `data/lung_cancer.csv` ✅ already present |
| `train_breast_cancer.py` | `models/breast_cancer.sav` | StackingClassifier | `data/breast_cancer.csv` or sklearn built-in |
| `train_chronic_kidney.py` | `models/chronic_model.sav` | LogisticRegression | `data/kidney_disease.csv` ⬇️ download needed |
| `train_hepatitis.py` | `models/hepititisc_model.sav` | RandomForestClassifier | `data/hepatitis.csv` ⬇️ download needed |
| `train_liver.py` | `models/liver_model.sav` | LogisticRegression | `data/liver.csv` ⬇️ download needed |

---

## How to Run

```powershell
# From the project root (MedPredict-Clean/)
.\venv\Scripts\python training\train_diabetes.py
.\venv\Scripts\python training\train_heart.py
# ... etc
```

---

## Datasets to Download

| Dataset | Kaggle Link |
|---|---|
| Heart Disease | https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset |
| Parkinson's | https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set |
| Breast Cancer (optional) | sklearn built-in (no download needed) |
| Chronic Kidney Disease | https://www.kaggle.com/datasets/mansoordaku/ckdisease |
| Hepatitis C | https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset |
| Indian Liver Patient | https://www.kaggle.com/datasets/uciml/indian-liver-patient-records |

Place downloaded CSV files in the `data/` folder.

---

## Algorithm Reference

| Model | Algorithm | Why chosen |
|---|---|---|
| Diabetes | **SVC (linear kernel)** | Good for small datasets; works well on PIMA dataset |
| Heart Disease | **Logistic Regression** | Interpretable; strong baseline for binary clinical data |
| Parkinson's | **SVC (linear kernel)** | Effective for high-dimensional voice features |
| Lung Cancer | **Pipeline (Scaler + OneHotEncoder + LR)** | Handles mixed numeric + categorical data cleanly |
| Breast Cancer | **StackingClassifier** | Ensemble of RF + GB + AdaBoost + SVM → meta RF for best accuracy |
| Chronic Kidney | **Logistic Regression** | Simple, fast, interpretable; good with 24 binary/numeric features |
| Hepatitis | **Random Forest** | Handles missing values well; robust on small imbalanced datasets |
| Liver | **Logistic Regression** | Effective baseline for liver enzyme levels |
| General Disease | **XGBoost (multi-class)** | 132 diseases, 377 symptoms — gradient boosting handles scale well |
