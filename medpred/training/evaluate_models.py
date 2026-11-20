"""
Model accuracy evaluation  ->  models/metrics.json
===================================================
The app shows these numbers after every prediction.

For every model we take the *exact estimator configuration saved in models/*, re-fit a fresh copy on the
dataset in data/ inside 5-fold stratified cross-validation, and record the mean accuracy. Cross-validation
means every score comes from rows the model never saw while fitting, so it is an honest estimate (the saved
.sav files themselves are never modified).

Run from the project root:   python training/evaluate_models.py [model ...]
e.g.                         python training/evaluate_models.py lung breast
Models whose dataset is missing are skipped (and keep any previous result in metrics.json).

Dataset files expected in data/ (same names as the training README):
  lung      data/lung_cancer.csv        breast    sklearn built-in (or data/breast_cancer.csv)
  diabetes  data/diabetes.csv           heart     data/heart.csv
  parkinson data/parkinsons.csv         kidney    data/kidney_disease.csv
  hepatitis data/hepatitis.csv          liver     data/liver.csv
  symptom   data/clean_dataset.tsv
"""
import datetime as dt
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")
OUT = os.path.join("models", "metrics.json")
FOLDS = 5


def first_existing(*names):
    for n in names:
        p = os.path.join("data", n)
        if os.path.exists(p):
            return p
    return None


# ───────────────────────── data loaders: each returns (X, y, dataset_label) ─────────────────────────
def load_lung():
    p = first_existing("lung_cancer.csv")
    if not p: return None
    df = pd.read_csv(p)
    df["GENDER"] = df["GENDER"].map({"M": "Male", "F": "Female"}).fillna(df["GENDER"])
    df.columns = df.columns.str.strip()
    bin_cols = [c for c in df.columns if c not in ("GENDER", "AGE", "LUNG_CANCER")]
    df[bin_cols] = df[bin_cols].replace({"NO": 1, "YES": 2}).apply(pd.to_numeric, errors="coerce")
    X = df.drop(columns=["LUNG_CANCER"])
    return X, df["LUNG_CANCER"], f"Lung cancer survey dataset ({len(df)} records)"


def load_breast():
    p = first_existing("breast_cancer.csv")
    if p:
        df = pd.read_csv(p).drop(columns=["id"], errors="ignore")
        y = df["diagnosis"].map({"M": 1, "B": 0}) if df["diagnosis"].dtype == object else df["diagnosis"]
        return df.drop(columns=["diagnosis"]), y, f"Wisconsin breast cancer dataset ({len(df)} samples)"
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return pd.DataFrame(d.data, columns=d.feature_names), pd.Series(1 - d.target), \
        "Wisconsin breast cancer dataset (569 samples)"


def load_diabetes():
    p = first_existing("diabetes.csv")
    if not p: return None
    df = pd.read_csv(p)
    return df.drop(columns=["Outcome"]), df["Outcome"], f"PIMA Indians diabetes dataset ({len(df)} patients)"


def load_heart():
    p = first_existing("heart.csv", "heart_disease.csv")
    if not p: return None
    df = pd.read_csv(p, encoding="utf-8-sig")
    return df.drop(columns=["target"]), df["target"], f"Heart disease dataset ({len(df)} patients)"


def load_parkinson():
    p = first_existing("parkinsons.csv", "parkinsons_data.csv", "parkinson.csv")
    if not p: return None
    df = pd.read_csv(p).drop(columns=["name"], errors="ignore")
    return df.drop(columns=["status"]), df["status"], f"Parkinson's voice dataset ({len(df)} recordings)"


def load_kidney():
    p = first_existing("kidney_disease.csv", "chronic_kidney.csv", "ckd.csv")
    if not p: return None
    df = pd.read_csv(p)
    df = df.apply(lambda c: c.str.strip() if c.dtype == "object" else c).replace("?", np.nan)
    cat = {"rbc": {"normal": 1, "abnormal": 0}, "pc": {"normal": 1, "abnormal": 0},
           "pcc": {"present": 1, "notpresent": 0}, "ba": {"present": 1, "notpresent": 0},
           "htn": {"yes": 1, "no": 0}, "dm": {"yes": 1, "no": 0}, "cad": {"yes": 1, "no": 0},
           "appet": {"good": 1, "poor": 0}, "pe": {"yes": 1, "no": 0}, "ane": {"yes": 1, "no": 0}}
    for c, m in cat.items():
        df[c] = df[c].map(m)
    feats = ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod", "pot",
             "hemo", "pcv", "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane"]
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["classification"] = df["classification"].str.strip().map({"ckd": 1, "notckd": 0})
    df = df.dropna(subset=["classification"])
    X = df[feats]
    return X.fillna(X.median()), df["classification"].astype(int), f"Chronic kidney disease dataset ({len(df)} patients)"


def load_hepatitis():
    p = first_existing("hepatitis.csv", "HepatitisCdata.csv", "hepatitis_c.csv", "hcv_data.csv")
    if not p: return None
    df = pd.read_csv(p).drop(columns=["Unnamed: 0"], errors="ignore")
    if df["Sex"].dtype == object:
        df["Sex"] = df["Sex"].map({"m": 1, "f": 2, "M": 1, "F": 2}).fillna(1)
    y = df["Category"].apply(lambda x: 0 if str(x).startswith("0") else 1)
    feats = [c for c in ["Age", "Sex", "ALB", "ALP", "ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"] if c in df]
    X = df[feats]
    return X.fillna(X.median()), y, f"Hepatitis C dataset ({len(df)} patients)"


def load_liver():
    p = first_existing("liver.csv", "indian_liver_patient.csv", "liver_patient.csv", "ILPD.csv")
    if not p: return None
    df = pd.read_csv(p)
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1, "M": 0, "F": 1}).fillna(0)
    y = df["Dataset"].map({1: 1, 2: 0})
    feats = ["Gender", "Age", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
             "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Protiens", "Albumin",
             "Albumin_and_Globulin_Ratio"]
    X = df[feats]
    X = X.fillna(X.median())
    keep = y.dropna().index
    return X.loc[keep], y.loc[keep].astype(int), f"Indian liver patient dataset ({len(keep)} patients)"


def load_symptom():
    p = first_existing("clean_dataset.tsv")
    if not p: return None
    df = pd.read_csv(p, sep="\t")
    y = df.iloc[:, -1].astype("category").cat.codes
    return df.iloc[:, :-1], y, f"Symptom–disease dataset ({len(df)} cases, {df.iloc[:, -1].nunique()} diseases)"


def load_estimator(key):
    if key == "symptom":
        import xgboost as xgb
        m = xgb.XGBClassifier(); m.load_model(os.path.join("model", "xgboost_model.json")); return m
    f = dict(diabetes="diabetes_model", heart="heart_disease_model", parkinson="parkinsons_model",
             liver="liver_model", hepatitis="hepititisc_model", lung="lung_cancer_model",
             kidney="chronic_model", breast="breast_cancer")[key]
    return joblib.load(os.path.join("models", f"{f}.sav"))


# Short caveats shown under the accuracy figure in the app (applied to every stored result on each run)
NOTES = {
    "symptom": "This dataset is highly idealised, so a near-perfect score overstates real-world accuracy.",
    "diabetes": "Evaluated on the PIMA dataset (adult women of Pima heritage); other populations may differ.",
}

LOADERS = dict(lung=load_lung, breast=load_breast, diabetes=load_diabetes, heart=load_heart,
               parkinson=load_parkinson, kidney=load_kidney, hepatitis=load_hepatitis, liver=load_liver,
               symptom=load_symptom)


def main(keys):
    results = {}
    if os.path.exists(OUT):
        results = json.load(open(OUT))
    for key in keys:
        t0 = time.time()
        try:
            data = LOADERS[key]()
            if data is None:
                print(f"- {key:10s} skipped (dataset not found in data/)")
                continue
            X, y, label = data
            est = clone(load_estimator(key))
            scores = cross_val_score(est, X, y, cv=StratifiedKFold(FOLDS, shuffle=True, random_state=42),
                                     scoring="accuracy", n_jobs=1)
            results[key] = dict(accuracy=round(float(scores.mean()), 4), std=round(float(scores.std()), 4),
                                folds=FOLDS, samples=int(len(y)), dataset=label,
                                method=f"{FOLDS}-fold cross-validation",
                                evaluated=dt.date.today().isoformat())
            print(f"✓ {key:10s} accuracy {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}  ({label})  {time.time()-t0:.0f}s")
        except Exception as e:  # keep going; one bad dataset must not stop the others
            print(f"✗ {key:10s} failed: {type(e).__name__}: {str(e)[:150]}")
        for k, note in NOTES.items():
            if k in results:
                results[k]["note"] = note
        json.dump(results, open(OUT, "w"), indent=2)
    print("saved ->", OUT)


if __name__ == "__main__":
    main(sys.argv[1:] or list(LOADERS))
