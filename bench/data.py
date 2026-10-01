"""Dataset loaders.

Each loader returns (X_train, X_test, y_train, y_test) as host (NumPy / SciPy) arrays.
Preprocessing is identical for every implementation so that timing and accuracy
differences come from the implementation only. Anything fitted on data (feature
selection, TF-IDF vocabulary) is fitted on the training split only.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def _downcast(df):
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["integer"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def _split(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)


def load_active_wiretap(data_dir):
    from sklearn.feature_selection import SelectKBest, f_classif

    df = pd.read_csv(os.path.join(data_dir, "Active_Wiretap_dataset.csv"), header=None, low_memory=False)
    labels = pd.read_csv(os.path.join(data_dir, "Active_Wiretap_labels.csv"),
                         names=["label"], header=None, low_memory=False)
    n = min(len(df), len(labels))
    df = _downcast(df.iloc[:n].copy())
    df["label"] = pd.to_numeric(labels["label"].iloc[:n], errors="coerce").values
    df = df.dropna()

    X = df.drop(columns="label").to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    X_train, X_test, y_train, y_test = _split(X, y, test_size=0.3)

    selector = SelectKBest(score_func=f_classif, k=min(30, X.shape[1]))
    X_train = selector.fit_transform(X_train, y_train).astype(np.float32)
    X_test = selector.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train, y_test


def load_cyberbully_detection(data_dir):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder

    data = pd.read_csv(os.path.join(data_dir, "final_hateXplain.csv"),
                       dtype={"comment": "string", "label": "category"})
    text = data["comment"].fillna("").astype(str).to_numpy()
    y = LabelEncoder().fit_transform(data["label"]).astype(np.int32)
    text_train, text_test, y_train, y_test = _split(text, y, test_size=0.2)

    tfidf = TfidfVectorizer(max_features=5000, dtype=np.float32)
    X_train = tfidf.fit_transform(text_train)
    X_test = tfidf.transform(text_test)
    return X_train, X_test, y_train, y_test


def load_diabetes(data_dir):
    data = pd.read_csv(os.path.join(data_dir, "diabetes_dataset.csv"))
    data = data.fillna(data.mean(numeric_only=True))
    for col in ["gender", "smoking_history"]:
        if col in data.columns:
            data[col] = data[col].fillna("Unknown").astype("category").cat.codes
    X = data.drop(columns="diabetes").astype(np.float32).to_numpy()
    y = data["diabetes"].to_numpy(dtype=np.int32)
    return _split(X, y, test_size=0.2)


def load_loan_prediction(data_dir):
    data = pd.read_csv(os.path.join(data_dir, "Loan Prediction.csv"))
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            data[col] = data[col].fillna(data[col].median())
        else:
            data[col] = data[col].fillna(data[col].mode()[0])
    categorical = ["Married/Single", "House_Ownership", "Car_Ownership", "Profession", "CITY", "STATE"]
    data = pd.get_dummies(data, columns=categorical, drop_first=True)
    X = data.drop(columns=["Risk_Flag", "Id"]).astype(np.float32).to_numpy()
    y = data["Risk_Flag"].to_numpy(dtype=np.int32)
    return _split(X, y, test_size=0.2)


def load_student_performance(data_dir):
    data = pd.read_csv(os.path.join(data_dir, "student_performance_dataset.csv"))
    categorical = ["Gender", "Parental_Education_Level", "Internet_Access_at_Home",
                   "Extracurricular_Activities", "Pass_Fail"]
    for col in categorical:
        data[col] = data[col].astype("category").cat.codes
    X = data.drop(columns=["Student_ID", "Pass_Fail"]).astype(np.float32).to_numpy()
    y = data["Pass_Fail"].to_numpy(dtype=np.int32)
    return _split(X, y, test_size=0.2)


# Keys match the "Dataset" names used in the result CSVs and chart file names.
DATASETS = {
    "active_Wiretap": load_active_wiretap,
    "cyberbully detection": load_cyberbully_detection,
    "diabetes": load_diabetes,
    "loan prediction": load_loan_prediction,
    "student_performance": load_student_performance,
}
