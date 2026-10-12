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


# The full Active Wiretap capture is ~2.28M rows / 6.2 GB. Training the slower models on
# all of it (minutes per fit) makes a 9-repeat sweep take hours, so we draw a stratified
# subsample. Set to None to use every row.
WIRETAP_MAX_ROWS = None


def load_active_wiretap(data_dir):
    from sklearn.feature_selection import SelectKBest, f_classif

    # The labels CSV has a header row ("","x") and a leading index column; read it normally
    # and take the label column so rows stay aligned 1:1 with the dataset (which has neither).
    labels = pd.read_csv(os.path.join(data_dir, "Active_Wiretap_labels.csv"))
    y_all = pd.to_numeric(labels.iloc[:, -1], errors="coerce").to_numpy()
    valid = np.arange(len(y_all))[~np.isnan(y_all)]

    keep = valid
    if WIRETAP_MAX_ROWS is not None and len(valid) > WIRETAP_MAX_ROWS:
        keep, _ = train_test_split(valid, train_size=WIRETAP_MAX_ROWS,
                                   random_state=RANDOM_STATE, stratify=y_all[valid])
    keep = np.sort(keep)
    keep_set = set(keep.tolist())

    # Read only the kept rows from the big dataset, chunk by chunk, so we never hold all
    # 6.2 GB in memory. Chunks preserve file order, so the result stays aligned with `keep`.
    parts, start = [], 0
    for chunk in pd.read_csv(os.path.join(data_dir, "Active_Wiretap_dataset.csv"),
                             header=None, low_memory=False, chunksize=200_000):
        wanted = [i for i, g in enumerate(range(start, start + len(chunk))) if g in keep_set]
        if wanted:
            parts.append(chunk.iloc[wanted])
        start += len(chunk)
    df = _downcast(pd.concat(parts).reset_index(drop=True))
    y = y_all[keep]

    row_ok = df.notna().all(axis=1).to_numpy() & ~np.isnan(y)
    X = df.to_numpy(dtype=np.float32)[row_ok]
    y = y[row_ok].astype(np.int32)
    X_train, X_test, y_train, y_test = _split(X, y, test_size=0.3)

    selector = SelectKBest(score_func=f_classif, k=min(30, X.shape[1]))
    # f_classif sums and squares each column; in float32 that overflows at this row count
    # (~1.6M rows), yielding NaN F-scores and an arbitrary feature choice. Fit the ANOVA
    # F-test in float64 so the scores are valid, then keep the selected columns in float32.
    selector.fit(X_train.astype(np.float64), y_train)
    X_train = selector.transform(X_train).astype(np.float32)
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
