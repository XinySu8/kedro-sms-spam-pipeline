from __future__ import annotations
import pandas as pd

from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def ingest_sms_spam(sms_spam_raw: pd.DataFrame) -> pd.DataFrame:
    df = sms_spam_raw.copy()
    # Kaggle 常见 spam.csv 会有多余列：Unnamed: 2/3/4
    df = df.iloc[:, :2]
    df.columns = ["label", "text"]

    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str)

    df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)
    return df



def split_sms_spam(
    sms_spam_clean: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Input: cleaned dataframe with at least [text, label] columns.
    Output: X_train, X_val, X_test, y_train, y_val, y_test (all as DataFrames).
    """
    df = sms_spam_clean.copy()

    # Common dataset columns: 'v1' (label) and 'v2' (text) or already 'label','text'
    if "text" not in df.columns:
        if "v2" in df.columns:
            df = df.rename(columns={"v2": "text"})
    if "label" not in df.columns:
        if "v1" in df.columns:
            df = df.rename(columns={"v1": "label"})

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected columns 'text' and 'label'. Got: {list(df.columns)}")

    # Normalize label to 0/1 (ham=0, spam=1)
    label_series = df["label"].astype(str).str.lower().str.strip()
    y = label_series.map({"ham": 0, "spam": 1})
    if y.isna().any():
        bad = sorted(set(label_series[y.isna()].tolist()))
        raise ValueError(f"Unknown labels found: {bad}. Expected only ham/spam.")

    X = df["text"].astype(str)

    test_size = float(params["test_size"])
    val_size = float(params["val_size"])
    random_state = int(params["random_state"])
    stratify = bool(params.get("stratify", True))

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    # val_size is relative to trainval
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        random_state=random_state,
        stratify=y_trainval if stratify else None,
    )

    # Return as DataFrames to store as ParquetDataset easily
    return (
        X_train.to_frame(name="text"),
        X_val.to_frame(name="text"),
        X_test.to_frame(name="text"),
        y_train.to_frame(name="label"),
        y_val.to_frame(name="label"),
        y_test.to_frame(name="label"),
    )


def featurize_tfidf(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    params: Dict[str, Any],
):
    """
    Fit TF-IDF on train, transform val/test.
    Outputs are sparse matrices kept in memory + the fitted vectorizer.
    """
    max_features = int(params.get("max_features", 20000))
    ngram_range = tuple(params.get("ngram_range", [1, 2]))
    min_df = int(params.get("min_df", 2))

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        lowercase=True,
        strip_accents=None,
    )

    X_train_text = X_train["text"].astype(str).tolist()
    X_val_text = X_val["text"].astype(str).tolist()
    X_test_text = X_test["text"].astype(str).tolist()

    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_val_vec = vectorizer.transform(X_val_text)
    X_test_vec = vectorizer.transform(X_test_text)

    return X_train_vec, X_val_vec, X_test_vec, vectorizer


def train_baseline_model(
    X_train_vec,
    y_train: pd.DataFrame,
    params: Dict[str, Any],
):
    """
    Train a baseline classifier (Logistic Regression).
    """
    C = float(params.get("C", 1.0))
    max_iter = int(params.get("max_iter", 200))
    class_weight = params.get("class_weight", "balanced")

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        solver="liblinear",
    )

    y = y_train["label"].astype(int).to_numpy()
    model.fit(X_train_vec, y)
    return model


def evaluate_classifier(
    model,
    X_test_vec,
    y_test: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute metrics and return a dict for JSONDataset.
    """
    y_true = y_test["label"].astype(int).to_numpy()
    y_pred = model.predict(X_test_vec)

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred).tolist()  # [[tn, fp],[fn,tp]]

    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "positive_label": "spam(1)",
        "negative_label": "ham(0)",
    }

