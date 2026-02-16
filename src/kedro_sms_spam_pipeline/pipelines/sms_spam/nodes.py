import pandas as pd

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
