import joblib
import json
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import load_ieee_cis
from src.features import basic_feature_engineering, split_time_based

def main():
    df = load_ieee_cis("data", train=True)
    df = basic_feature_engineering(df)
    train_df, valid_df = split_time_based(df, time_col="TransactionDT", valid_frac=0.2)

    # Load model + schema
    model = joblib.load("models/lgbm_fraud.joblib")
    meta = json.load(open("models/meta.json", "r", encoding="utf-8"))
    feat_cols = meta["feature_columns"]
    cat_cols = set(meta["categorical_columns"])

    # Prep validation features
    drop_cols = [c for c in ["isFraud", "TransactionID"] if c in valid_df.columns]
    X_valid = valid_df.drop(columns=drop_cols)
    y_valid = valid_df["isFraud"].astype(int)

    # Enforce column order and types
    X_valid = X_valid[feat_cols]
    for c in X_valid.columns:
        if c in cat_cols:
            X_valid[c] = X_valid[c].astype("category")

    pred = model.predict_proba(X_valid)[:, 1]
    
    print("PR-AUC:", average_precision_score(y_valid, pred))
    print("ROC-AUC:", roc_auc_score(y_valid, pred))

    alert_rate = 0.005  # top 0.5% flagged
    threshold = np.quantile(pred, 1 - alert_rate)

    print(f"Operational threshold (top {alert_rate*100:.2f}%): {threshold:.5f}")

if __name__ == "__main__":
    main()
