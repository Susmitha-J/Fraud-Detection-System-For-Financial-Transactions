import json
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from src.data import load_ieee_cis
from src.features import basic_feature_engineering, split_time_based

DATA_DIR = "data"
MODEL_DIR = "models"

def prep_xy(df: pd.DataFrame, target_col: str = "isFraud"):
    # Drop obvious identifiers and target
    drop_cols = [c for c in ["isFraud", "TransactionID"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col].astype(int) if target_col in df.columns else None

    # Convert categoricals to category dtype for LightGBM
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    return X, y, cat_cols

def main():
    df = load_ieee_cis(DATA_DIR, train=True)
    df = basic_feature_engineering(df)

    train_df, valid_df = split_time_based(df, time_col="TransactionDT", valid_frac=0.2)

    X_train, y_train, cat_cols = prep_xy(train_df)
    X_valid, y_valid, _ = prep_xy(valid_df)

    # Class imbalance handling: scale_pos_weight = negatives/positives
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="average_precision",
        categorical_feature=cat_cols
    )

    # Evaluate
    valid_pred = model.predict_proba(X_valid)[:, 1]
    ap = average_precision_score(y_valid, valid_pred)
    print(f"Validation PR-AUC (Average Precision): {ap:.5f}")

    # Flag top 0.5% as fraud
    alert_rate = 0.005
    threshold = np.quantile(valid_pred, 1 - alert_rate)

    print(f"Operational threshold (top {alert_rate*100:.2f}%): {threshold:.5f}")
    # Save model + feature schema
    joblib.dump(model, f"{MODEL_DIR}/lgbm_fraud.joblib")
    meta = {
        "categorical_columns": cat_cols,
        "feature_columns": list(X_train.columns),
        "threshold": threshold
    }
    with open(f"{MODEL_DIR}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved model to models/lgbm_fraud.joblib and models/meta.json")

if __name__ == "__main__":
    main()
