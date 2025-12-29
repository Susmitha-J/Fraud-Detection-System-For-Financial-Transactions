import numpy as np
import pandas as pd

def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, safe feature engineering:
    - keep original columns
    - add a couple of generic signals
    """
    out = df.copy()

    # Transaction amount log transform (common fraud signal)
    if "TransactionAmt" in out.columns:
        out["TransactionAmt_log"] = np.log1p(out["TransactionAmt"])

    # Missing value count per row (surprisingly useful)
    out["missing_count"] = out.isna().sum(axis=1)

    return out

def split_time_based(df: pd.DataFrame, time_col: str = "TransactionDT", valid_frac: float = 0.2):
    """
    Sort by time and split the last valid_frac as validation.
    """
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    cut = int(n * (1 - valid_frac))
    train_df = df_sorted.iloc[:cut].copy()
    valid_df = df_sorted.iloc[cut:].copy()
    return train_df, valid_df
