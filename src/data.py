import pandas as pd

def load_ieee_cis(data_dir: str, train: bool = True) -> pd.DataFrame:
    """
    Loads IEEE-CIS train or test and merges transaction + identity on TransactionID.
    """
    if train:
        tx = pd.read_csv(f"{data_dir}/train_transaction.csv")
        idn = pd.read_csv(f"{data_dir}/train_identity.csv")
        df = tx.merge(idn, on="TransactionID", how="left")
        return df
    else:
        tx = pd.read_csv(f"{data_dir}/test_transaction.csv")
        idn = pd.read_csv(f"{data_dir}/test_identity.csv")
        df = tx.merge(idn, on="TransactionID", how="left")
        return df
