import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(title="Intelligent Fraud Detection API")

model = joblib.load("models/lgbm_fraud.joblib")
meta = json.load(open("models/meta.json", "r", encoding="utf-8"))
FEATURES = meta["feature_columns"]
CAT_COLS = set(meta["categorical_columns"])
THRESHOLD = float(meta.get("threshold", 0.5))

class TransactionPayload(BaseModel):
    # Send arbitrary key-value pairs (must match feature names)
    features: Dict[str, Any]

@app.get("/")
def root():
    return {"status": "ok", "message": "Fraud API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score(payload: TransactionPayload):
    # Build exactly the same feature set and order as training
    full = {f: payload.features.get(f, None) for f in FEATURES}
    x = pd.DataFrame([full])

    # Force categorical columns to 'category' dtype (even if None)
    for c in CAT_COLS:
        if c in x.columns:
            x[c] = x[c].astype("category")

    # Optional: ensure numeric columns are numeric when possible
    # (LightGBM can handle NaN, but not random strings in numeric cols)
    for col in x.columns:
        if col not in CAT_COLS:
            x[col] = pd.to_numeric(x[col], errors="coerce")

    proba = float(model.predict_proba(x)[:, 1][0])
    decision = int(proba >= THRESHOLD)
    return {"fraud_probability": proba, "decision": decision, "threshold": THRESHOLD}