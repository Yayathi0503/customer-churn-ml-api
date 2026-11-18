"""A reproducible teaching model for synthetic subscription churn."""
from functools import lru_cache
import json

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    tenure_months: int = Field(ge=1, le=72)
    monthly_fee: float = Field(ge=20, le=150)
    support_calls: int = Field(ge=0, le=10)


@lru_cache(maxsize=1)
def experiment():
    rng = np.random.default_rng(42)
    tenure = rng.integers(1, 73, 2400)
    fee = rng.uniform(20, 150, 2400)
    calls = rng.binomial(10, 0.2, 2400)
    features = np.column_stack([tenure, fee, calls])
    log_odds = -1.0 - 0.055 * tenure + 0.015 * fee + 0.45 * calls
    labels = rng.binomial(1, 1 / (1 + np.exp(-log_odds)))
    train, test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, stratify=labels, random_state=42
    )
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(train, y_train)
    scores = model.predict_proba(test)[:, 1]
    return model, {"training_rows": len(train), "test_rows": len(test),
                   "roc_auc": float(roc_auc_score(y_test, scores)),
                   "constant_score_baseline_auc": 0.5}


app = FastAPI(title="Synthetic Churn Scoring API")


@app.post("/predict")
def predict(customer: Customer):
    row = [[customer.tenure_months, customer.monthly_fee, customer.support_calls]]
    model, _ = experiment()
    return {"churn_probability": float(model.predict_proba(row)[0, 1])}


if __name__ == "__main__":
    print(json.dumps(experiment()[1], indent=2))
