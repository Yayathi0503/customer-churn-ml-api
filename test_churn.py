from fastapi.testclient import TestClient
import pytest
from churn import app, experiment

client = TestClient(app)
sample = {"tenure_months": 24, "monthly_fee": 80, "support_calls": 2}


def test_probability_moves_with_support_calls():
    scores = []
    for calls in (0, 8):
        response = client.post("/predict", json={**sample, "support_calls": calls})
        assert response.status_code == 200
        scores.append(response.json()["churn_probability"])
    assert 0 < scores[0] < scores[1] < 1


@pytest.mark.parametrize("field,value", [("monthly_fee", -1), ("support_calls", 11),
                                         ("tenure_months", 0), ("unknown", 1)])
def test_rejects_invalid_customers(field, value):
    assert client.post("/predict", json={**sample, field: value}).status_code == 422


def test_holdout_beats_constant_baseline():
    _, report = experiment()
    assert report["test_rows"] == 600
    assert report["roc_auc"] > report["constant_score_baseline_auc"]
