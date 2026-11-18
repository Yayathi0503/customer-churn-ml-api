# Customer churn scoring API

Score a fictional subscription customer using tenure, monthly fee, and recent support calls. FastAPI validates the request and returns the logistic regression model's estimated churn probability.

## Run the demo

Use Python 3.12. From the repository root, create a virtual environment with `python -m venv .venv`. Activate it (`.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux), then run:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
```

```sh
python churn.py
python -m uvicorn churn:app
```

Open http://127.0.0.1:8000/docs, choose **POST /predict**, and try `{"tenure_months":24,"monthly_fee":80,"support_calls":2}`. Change one field and compare the predictions.

## How the evaluation works

The seeded generator creates 2,400 example subscriptions. It draws churn labels from a known probability model. A stratified split reserves 600 rows for evaluation. Scaling and logistic regression fit only on the training rows. Running the script prints holdout ROC-AUC and a constant-score baseline. These figures describe the synthetic experiment. Real customer data would require separate training and validation.
