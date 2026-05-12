# Fraud Detection API

A production-ready REST API for real-time credit card fraud detection using XGBoost, FastAPI, and Docker. Trained on 284,807 real transactions with SMOTE-based class balancing and optimised decision threshold — achieving **ROC-AUC of 0.9739** and **F1-Score of 0.81**.

---

## The Problem

Credit card fraud detection is a severely imbalanced classification problem. In this dataset, only **0.17% of transactions are fraudulent** — meaning a naive model that predicts "not fraud" every time achieves 99.83% accuracy while catching zero fraudsters. This project addresses that challenge end-to-end: from imbalanced data handling to threshold optimisation to production deployment.

---

## Results

| Metric | Logistic Regression (Baseline) | XGBoost (Final) |
|---|---|---|
| Fraud Precision | 0.06 | **0.81** |
| Fraud Recall | 0.92 | **0.81** |
| Fraud F1-Score | 0.11 | **0.81** |
| ROC-AUC | 0.9459 | **0.9739** |

> Optimal decision threshold tuned to **0.95** via precision-recall curve analysis — improving F1 from 0.48 (default 0.5 threshold) to 0.81.

---

## Tech Stack

- **Model:** XGBoost Classifier
- **Class Balancing:** SMOTE (imbalanced-learn)
- **API:** FastAPI + Uvicorn
- **Validation:** Pydantic v2
- **Containerisation:** Docker
- **Testing:** Pytest (8/8 passing)
- **Serialisation:** Joblib

---

## Project Structure

```
fraud-detection-api/
├── app/
│   ├── main.py          # FastAPI app — 3 endpoints
│   ├── model.py         # Model loading + prediction logic
│   └── schemas.py       # Pydantic input/output schemas
├── models/
│   ├── fraud_model.pkl  # Trained XGBoost model
│   ├── scaler.pkl       # StandardScaler for Amount feature
│   └── threshold.pkl    # Optimised decision threshold (0.95)
├── notebooks/
│   └── fraud_detection.ipynb  # EDA, SMOTE, training, evaluation
├── tests/
│   └── test_main.py     # 8 pytest tests
├── Dockerfile
├── requirements.txt
└── conftest.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root — confirms API is running |
| GET | `/health` | Health check — model info + metrics |
| POST | `/predict` | Predict fraud from transaction features |

### Sample Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363,
    "V4": 1.3782, "V5": -0.3383, "V6": 0.4624,
    "V7": 0.2396, "V8": 0.0987, "V9": 0.3638,
    "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
    "V13": -0.9914, "V14": -0.3112, "V15": 1.4682,
    "V16": -0.4704, "V17": 0.2080, "V18": 0.0258,
    "V19": 0.4040, "V20": 0.2514, "V21": -0.0183,
    "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
    "V25": 0.1285, "V26": -0.1891, "V27": 0.1336,
    "V28": -0.0211, "Amount": 149.62
  }'
```

### Sample Response

```json
{
  "is_fraud": false,
  "fraud_probability": 0.035,
  "risk_level": "LOW",
  "message": "Transaction appears LEGITIMATE with 3.5% fraud probability."
}
```

---

## Run Locally

### Option 1 — Docker (recommended)

```bash
# Build the image
docker build -t fraud-detection-api .

# Run the container
docker run -p 8000:8000 fraud-detection-api
```

### Option 2 — Python virtual environment

```bash
# Clone the repo
git clone https://github.com/Retr0Rat/fraud-detection-api.git
cd fraud-detection-api

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload
```

Then open **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

---

## Run Tests

```bash
python -m pytest tests/ -v
```

```
tests/test_main.py::test_root_endpoint PASSED
tests/test_main.py::test_health_endpoint PASSED
tests/test_main.py::test_predict_legitimate PASSED
tests/test_main.py::test_predict_suspicious PASSED
tests/test_main.py::test_predict_missing_field PASSED
tests/test_main.py::test_predict_invalid_type PASSED
tests/test_main.py::test_predict_response_structure PASSED
tests/test_main.py::test_fraud_probability_range PASSED

8 passed in 2.85s
```

---

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions from European cardholders
- 492 fraudulent transactions (0.17%)
- 28 PCA-anonymised features (V1–V28) + Amount + Time
- `creditcard.csv` is excluded from this repo due to file size — download directly from Kaggle and place in `data/`

---

## Key Design Decisions

**Why SMOTE over class weighting?**
SMOTE generates synthetic minority samples, helping the model learn the fraud decision boundary rather than just penalising misclassifications. More effective for extreme imbalances like 0.17%.

**Why threshold 0.95 over default 0.5?**
At 0.5, the model flags too many legitimate transactions (precision: 0.33). Tuning to 0.95 raised precision to 0.81 while maintaining recall at 0.81 — a balanced, production-viable tradeoff.

**Why FastAPI over Flask?**
Automatic Swagger UI, native Pydantic validation, async support, and significantly better performance for ML inference workloads.

---

## Author

**Aman** — AI & Cybersecurity Post-Graduate, Durham College  
[github.com/Retr0Rat](https://github.com/Retr0Rat) · aman23092003@gmail.com · Oshawa, ON
