from fastapi import FastAPI, HTTPException
from app.schemas import TransactionInput, PredictionOutput
from app.model import predict_fraud

app = FastAPI(
    title="Fraud Detection API",
    description="XGBoost-powered credit card fraud detection API with SMOTE balancing and optimised threshold. ROC-AUC: 0.9739",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "XGBoost",
        "threshold": 0.95,
        "roc_auc": 0.9739
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput):
    try:
        result = predict_fraud(transaction.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))