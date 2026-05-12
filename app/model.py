import joblib
import numpy as np
import os
import pandas as pd

# Paths to saved model files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.pkl")

# Load model, scaler and threshold at startup
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
threshold = joblib.load(THRESHOLD_PATH)

def predict_fraud(transaction: dict) -> dict:
    # Extract Amount and scale it
    amount = transaction["Amount"]
    amount_scaled = scaler.transform(pd.DataFrame([[amount]], columns=["Amount"]))[0][0]

    # Build feature array in correct order
    features = [
        transaction["V1"], transaction["V2"], transaction["V3"],
        transaction["V4"], transaction["V5"], transaction["V6"],
        transaction["V7"], transaction["V8"], transaction["V9"],
        transaction["V10"], transaction["V11"], transaction["V12"],
        transaction["V13"], transaction["V14"], transaction["V15"],
        transaction["V16"], transaction["V17"], transaction["V18"],
        transaction["V19"], transaction["V20"], transaction["V21"],
        transaction["V22"], transaction["V23"], transaction["V24"],
        transaction["V25"], transaction["V26"], transaction["V27"],
        transaction["V28"], amount_scaled
    ]

    # Get fraud probability
    input_array = np.array(features).reshape(1, -1)
    fraud_prob = model.predict_proba(input_array)[0][1]

    # Apply threshold
    is_fraud = bool(fraud_prob >= threshold)

    # Assign risk level
    if fraud_prob < 0.3:
        risk_level = "LOW"
    elif fraud_prob < 0.7:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    # Build message
    if is_fraud:
        message = f"Transaction flagged as FRAUDULENT with {round(float(fraud_prob) * 100, 2)}% probability."
    else:
        message = f"Transaction appears LEGITIMATE with {round(float(fraud_prob) * 100, 2)}% fraud probability."

    return {
        "is_fraud": is_fraud,
        "fraud_probability": round(fraud_prob, 4),
        "risk_level": risk_level,
        "message": message
    }