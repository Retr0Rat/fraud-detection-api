from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Sample legitimate transaction
LEGITIMATE_TRANSACTION = {
    "V1": -1.3598071336738, "V2": -0.0727811733098497,
    "V3": 2.53634673796914, "V4": 1.37815522427443,
    "V5": -0.338320769942518, "V6": 0.462387777762292,
    "V7": 0.239598554061257, "V8": 0.0986979012610507,
    "V9": 0.363786969611213, "V10": 0.0907941719789316,
    "V11": -0.551599533260813, "V12": -0.617800855762348,
    "V13": -0.991389847235408, "V14": -0.311169353699879,
    "V15": 1.46817697209427, "V16": -0.470400525259478,
    "V17": 0.207971241929242, "V18": 0.0257905801985591,
    "V19": 0.403992960255733, "V20": 0.251412098239705,
    "V21": -0.018306777944153, "V22": 0.277837575558899,
    "V23": -0.110473910188767, "V24": 0.0669280749146731,
    "V25": 0.128539358273528, "V26": -0.189114843888824,
    "V27": 0.133558376740387, "V28": -0.0210530534538215,
    "Amount": 149.62
}

# Sample suspicious transaction
SUSPICIOUS_TRANSACTION = {
    "V1": -2.3122265423263, "V2": 1.19219628980344,
    "V3": -1.01545657571929, "V4": 0.251705964096468,
    "V5": -1.36447782842639, "V6": -1.01919600653974,
    "V7": 0.574400900178384, "V8": -0.679658068694789,
    "V9": -0.601530800560521, "V10": -2.16189646907425,
    "V11": 1.56937761886481, "V12": -1.25322942846987,
    "V13": -0.670517108918725, "V14": -2.35616285266565,
    "V15": 1.18383987040713, "V16": -1.18256984527425,
    "V17": -1.68157268602488, "V18": 0.384583748450592,
    "V19": 0.261390979473099, "V20": 0.0129241727552994,
    "V21": 0.517232371220395, "V22": -0.0350493686884622,
    "V23": -0.465211076986977, "V24": 0.320198199559022,
    "V25": 0.0445191674731601, "V26": 0.177839798417879,
    "V27": 0.261145002567677, "V28": -0.143275874698919,
    "Amount": 0.00
}


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model"] == "XGBoost"
    assert data["roc_auc"] == 0.9739


def test_predict_legitimate():
    response = client.post("/predict", json=LEGITIMATE_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] == False
    assert data["fraud_probability"] < 0.5
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert "message" in data


def test_predict_suspicious():
    response = client.post("/predict", json=SUSPICIOUS_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert data["fraud_probability"] > 0.3
    assert data["risk_level"] in ["MEDIUM", "HIGH"]
    assert "message" in data


def test_predict_missing_field():
    incomplete = LEGITIMATE_TRANSACTION.copy()
    del incomplete["Amount"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_invalid_type():
    invalid = LEGITIMATE_TRANSACTION.copy()
    invalid["V1"] = "not_a_number"
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_response_structure():
    response = client.post("/predict", json=LEGITIMATE_TRANSACTION)
    data = response.json()
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert "risk_level" in data
    assert "message" in data


def test_fraud_probability_range():
    response = client.post("/predict", json=LEGITIMATE_TRANSACTION)
    prob = response.json()["fraud_probability"]
    assert 0.0 <= prob <= 1.0