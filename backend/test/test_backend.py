import sys
from pathlib import Path

# Ensure the project root is on sys.path so `main` is importable
# when pytest is run from the repo root in CI (working-directory: .)
sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Sample valid request
valid_payload = {
    "age": 35,
    "sex": "male",
    "job": 2,
    "housing": "own",
    "saving_accounts": "little",
    "checking_account": "moderate",
    "credit_amount": 5000,
    "duration": 24,
    "purpose": "car"
}


def test_health():
    """Test health endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_valid_input():
    """Prediction should work with valid input."""
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200

    data = response.json()

    assert "predicted_category" in data
    assert data["predicted_category"] is not None


def test_missing_required_field():
    """API should reject requests with missing fields."""
    payload = valid_payload.copy()
    payload.pop("age")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_invalid_age():
    """Age cannot be negative."""
    payload = valid_payload.copy()
    payload["age"] = -5

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_invalid_sex():
    """Only male/female are allowed."""
    payload = valid_payload.copy()
    payload["sex"] = "robot"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_invalid_housing():
    """Housing must be one of own/rent/free."""
    payload = valid_payload.copy()
    payload["housing"] = "hostel"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_invalid_credit_amount_type():
    """Credit amount must be an integer."""
    payload = valid_payload.copy()
    payload["credit_amount"] = "five thousand"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_invalid_purpose():
    """Purpose must match one of the allowed categories."""
    payload = valid_payload.copy()
    payload["purpose"] = "shopping"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_multiple_predictions():
    """API should handle multiple consecutive requests."""
    for _ in range(10):
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200