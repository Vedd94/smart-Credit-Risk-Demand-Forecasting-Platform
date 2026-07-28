import requests
import pytest
import requests

API_URL = "http://backend:8000/predict"


def test_api_call(requests_mock):

    requests_mock.post(
        API_URL,
        json={"predicted_category": "Good"},
        status_code=200,
    )

    # 19,female,3,own,little,moderate,3964,22,car,good
    payload = {
        "age": 19,
        "sex": "female",
        "job": 3,
        "housing": "own",
        "saving_accounts": "little",
        "checking_account": "moderate",
        "credit_amount": 3964,
        "duration": 22,
        "purpose": "car",
    }

    response = requests.post(API_URL, json=payload)

    assert response.status_code == 200
    assert response.json()["predicted_category"] == "Good"

def test_connection_error(monkeypatch):

    def mock_post(*args, **kwargs):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(requests, "post", mock_post)

    with pytest.raises(requests.exceptions.ConnectionError):
        requests.post("http://backend:8000/predict")


def test_server_error(requests_mock):

    requests_mock.post(
        API_URL,
        json={"detail": "Internal Server Error"},
        status_code=500,
    )

    response = requests.post(API_URL)

    assert response.status_code == 500


def test_response_contains_prediction(requests_mock):

    requests_mock.post(
        API_URL,
        json={"predicted_category": "Good"},
        status_code=200,
    )

    response = requests.post(API_URL)

    data = response.json()

    assert "predicted_category" in data

def test_payload():

    payload = {
        "age": 45,
        "sex": "female",
        "job": 1,
        "housing": "rent",
        "saving_accounts": "rich",
        "checking_account": "little",
        "credit_amount": 6000,
        "duration": 18,
        "purpose": "business",
    }

    expected_keys = {
        "age",
        "sex",
        "job",
        "housing",
        "saving_accounts",
        "checking_account",
        "credit_amount",
        "duration",
        "purpose",
    }

    assert set(payload.keys()) == expected_keys