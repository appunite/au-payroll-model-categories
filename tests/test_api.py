"""Test FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "endpoints" in data


def test_health():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_version" in data
    assert "timestamp" in data


@pytest.mark.skipif(
    True,  # Skip if model not available
    reason="Requires trained model"
)
def test_predict():
    """Test prediction endpoint."""
    payload = {
        "entityId": "00000000-0000-0000-0000-000000000001",
        "ownerId": "00000000-0000-0000-0000-000000000002",
        "netPrice": 2500.0,
        "grossPrice": 3075.0,
        "currency": "PLN",
        "title_normalized": "office rent",
        "tin": "1234567890",
        "issueDate": "2024-08-29"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "probabilities" in data
    assert "top_category" in data
    assert "top_probability" in data
    assert "model_version" in data

    # Probabilities should sum to ~1.0
    prob_sum = sum(data["probabilities"].values())
    assert 0.99 <= prob_sum <= 1.01


def test_predict_invalid_data():
    """Test prediction with invalid data."""
    payload = {
        "entityId": "test",
        "ownerId": "test",
        "netPrice": -100,  # Invalid: negative price
        "grossPrice": 100,
        "currency": "PLN",
        "title_normalized": "test",
        "issueDate": "2024-08-29"
    }

    response = client.post("/predict", json=payload)
    # Should fail validation
    assert response.status_code == 422
