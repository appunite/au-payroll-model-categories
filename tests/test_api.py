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
    reason="Requires trained category model",
)
def test_predict_category():
    """Test category prediction endpoint."""
    payload = {
        "entity_id": "00000000-0000-0000-0000-000000000001",
        "owner_id": "00000000-0000-0000-0000-000000000002",
        "net_price": 2500.0,
        "gross_price": 3075.0,
        "currency": "PLN",
        "invoice_title": "office rent",
        "tin": "1234567890",
        "issue_date": "2024-08-29",
    }

    response = client.post("/predict/category", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "probabilities" in data
    assert "top_category" in data
    assert "top_probability" in data
    assert "model_version" in data

    # Probabilities should sum to ~1.0
    prob_sum = sum(data["probabilities"].values())
    assert 0.99 <= prob_sum <= 1.01


@pytest.mark.skipif(
    True,  # Skip if model not available
    reason="Requires trained tag model",
)
def test_predict_tag():
    """Test tag prediction endpoint."""
    payload = {
        "entity_id": "00000000-0000-0000-0000-000000000001",
        "owner_id": "00000000-0000-0000-0000-000000000002",
        "net_price": 2500.0,
        "gross_price": 3075.0,
        "currency": "PLN",
        "invoice_title": "office rent",
        "tin": "1234567890",
        "issue_date": "2024-08-29",
    }

    response = client.post("/predict/tag", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "probabilities" in data
    assert "top_tag" in data
    assert "top_probability" in data
    assert "model_version" in data

    # Probabilities should sum to ~1.0
    prob_sum = sum(data["probabilities"].values())
    assert 0.99 <= prob_sum <= 1.01


def test_predict_category_invalid_data():
    """Test category prediction with invalid data."""
    payload = {
        "entity_id": "test",
        "owner_id": "test",
        "net_price": -100,  # Invalid: negative price
        "gross_price": 100,
        "currency": "PLN",
        "invoice_title": "test",
        "issue_date": "2024-08-29",
    }

    response = client.post("/predict/category", json=payload)
    # Should fail validation
    assert response.status_code == 422


def test_predict_tag_invalid_data():
    """Test tag prediction with invalid data."""
    payload = {
        "entity_id": "test",
        "owner_id": "test",
        "net_price": -100,  # Invalid: negative price
        "gross_price": 100,
        "currency": "PLN",
        "invoice_title": "test",
        "issue_date": "2024-08-29",
    }

    response = client.post("/predict/tag", json=payload)
    # Should fail validation
    assert response.status_code == 422
