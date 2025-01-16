# tests/test_integration.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "GPU Resource Manager"}

def test_allocation():
    request_data = {
        "function_id": "test-func",
        "memory": "2Gi",
        "compute_percentage": 50
    }
    response = client.post("/allocate", json=request_data)
    assert response.status_code == 200
    assert response.json()["status"] == "success"