# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def test_config():
    return {
        "kubernetes": {
            "namespace": "gpu-faas",
            "service_account": "gpu-service",
            "pod_limits": {
                "cpu": "1",
                "memory": "2Gi"
            }
        },
        "mps": {
            "max_processes": 48,
            "compute_percentage": 10,
            "default_memory": "1Gi",
            "max_memory": "16Gi"
        },
        "gpu_slice": {
            "min_memory": "1Gi",
            "max_memory": "32Gi",
            "default_compute": 20,
            "oversubscription_limit": 1.2
        }
    }