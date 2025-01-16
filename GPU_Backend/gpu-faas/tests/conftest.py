# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def test_config():
    return {
        "scheduler": {
            "lalb": {
                "cache_weight": 0.7,
                "load_weight": 0.3,
                "max_retry": 3
            },
            "global_queue": {
                "max_size": 1000,
                "priority_levels": 5
            },
            "local_queue": {
                "per_gpu_size": 100,
                "timeout_seconds": 300
            },
            "time_slot": {
                "min_duration": 1,
                "max_duration": 300,
                "scheduling_window": 3600
            }
        },
        "gpu_slice": {
            "min_memory": "1Gi",
            "max_memory": "32Gi",
            "default_compute": 20,
            "oversubscription_limit": 1.2
        },
        "mps": {
            "max_processes": 48,
            "compute_percentage": 10,
            "default_memory": "1Gi",
            "max_memory": "16Gi"
        },
        "kubernetes": {
            "namespace": "test-namespace",
            "service_account": "test-sa",
            "pod_limits": {
                "cpu": "1",
                "memory": "2Gi"
            }
        }
    }

@pytest.fixture
def client():
    return TestClient(app)