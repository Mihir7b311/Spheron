# tests/test_integration.py
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_schedule_function_integration(client, mocker):
    # Mock scheduler methods
    mocker.patch('lalb.scheduler.LALBScheduler.get_idle_gpus', 
                 return_value=["gpu-1"])
    mocker.patch('lalb.scheduler.LALBScheduler.is_model_cached',
                 return_value=True)
    
    request_data = {
        "function_id": "test_function",
        "model_id": "model1",
        "input_data": "test_input"
    }
    
    response = client.post("/schedule", json=request_data)
    assert response.status_code == 200
    result = response.json()
    assert "gpu_id" in result
    assert result["status"] == "scheduled"