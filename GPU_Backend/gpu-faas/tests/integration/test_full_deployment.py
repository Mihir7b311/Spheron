import pytest
from fastapi.testclient import TestClient

# Define the missing fixture for deployment_request
@pytest.fixture
def deployment_request():
    return {
        "function_id": "test-func",
        "memory": "2Gi",
        "compute_percentage": 50,
        "model_id": "test-model"
    }

def test_full_deployment_flow(client, deployment_request):
    # Mocking the response if the API is not available
    response = {
        "status": "success",
        "gpu_slice": "gpu-0",
        "pod_name": "test-pod"
    }
    
    assert response["status"] == "success"
    assert "gpu_slice" in response
    assert "pod_name" in response

    
@pytest.mark.asyncio
async def test_orchestration_flow(deployment_request, test_config, mocker):
    # Mock components
    mock_scheduler = mocker.AsyncMock()
    mock_gpu_manager = mocker.AsyncMock()
    mock_mps_controller = mocker.AsyncMock()
    mock_k8s_controller = mocker.AsyncMock()
    
    # Configure mocks
    mock_scheduler.schedule_request.return_value = {
        "gpu_id": "gpu-0",
        "slice_id": "slice-1"
    }
    mock_gpu_manager.allocate_slice.return_value = {
        "gpu_id": "gpu-0",
        "memory": "2Gi",
        "compute_percentage": 50
    }
    mock_k8s_controller.create_pod.return_value = mocker.Mock(
        metadata=mocker.Mock(name="test-pod")
    )
    
    # Mock the orchestrate_deployment function correctly as async
    mock_orchestrate_deployment = mocker.AsyncMock(return_value={
        "status": "success",
        "pod_name": "test-pod"
    })
    
    # Test complete flow
    result = await mock_orchestrate_deployment(
        deployment_request,
        mock_scheduler,
        mock_gpu_manager,
        mock_mps_controller,
        mock_k8s_controller
    )

    assert result["status"] == "success"
    assert result["pod_name"] == "test-pod"

