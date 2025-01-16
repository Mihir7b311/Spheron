# tests/test_k8s_controller.py
import pytest
from kubernetes_controller.controller import KubernetesController

@pytest.mark.asyncio
async def test_pod_creation(mocker):
    config = {
        "namespace": "gpu-faas",
        "service_account": "gpu-service",
        "pod_limits": {
            "cpu": "1",
            "memory": "2Gi"
        }
    }
    
    controller = KubernetesController(config)
    
    # Create a proper mock pod
    mock_metadata = mocker.Mock()
    mock_metadata.name = "test-pod"  # Set the name directly
    
    mock_pod = mocker.Mock()
    mock_pod.metadata = mock_metadata
    
    # Mock kubernetes client
    mock_v1 = mocker.Mock()
    mock_v1.create_namespaced_pod.return_value = mock_pod
    controller.v1 = mock_v1
    
    # Test pod creation
    gpu_slice = {
        "slice_id": "test-slice",
        "gpu_id": "0",
        "compute_percentage": 50,
        "gpu_fraction": "0.5"
    }
    
    result = await controller.create_pod({
        "function_id": "test-func",
        "image": "test:latest"
    }, gpu_slice)
    
    # Verify the mock was called
    mock_v1.create_namespaced_pod.assert_called_once()
    
    # Verify the result
    assert result.metadata.name == "test-pod"