import pytest
from resource_manager.kubernetes_controller.controller import KubernetesController

@pytest.mark.asyncio
async def test_pod_creation(test_config, mocker):
    controller = KubernetesController(test_config["kubernetes"])
    
    mock_v1 = mocker.Mock()
    mock_metadata = mocker.Mock()
    mock_metadata.name = "test-pod"
    mock_v1.create_namespaced_pod.return_value = mocker.Mock(metadata=mock_metadata)
    controller.v1 = mock_v1
    
    pod_spec = {
        "function_id": "test-func",
        "image": "test:latest"
    }
    
    gpu_slice = {
        "gpu_id": "gpu-0",
        "memory": "2Gi",
        "gpu_fraction": 1,
        "compute_percentage": 100,
        "slice_id": "slice-0"
    }
    
    result = await controller.create_pod(pod_spec, gpu_slice)
    assert result.metadata.name == "test-pod"
