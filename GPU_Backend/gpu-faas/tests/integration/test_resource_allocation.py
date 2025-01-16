import pytest

@pytest.mark.asyncio
async def test_resource_allocation_flow(test_config, mocker):
    gpu_manager = mocker.AsyncMock()
    gpu_manager.allocate_slice.return_value = {
        "gpu_id": "gpu-0",
        "memory": "2Gi",
        "compute_percentage": 50
    }

    request = {
        "gpu_id": "gpu-0",
        "memory": "2Gi",
        "compute_percentage": 50
    }

    result = await gpu_manager.allocate_slice(request)
    assert result["gpu_id"] == "gpu-0"
    assert result["memory"] == "2Gi"

@pytest.mark.asyncio
async def test_resource_allocation_failure(test_config, mocker):
    gpu_manager = mocker.AsyncMock()
    gpu_manager.allocate_slice.side_effect = Exception("Out of memory")

    request = {
        "gpu_id": "gpu-0",
        "memory": "64Gi",  # Too much memory
        "compute_percentage": 50
    }

    with pytest.raises(Exception) as exc_info:
        await gpu_manager.allocate_slice(request)
    assert "Out of memory" in str(exc_info.value)