# tests/unit/test_gpu_manager.py
import pytest
from resource_manager.gpu_slice_manager.manager import GPUSliceManager

@pytest.mark.asyncio
async def test_gpu_slice_allocation(test_config):
    manager = GPUSliceManager(test_config["gpu_slice"])
    
    request = {
        "gpu_id": "gpu-0",
        "memory": "2Gi",
        "compute_percentage": 50
    }
    
    result = await manager.allocate_slice(request)
    assert result["gpu_id"] == "gpu-0"

@pytest.mark.asyncio
async def test_memory_parsing(test_config):
    manager = GPUSliceManager({
        "min_memory": "1Gi",
        "max_memory": "32Gi",
        "default_compute": 20
    })
    assert manager._parse_memory("2Gi") == 2 * 1024 * 1024 * 1024