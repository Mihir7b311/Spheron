# tests/test_gpu_manager.py
import pytest
from gpu_slice_manager.manager import GPUSliceManager

@pytest.mark.asyncio
async def test_gpu_allocation():
    # Initialize manager with test config
    config = {
        "min_memory": "1Gi",
        "max_memory": "32Gi",
        "default_compute": 20,
        "oversubscription_limit": 1.2
    }
    
    manager = GPUSliceManager(config)
    
    # Test basic allocation
    request = {
        "memory": "2Gi",
        "compute_percentage": 30
    }
    
    slice_info = await manager.allocate_slice(request)
    assert slice_info is not None
    assert "slice_id" in slice_info
    assert "gpu_id" in slice_info

    # Test memory conversion
    assert slice_info["memory"] == 2 * 1024 * 1024 * 1024  # 2Gi in bytes

    # Test compute percentage
    assert slice_info["compute_percentage"] == 30