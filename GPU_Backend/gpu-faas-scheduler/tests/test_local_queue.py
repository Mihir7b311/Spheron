 
# tests/test_local_queue.py
import pytest
from local_queue.gpu_queue import LocalQueueManager

@pytest.mark.asyncio
async def test_local_queue_management():
    config = {"per_gpu_size": 3, "timeout_seconds": 300}
    local_queue = LocalQueueManager(config)
    
    # Test adding to GPU queue
    gpu_id = "gpu-1"
    request = {"function_id": "test_func", "model_id": "model1"}
    await local_queue.add_to_gpu_queue(gpu_id, request)
    
    # Verify queue length
    length = await local_queue.get_gpu_queue_length(gpu_id)
    assert length == 1

@pytest.mark.asyncio
async def test_local_queue_overflow():
    config = {"per_gpu_size": 2, "timeout_seconds": 300}
    local_queue = LocalQueueManager(config)
    gpu_id = "gpu-1"
    
    # Fill queue to capacity
    await local_queue.add_to_gpu_queue(gpu_id, {"id": "1"})
    await local_queue.add_to_gpu_queue(gpu_id, {"id": "2"})
    
    # Verify overflow raises exception
    with pytest.raises(Exception) as exc_info:
        await local_queue.add_to_gpu_queue(gpu_id, {"id": "3"})
    assert "Queue is full" in str(exc_info.value)