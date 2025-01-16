 
# tests/test_global_queue.py
import pytest
from global_queue.queue_manager import GlobalQueueManager

@pytest.mark.asyncio
async def test_global_queue_add_request():
    config = {"max_size": 5, "starvation_threshold": 25}
    queue_manager = GlobalQueueManager(config)
    
    # Test adding request
    request = {"function_id": "test_func", "model_id": "model1"}
    await queue_manager.add_request(request)
    
    # Verify request was added
    next_request = await queue_manager.get_next_request()
    assert next_request["function_id"] == "test_func"
    assert next_request["skip_count"] == 0

@pytest.mark.asyncio
async def test_global_queue_overflow():
    config = {"max_size": 2, "starvation_threshold": 25}
    queue_manager = GlobalQueueManager(config)
    
    # Add requests up to max size
    await queue_manager.add_request({"id": "1"})
    await queue_manager.add_request({"id": "2"})
    
    # Verify overflow raises exception
    with pytest.raises(Exception) as exc_info:
        await queue_manager.add_request({"id": "3"})
    assert "Queue is full" in str(exc_info.value)