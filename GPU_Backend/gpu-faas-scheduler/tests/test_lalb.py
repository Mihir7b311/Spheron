# tests/test_lalb.py
import pytest
from lalb.scheduler import LALBScheduler
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_lalb_cache_hit_idle_gpu(mocker):
    # Mock dependencies
    global_queue = mocker.Mock()
    local_queue = mocker.Mock()
    local_queue.add_to_gpu_queue = AsyncMock()  # Use AsyncMock for async methods
    time_slot = mocker.Mock()
    
    config = {
        "cache_weight": 0.7,
        "load_weight": 0.3,
        "max_retry": 3
    }
    
    scheduler = LALBScheduler(config, global_queue, local_queue, time_slot)
    
    # Initialize GPU states
    scheduler.gpu_states = {
        "gpu-1": {"is_busy": False, "cached_models": {"model1"}},
        "gpu-2": {"is_busy": True, "cached_models": set()}
    }
    
    request = {
        "function_id": "test_func",
        "model_id": "model1"
    }
    
    result = await scheduler.schedule_request(request)
    assert result["gpu_id"] == "gpu-1"
    assert result["status"] == "scheduled"