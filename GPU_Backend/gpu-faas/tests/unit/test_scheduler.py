import pytest
from scheduler.lalb.scheduler import LALBScheduler

@pytest.fixture
def scheduler(mocker):
    config = {
        "cache_weight": 0.7,
        "load_weight": 0.3,
        "max_retry": 5
    }
    
    # Mock the queue and time_slot objects
    global_queue = mocker.Mock()
    local_queue = mocker.Mock()
    
    # Mock the async method `add_to_gpu_queue` on `local_queue`
    local_queue.add_to_gpu_queue = mocker.AsyncMock(return_value=None)  # or set a suitable return value
    
    time_slot = mocker.Mock()
    
    return LALBScheduler(config, global_queue, local_queue, time_slot)
