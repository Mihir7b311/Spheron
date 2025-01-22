import pytest
import torch
from src.batch_processor.batch_manager import BatchManager
from src.cuda.stream_manager import CUDAStreamManager

@pytest.fixture
async def setup_batch_manager():
    stream_manager = CUDAStreamManager()
    return BatchManager(stream_manager)

@pytest.mark.asyncio
async def test_batch_processing(setup_batch_manager):
    manager = setup_batch_manager
    
    # Add requests
    for i in range(5):
        await manager.add_request({
            "input": torch.tensor([float(i)])
        })
    
    results = await manager.process_batch(gpu_id=0)
    assert len(results) == 5
    assert all(r['output'].device.type == ('cuda' if torch.cuda.is_available() else 'cpu') for r in results)
