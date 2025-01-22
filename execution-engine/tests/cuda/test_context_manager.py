# tests/cuda/test_context_manager.py

import pytest
import torch
from src.cuda.context_manager import CUDAContextManager

@pytest.fixture
async def setup_context_manager():
    return CUDAContextManager()

@pytest.mark.asyncio
async def test_context_creation(setup_context_manager):
    manager = setup_context_manager
    context = await manager.create_context(gpu_id=0)
    
    assert "gpu_id" in context
    assert "stream" in context
    assert "device" in context