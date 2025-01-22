# tests/cuda/test_memory_manager.py

import pytest
import torch
from src.cuda.memory_manager import CUDAMemoryManager


class TestCUDAMemoryManager:
    @pytest.fixture
    def memory_manager(self, test_config):
        return CUDAMemoryManager(test_config["cuda"])

    @pytest.mark.asyncio
    async def test_memory_allocation(self, memory_manager):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Test allocation
        size = 1024 * 1024  # 1MB
        buffer = await memory_manager.allocate_buffer(0, size)
        assert buffer is not None
        assert buffer.size()[0] == size

        # Test memory info
        info = memory_manager.get_memory_info(0)
        assert info["reserved"] >= size

        # Test cleanup
        await memory_manager.free_buffer(buffer)
        info = memory_manager.get_memory_info(0)
        assert info["reserved"] < size