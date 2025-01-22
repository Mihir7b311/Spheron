# tests/runtime/test_cuda_context.py

import pytest
import torch
import asyncio
from src.runtime.cuda_context import CUDAContext
from src.common.exceptions import CUDAError

class TestCUDAContext:
    @pytest.fixture
    def cuda_available(self):
        """Check if CUDA is available"""
        return torch.cuda.is_available()

    @pytest.fixture
    async def setup_context(self, cuda_available):
        """Setup test CUDA context"""
        if not cuda_available:
            pytest.skip("CUDA not available")
        return CUDAContext(gpu_id=0)

    # tests/runtime/test_cuda_context.py

    @pytest.mark.asyncio
    async def test_context_initialization(self, setup_context):
        """Test CUDA context initialization"""
        context = setup_context
        assert context.gpu_id == 0
        assert isinstance(context.device, torch.device)
        assert isinstance(context.stream, torch.cuda.Stream)
        # Remove memory_pool check since we're not using CUDAPluggableAllocator
        assert torch.cuda.current_device() == context.gpu_id

    @pytest.mark.asyncio
    async def test_context_info(self, setup_context):
        """Test getting context information"""
        context = setup_context
        info = context.get_context_info()
        
        assert "gpu_id" in info
        assert "device" in info
        assert "stream" in info
        assert "memory_allocated" in info
        assert "memory_reserved" in info
        assert isinstance(info["memory_allocated"], int)

    @pytest.mark.asyncio
    async def test_context_manager(self, setup_context):
        """Test context manager functionality"""
        context = setup_context
        
        with context as ctx:
            assert torch.cuda.current_device() == context.gpu_id
            # Perform some CUDA operations
            tensor = torch.zeros(100, device=ctx.device)
            assert tensor.device.index == context.gpu_id

    @pytest.mark.asyncio
    async def test_memory_tracking(self, setup_context):
        """Test memory tracking in context"""
        context = setup_context
        
        # Get initial memory state
        initial_info = context.get_context_info()
        initial_allocated = initial_info["memory_allocated"]
        
        # Allocate some memory
        tensor = torch.zeros(1000000, device=context.device)  # 4MB tensor
        
        # Check new memory state
        new_info = context.get_context_info()
        assert new_info["memory_allocated"] > initial_allocated

        # Cleanup
        del tensor
        torch.cuda.empty_cache()

    @pytest.mark.asyncio
    async def test_stream_synchronization(self, setup_context):
        """Test stream synchronization"""
        context = setup_context
        
        with context:
            # Execute async operation
            tensor = torch.zeros(100, device=context.device)
            tensor.add_(1)
            
            # Synchronize
            context.stream.synchronize()
            
            # Verify operation completed
            assert torch.all(tensor == 1)

    @pytest.mark.asyncio
    async def test_multiple_contexts(self, cuda_available):
        """Test multiple CUDA contexts"""
        if not cuda_available:
            pytest.skip("CUDA not available")
            
        n_devices = torch.cuda.device_count()
        if n_devices < 2:
            pytest.skip("Need at least 2 GPUs for this test")
            
        context1 = CUDAContext(gpu_id=0)
        context2 = CUDAContext(gpu_id=1)
        
        assert context1.gpu_id != context2.gpu_id
        assert context1.device != context2.device
        assert context1.stream != context2.stream

    @pytest.mark.asyncio
    async def test_error_handling(self, cuda_available):
        """Test error handling in context"""
        if not cuda_available:
            pytest.skip("CUDA not available")
            
        with pytest.raises(CUDAError):
            # Try to create context with invalid GPU ID
            invalid_context = CUDAContext(gpu_id=99999)

    @pytest.mark.asyncio
    async def test_memory_pool(self, setup_context):
        """Test memory pool functionality"""
        context = setup_context
        
        assert context.memory_pool is not None
        
        # Allocate and free memory repeatedly
        for _ in range(5):
            tensor = torch.zeros(1000000, device=context.device)
            del tensor
            torch.cuda.empty_cache()

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, setup_context):
        """Test concurrent operations in context"""
        context = setup_context
        
        async def async_operation(value):
            with context:
                tensor = torch.zeros(100, device=context.device)
                tensor.add_(value)
                await asyncio.sleep(0.1)  # Simulate work
                return tensor.mean().item()
        
        # Run multiple operations concurrently
        tasks = [async_operation(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(isinstance(r, float) for r in results)

    @pytest.mark.asyncio
    async def test_context_cleanup(self, setup_context):
        """Test context cleanup"""
        context = setup_context

        # Record initial memory state
        initial_info = context.get_context_info()
        initial_memory = initial_info['memory_allocated']

        # Perform some operations that allocate memory
        tensor = torch.zeros(10000, device=context.device)

        # Release tensor memory
        del tensor
        torch.cuda.empty_cache()

        # Record memory state after cleanup
        final_info = context.get_context_info()
        final_memory = final_info['memory_allocated']

        # Assert that memory after cleanup is less than or equal to initial memory
        assert final_memory <= initial_memory


    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.asyncio
    async def test_device_properties(self, setup_context):
        """Test accessing device properties"""
        context = setup_context
        properties = torch.cuda.get_device_properties(context.device)
        
        assert properties.total_memory > 0
        assert properties.major >= 0
        assert properties.minor >= 0
        assert isinstance(properties.multi_processor_count, int)