# tests/execution_engine/runtime/test_cuda_context.py

import pytest
import torch
import asyncio
from src.execution_engine.runtime.cuda_context import CUDAContext
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
        
        context = CUDAContext(gpu_id=0)
        yield context
        await context.cleanup()

    @pytest.mark.asyncio
    async def test_context_initialization(self, setup_context):
        """Test CUDA context initialization"""
        context = setup_context
        assert context.gpu_id == 0
        assert isinstance(context.device, torch.device)
        assert isinstance(context.stream, torch.cuda.Stream)
        assert context.memory_pool is not None
        assert torch.cuda.current_device() == context.gpu_id

    @pytest.mark.asyncio
    async def test_context_info(self, setup_context):
        """Test getting context information"""
        context = setup_context
        info = context.get_context_info()
        
        assert "gpu_id" in info
        assert "device" in info
        assert "stream" in info
        assert "memory" in info
        assert info["memory"]["total"] > 0
        assert info["memory"]["allocated"] >= 0
        assert info["memory"]["reserved"] >= 0
        assert info["memory"]["utilization"] >= 0

    @pytest.mark.asyncio
    async def test_memory_management(self, setup_context):
        """Test memory management functionality"""
        context = setup_context
        
        # Track initial memory state
        initial_info = context.get_context_info()
        initial_allocated = initial_info["memory"]["allocated"]
        
        # Allocate tensor
        with context:
            tensor = torch.zeros(1000000, device=context.device)  # 4MB tensor
            mid_info = context.get_context_info()
            assert mid_info["memory"]["allocated"] > initial_allocated

        # Verify cleanup
        final_info = context.get_context_info()
        assert final_info["memory"]["allocated"] <= initial_allocated

    @pytest.mark.asyncio
    async def test_memory_threshold(self, setup_context):
        """Test memory threshold monitoring"""
        context = setup_context
        initial_memory = torch.cuda.memory_allocated(context.device)

        # Push memory usage close to threshold
        tensors = []
        with pytest.raises(CUDAError):
            while True:
                tensors.append(torch.zeros(1024*1024*256, device=context.device))  # 1GB
                if not context.check_memory_status():
                    break

        # Cleanup
        del tensors
        torch.cuda.empty_cache()
        assert torch.cuda.memory_allocated(context.device) <= initial_memory

    @pytest.mark.asyncio
    async def test_stream_synchronization(self, setup_context):
        """Test stream synchronization"""
        context = setup_context
        
        with context:
            # Execute async operation
            tensor = torch.zeros(100, device=context.device)
            context.stream.synchronize()
            
            # Record events
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            tensor.add_(1)
            end_event.record()
            
            # Wait for completion
            end_event.synchronize()
            
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

        await context1.cleanup()
        await context2.cleanup()

    @pytest.mark.asyncio
    async def test_context_reuse(self, setup_context):
        """Test context reuse"""
        context = setup_context
        
        # Multiple operations in same context
        for _ in range(5):
            with context:
                tensor = torch.zeros(100, device=context.device)
                tensor.add_(1)
                context.stream.synchronize()

        # Context should still be valid
        assert context.active
        info = context.get_context_info()
        assert info["status"] != "error"

    @pytest.mark.asyncio
    async def test_error_handling(self, setup_context):
        """Test error handling in context"""
        context = setup_context
        
        # Invalid CUDA operations
        with context:
            with pytest.raises(RuntimeError):
                # Access invalid memory
                tensor = torch.zeros(1, device=context.device)
                tensor[999999] = 1

        # Context should handle error gracefully
        assert context.active
        assert context.last_error is not None

    @pytest.mark.asyncio
    async def test_memory_pool_management(self, setup_context):
        """Test memory pool management"""
        context = setup_context
        
        # Track memory pool states
        initial_pool = context.memory_pool
        
        with context:
            # Allocate and free multiple times
            for _ in range(5):
                tensor = torch.zeros(1000000, device=context.device)
                del tensor
                torch.cuda.empty_cache()
                
        # Memory pool should be managed properly
        assert context.memory_pool >= initial_pool

    @pytest.mark.asyncio
    async def test_concurrent_access(self, setup_context):
        """Test concurrent context access"""
        context = setup_context
        
        async def context_operation():
            with context:
                tensor = torch.zeros(100, device=context.device)
                await asyncio.sleep(0.1)
                tensor.add_(1)
                context.stream.synchronize()
                return tensor.item()

        # Run multiple concurrent operations
        tasks = [context_operation() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert all(r == 1 for r in results)

    @pytest.mark.asyncio
    async def test_context_priorities(self, setup_context):
        """Test context stream priorities"""
        context = setup_context
        
        # Create high priority stream
        high_priority_stream = torch.cuda.Stream(
            device=context.device,
            priority=-1  # High priority
        )
        
        with torch.cuda.stream(high_priority_stream):
            tensor = torch.zeros(100, device=context.device)
            tensor.add_(1)
            
        high_priority_stream.synchronize()
        assert torch.all(tensor == 1)

    @pytest.mark.asyncio
    async def test_memory_fragmentation(self, setup_context):
        """Test memory fragmentation handling"""
        context = setup_context
        
        # Create fragmented memory pattern
        tensors = {}
        with context:
            # Allocate alternating sizes
            for i in range(10):
                size = 1000000 if i % 2 == 0 else 500000
                tensors[i] = torch.zeros(size, device=context.device)
            
            # Free alternate tensors
            for i in range(0, 10, 2):
                del tensors[i]
            
            # Try to allocate large tensor
            with pytest.raises(RuntimeError):
                large_tensor = torch.zeros(2000000, device=context.device)

    @pytest.mark.asyncio
    async def test_context_cleanup(self, setup_context):
        """Test context cleanup"""
        context = setup_context
        initial_memory = torch.cuda.memory_allocated()

        with context:
            tensor = torch.zeros(1000000, device=context.device)
            del tensor

        await context.cleanup()
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()

        assert final_memory <= initial_memory

    @pytest.mark.asyncio
    async def test_out_of_memory_recovery(self, setup_context):
        """Test recovery from out of memory condition"""
        context = setup_context
        
        try:
            with context:
                # Try to allocate more than available memory
                huge_tensor = torch.zeros(1024*1024*1024*10, device=context.device)  # 40GB
        except RuntimeError:
            # Should recover gracefully
            assert context.active
            info = context.get_context_info()
            assert info["status"] != "error"

    @pytest.mark.asyncio
    async def test_resource_tracking(self, setup_context):
        """Test resource usage tracking"""
        context = setup_context
        
        with context:
            tensor = torch.zeros(1000000, device=context.device)
            info = context.get_context_info()
            
            assert info["memory"]["allocated"] > 0
            assert "compute_units_active" in info
            assert "temperature" in info
            assert "power_usage" in info

    @pytest.mark.asyncio
    async def test_peer_access(self, cuda_available):
        """Test peer access between contexts"""
        if not cuda_available or torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 GPUs")
            
        context1 = CUDAContext(gpu_id=0)
        context2 = CUDAContext(gpu_id=1)
        
        # Enable peer access
        if torch.cuda.can_device_access_peer(0, 1):
            torch.cuda.device(0).enable_peer_access(1)
            
            with context1:
                tensor1 = torch.zeros(100, device=context1.device)
                
            with context2:
                tensor2 = tensor1.to(context2.device)
                
            assert tensor2.device.index == 1

        await context1.cleanup()
        await context2.cleanup()



    @pytest.mark.asyncio
    async def test_multi_stream_ordering(self, setup_context):
        """Test ordering of operations across multiple streams"""
        context = setup_context
        
        # Create multiple streams
        stream1 = torch.cuda.Stream(device=context.device)
        stream2 = torch.cuda.Stream(device=context.device)
        
        # Test operation ordering
        tensor = torch.zeros(100, device=context.device)
        event = torch.cuda.Event()
        
        with torch.cuda.stream(stream1):
            tensor.add_(1)
            event.record()
            
        stream2.wait_event(event)
        with torch.cuda.stream(stream2):
            tensor.mul_(2)
            
        stream2.synchronize()
        assert tensor.item() == 2

    @pytest.mark.asyncio
    async def test_context_profiling(self, setup_context):
        """Test CUDA profiler integration"""
        context = setup_context
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            with context:
                tensor = torch.randn(1000, 1000, device=context.device)
                tensor = tensor @ tensor
                
        profile_stats = prof.key_averages().table()
        assert "gemm" in profile_stats.lower()