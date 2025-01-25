# tests/execution_engine/cuda/test_context_manager.py

import pytest
import torch
import asyncio
from src.execution_engine.cuda.context_manager import CUDAContextManager
from src.common.exceptions import CUDAError

class TestCUDAContextManager:
    @pytest.fixture
    async def context_manager(self):
        """Setup CUDA context manager"""
        manager = CUDAContextManager()
        await manager._initialize()
        yield manager
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_initialization(self, context_manager):
        """Test context manager initialization"""
        assert context_manager.initialized is True
        assert context_manager.device_count > 0
        assert len(context_manager.contexts) == 0

    @pytest.mark.asyncio
    async def test_context_creation(self, context_manager):
        """Test CUDA context creation"""
        context = await context_manager.create_context(gpu_id=0)
        
        assert context["gpu_id"] == 0
        assert isinstance(context["device"], torch.device)
        assert isinstance(context["stream"], torch.cuda.Stream)
        assert "cuda_context" in context

    @pytest.mark.asyncio
    async def test_context_reuse(self, context_manager):
        """Test context reuse"""
        context1 = await context_manager.get_context(gpu_id=0)
        context2 = await context_manager.get_context(gpu_id=0)
        
        assert context1["gpu_id"] == context2["gpu_id"]
        assert context1["stream"] == context2["stream"]

    @pytest.mark.asyncio
    async def test_multiple_gpus(self, context_manager):
        """Test handling multiple GPUs"""
        if torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 GPUs")
            
        context0 = await context_manager.create_context(gpu_id=0)
        context1 = await context_manager.create_context(gpu_id=1)
        
        assert context0["gpu_id"] != context1["gpu_id"]
        assert context0["device"] != context1["device"]
        assert context0["stream"] != context1["stream"]

    @pytest.mark.asyncio
    async def test_context_info(self, context_manager):
        """Test getting context information"""
        context = await context_manager.create_context(gpu_id=0)
        info = context_manager.get_context_info(gpu_id=0)
        
        assert info["gpu_id"] == 0
        assert "device" in info
        assert "stream" in info
        assert "memory" in info
        assert info["memory"]["total"] > 0

    @pytest.mark.asyncio
    async def test_memory_tracking(self, context_manager):
        """Test memory tracking"""
        context = await context_manager.create_context(gpu_id=0)
        initial_memory = context_manager.get_context_info(0)["memory"]["allocated"]
        
        # Allocate tensor
        tensor = torch.zeros(1000000, device=context["device"])  # 4MB
        
        new_memory = context_manager.get_context_info(0)["memory"]["allocated"]
        assert new_memory > initial_memory
        
        del tensor
        torch.cuda.empty_cache()

    @pytest.mark.asyncio
    async def test_stream_synchronization(self, context_manager):
        """Test stream synchronization"""
        context = await context_manager.get_context(gpu_id=0)
        
        # Record events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.cuda.stream(context["stream"]):
            start_event.record()
            # Some computation
            tensor = torch.randn(1000, 1000, device=context["device"])
            tensor = tensor @ tensor
            end_event.record()
            
        await context_manager.synchronize_context(0)
        elapsed = start_event.elapsed_time(end_event)
        assert elapsed > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, context_manager):
        """Test error handling"""
        # Invalid GPU ID
        with pytest.raises(CUDAError):
            await context_manager.create_context(gpu_id=9999)
            
        # Release non-existent context
        success = await context_manager.release_context(9999)
        assert not success

    @pytest.mark.asyncio
    async def test_memory_constraints(self, context_manager):
        """Test memory constraint handling"""
        context = await context_manager.create_context(gpu_id=0)
        
        # Try to allocate too much memory
        try:
            huge_tensor = torch.zeros(1024*1024*1024*10, device=context["device"])  # 40GB
        except RuntimeError as e:
            assert "out of memory" in str(e)

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, context_manager):
        """Test concurrent context operations"""
        async def context_task(gpu_id):
            context = await context_manager.get_context(gpu_id)
            tensor = torch.zeros(1000, device=context["device"])
            await asyncio.sleep(0.1)
            return tensor.device.index
            
        tasks = [context_task(0) for _ in range(5)]
        results = await asyncio.gather(*tasks)
        assert all(r == 0 for r in results)

    @pytest.mark.asyncio
    async def test_cleanup(self, context_manager):
        """Test cleanup"""
        context = await context_manager.create_context(gpu_id=0)
        tensor = torch.zeros(1000000, device=context["device"])
        
        await context_manager.cleanup()
        assert len(context_manager.contexts) == 0

    @pytest.mark.asyncio
    async def test_peer_access(self, context_manager):
        """Test peer access between GPUs"""
        if torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 GPUs")
            
        if torch.cuda.can_device_access_peer(0, 1):
            context0 = await context_manager.create_context(gpu_id=0)
            context1 = await context_manager.create_context(gpu_id=1)
            
            # Test data transfer
            data = torch.randn(1000, device=context0["device"])
            data = data.to(context1["device"])
            assert data.device.index == 1

    @pytest.mark.asyncio
    async def test_memory_fragmentation(self, context_manager):
        """Test memory fragmentation handling"""
        context = await context_manager.create_context(gpu_id=0)
        device = context["device"]
        
        tensors = []
        # Create fragmented pattern
        for _ in range(10):
            tensors.append(torch.zeros(1000000, device=device))
            
        # Delete alternating tensors
        for i in range(0, len(tensors), 2):
            del tensors[i]
        
        # Try to allocate large tensor
        with pytest.raises(RuntimeError):
            torch.zeros(2000000, device=device)
            
        del tensors
        torch.cuda.empty_cache()


    @pytest.mark.asyncio
    async def test_context_priority_scheduling(self, context_manager):
        """Test context priority scheduling"""
        high_priority = await context_manager.create_context(
            gpu_id=0, 
            priority_level="high"
        )
        low_priority = await context_manager.create_context(
            gpu_id=0,
            priority_level="low"
        )
        
        # Verify priority execution
        results = await asyncio.gather(
            self._run_on_context(high_priority),
            self._run_on_context(low_priority)
        )
        assert results[0]["completion_time"] < results[1]["completion_time"]

    @pytest.mark.asyncio
    async def test_memory_pool_management(self, context_manager):
        """Test memory pool configurations"""
        context = await context_manager.create_context(
            gpu_id=0,
            memory_pool_config={
                "initial_size": "1GB",
                "max_size": "4GB",
                "growth_factor": 2.0
            }
        )
        
        # Test pool growth
        allocations = []
        for size in [256, 512, 1024]:  # MB
            tensor = torch.zeros(size * 256 * 1024, device=context["device"])
            allocations.append(tensor)
            
        pool_info = context_manager.get_memory_pool_info(0)
        assert pool_info["current_size"] > pool_info["initial_size"]

    @pytest.mark.asyncio
    async def test_unified_memory(self, context_manager):
        """Test unified memory support"""
        context = await context_manager.create_context(
            gpu_id=0,
            enable_unified_memory=True
        )
        
        # Test unified memory allocation
        unified_tensor = torch.cuda.FloatTensor(1000).pin_memory()
        assert unified_tensor.is_pinned()



    @pytest.mark.asyncio
    async def test_nvml_integration(self, context_manager):
        """Test NVML monitoring integration"""
        context = await context_manager.create_context(gpu_id=0)
        
        # Monitor GPU metrics
        metrics = await context_manager.get_gpu_metrics(0)
        assert "temperature" in metrics
        assert "power_usage" in metrics
        assert "utilization" in metrics

    @pytest.mark.asyncio
    async def test_multi_process_service(self, context_manager):
        """Test MPS (Multi-Process Service) integration"""
        if not context_manager.mps_available():
            pytest.skip("MPS not available")
            
        context = await context_manager.create_context(
            gpu_id=0,
            enable_mps=True
        )
        
        assert context["mps_active"]
        # Test MPS compute distribution
        compute_quota = await context_manager.get_mps_compute_quota(0)
        assert compute_quota > 0



        