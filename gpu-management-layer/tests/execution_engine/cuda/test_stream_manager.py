# tests/execution_engine/cuda/test_stream_manager.py

import pytest
import torch
import asyncio
from src.execution_engine.cuda.stream_manager import CUDAStreamManager, StreamInfo
from src.common.exceptions import CUDAError

class TestCUDAStreamManager:
    @pytest.fixture
    async def stream_manager(self):
        """Setup stream manager"""
        manager = CUDAStreamManager()
        yield manager
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_stream_creation(self, stream_manager):
        """Test stream creation"""
        stream_info = await stream_manager.create_stream(
            gpu_id=0,
            stream_id="test_stream"
        )
        
        assert isinstance(stream_info, StreamInfo)
        assert stream_info.stream_id == "test_stream"
        assert stream_info.gpu_id == 0
        assert isinstance(stream_info.stream, torch.cuda.Stream)
        assert stream_info.is_active

    @pytest.mark.asyncio
    async def test_stream_reuse(self, stream_manager):
        """Test stream reuse"""
        stream1 = await stream_manager.create_stream(gpu_id=0, stream_id="reuse")
        stream2 = await stream_manager.get_stream("reuse")
        
        assert stream1.stream_id == stream2.stream_id
        assert stream1.stream == stream2.stream

    @pytest.mark.asyncio
    async def test_stream_priorities(self, stream_manager):
        """Test stream priorities"""
        high_priority = await stream_manager.create_stream(
            gpu_id=0,
            priority=-1  # High priority
        )
        
        low_priority = await stream_manager.create_stream(
            gpu_id=0,
            priority=0  # Default priority
        )
        
        assert high_priority.priority < low_priority.priority

    @pytest.mark.asyncio
    async def test_stream_synchronization(self, stream_manager):
        """Test stream synchronization"""
        stream_info = await stream_manager.create_stream(gpu_id=0)
        
        with torch.cuda.stream(stream_info.stream):
            tensor = torch.zeros(1000, device=f'cuda:{stream_info.gpu_id}')
            tensor.add_(1)
            
        await stream_manager.synchronize_stream(stream_info.stream_id)
        assert torch.all(tensor == 1)

    @pytest.mark.asyncio
    async def test_multiple_streams(self, stream_manager):
        """Test multiple streams"""
        streams = []
        for i in range(3):
            stream = await stream_manager.create_stream(
                gpu_id=0,
                stream_id=f"stream_{i}"
            )
            streams.append(stream)
            
        assert len(streams) == 3
        assert len(set(s.stream_id for s in streams)) == 3

    @pytest.mark.asyncio
    async def test_stream_cleanup(self, stream_manager):
        """Test stream cleanup"""
        stream = await stream_manager.create_stream(gpu_id=0)
        await stream_manager.release_stream(stream.stream_id)
        
        assert stream.stream_id not in stream_manager.streams
        assert stream_manager.gpu_stream_count[0] == 0

    @pytest.mark.asyncio
    async def test_stream_limits(self, stream_manager):
        """Test stream limits per GPU"""
        streams = []
        
        # Try to exceed limit
        with pytest.raises(CUDAError):
            for _ in range(stream_manager.max_streams_per_gpu + 1):
                stream = await stream_manager.create_stream(gpu_id=0)
                streams.append(stream)

    @pytest.mark.asyncio
    async def test_stream_events(self, stream_manager):
        """Test stream event handling"""
        stream = await stream_manager.create_stream(gpu_id=0)
        
        # Create events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.cuda.stream(stream.stream):
            start_event.record()
            tensor = torch.randn(1000, 1000, device=f'cuda:{stream.gpu_id}')
            tensor = tensor @ tensor
            end_event.record()
            
        await stream_manager.wait_stream(stream.stream_id)
        elapsed = start_event.elapsed_time(end_event)
        assert elapsed > 0

    @pytest.mark.asyncio
    async def test_concurrent_streams(self, stream_manager):
        """Test concurrent stream operations"""
        async def stream_task():
            stream = await stream_manager.create_stream(gpu_id=0)
            with torch.cuda.stream(stream.stream):
                tensor = torch.zeros(1000, device=f'cuda:{stream.gpu_id}')
                tensor.add_(1)
            await stream_manager.wait_stream(stream.stream_id)
            return tensor.item()
            
        tasks = [stream_task() for _ in range(3)]
        results = await asyncio.gather(*tasks)
        assert all(r == 1 for r in results)

    @pytest.mark.asyncio
    async def test_stream_timeout(self, stream_manager):
        """Test stream wait timeout"""
        stream = await stream_manager.create_stream(gpu_id=0)
        
        # Long operation
        with torch.cuda.stream(stream.stream):
            tensor = torch.randn(5000, 5000, device=f'cuda:{stream.gpu_id}')
            tensor = tensor @ tensor
            
        with pytest.raises(CUDAError):
            await stream_manager.wait_stream(
                stream.stream_id,
                timeout=0.0001  # Very short timeout
            )

    @pytest.mark.asyncio
    async def test_stream_stats(self, stream_manager):
        """Test stream statistics"""
        stream = await stream_manager.create_stream(gpu_id=0)
        
        # Perform some operations
        with torch.cuda.stream(stream.stream):
            tensor = torch.zeros(1000, device=f'cuda:{stream.gpu_id}')
            tensor.add_(1)
            
        stats = stream_manager.get_stream_stats(stream.stream_id)
        assert "total_operations" in stats
        assert "last_used" in stats
        assert stats["is_active"]

    @pytest.mark.asyncio
    async def test_idle_cleanup(self, stream_manager):
        """Test idle stream cleanup"""
        stream = await stream_manager.create_stream(gpu_id=0)
        
        # Wait for stream to become idle
        await asyncio.sleep(0.1)
        
        await stream_manager.cleanup_idle_streams(idle_timeout=0.05)
        assert stream.stream_id not in stream_manager.streams



    @pytest.mark.asyncio
    async def test_stream_callback_handling(self, stream_manager):
        """Test stream callback mechanisms"""
        callback_executed = False
        
        def stream_callback():
            nonlocal callback_executed
            callback_executed = True
        
        stream = await stream_manager.create_stream(gpu_id=0)
        await stream_manager.add_callback(
            stream.stream_id,
            stream_callback
        )
        
        # Execute operation and verify callback
        with torch.cuda.stream(stream.stream):
            tensor = torch.zeros(100, device=f'cuda:{stream.gpu_id}')
            
        await stream_manager.synchronize_stream(stream.stream_id)
        assert callback_executed

    @pytest.mark.asyncio
    async def test_stream_memory_ordering(self, stream_manager):
        """Test memory ordering between streams"""
        stream1 = await stream_manager.create_stream(gpu_id=0)
        stream2 = await stream_manager.create_stream(gpu_id=0)
        
        # Create dependency between streams
        tensor = torch.zeros(100, device=f'cuda:{stream1.gpu_id}')
        
        with torch.cuda.stream(stream1.stream):
            tensor.add_(1)
            
        stream2.stream.wait_stream(stream1.stream)
        with torch.cuda.stream(stream2.stream):
            tensor.add_(2)
            
        await stream_manager.wait_stream(stream2.stream_id)
        assert tensor.item() == 3

    @pytest.mark.asyncio
    async def test_stream_capture(self, stream_manager):
        """Test CUDA graph capture in streams"""
        if not hasattr(torch.cuda, 'graphs'):
            pytest.skip("CUDA graphs not supported")
            
        stream = await stream_manager.create_stream(gpu_id=0)
        
        # Capture operations
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            tensor = torch.zeros(100, device=f'cuda:{stream.gpu_id}')
            tensor.add_(1)
            
        # Replay graph
        g.replay()
        assert tensor.item() == 1




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


    

if __name__ == "__main__":
    pytest.main(["-v"])