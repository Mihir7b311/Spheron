# tests/execution_engine/batch_processor/test_batch_manager.py

import pytest
import torch
import asyncio
from src.execution_engine.batch_processor.batch_manager import BatchManager
from src.execution_engine.cuda import CUDAStreamManager
from src.common.exceptions import BatchError

class TestBatchManager:
    @pytest.fixture
    async def setup_batch_manager(self):
        """Setup batch manager with dependencies"""
        stream_manager = CUDAStreamManager()
        manager = BatchManager(stream_manager)
        yield manager
        await manager.cleanup()

    @pytest.fixture
    def sample_input_batch(self):
        """Create sample batch input"""
        return [torch.randn(1, 10) for _ in range(32)]

    @pytest.fixture
    def sample_model(self):
        """Create sample model for testing"""
        return torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )

    @pytest.mark.asyncio
    async def test_basic_batch_processing(self, setup_batch_manager, sample_input_batch):
        """Test basic batch processing"""
        manager = setup_batch_manager
        
        # Add requests to batch
        for tensor in sample_input_batch:
            request = {
                "input": tensor,
                "function": "model(input)",
                "id": f"req_{id(tensor)}"
            }
            await manager.add_request(request, gpu_id=0)

        # Process batch
        results = await manager.process_batch(gpu_id=0)
        
        assert len(results) == len(sample_input_batch)
        assert all(isinstance(r["output"], torch.Tensor) for r in results)
        assert all(r["status"] == "success" for r in results)

    @pytest.mark.asyncio
    async def test_dynamic_batching(self, setup_batch_manager):
        """Test dynamic batch size adjustment"""
        manager = setup_batch_manager
        
        async def submit_request(delay):
            await asyncio.sleep(delay)
            request = {
                "input": torch.randn(1, 10),
                "function": "model(input)",
                "id": f"req_{delay}"
            }
            await manager.add_request(request, gpu_id=0)

        # Submit requests with different delays
        tasks = [submit_request(i*0.1) for i in range(5)]
        await asyncio.gather(*tasks)

        results = await manager.process_batch(gpu_id=0)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_batch_optimization(self, setup_batch_manager, sample_input_batch):
        """Test batch size optimization"""
        manager = setup_batch_manager
        
        # Test different batch sizes
        batch_sizes = [8, 16, 32]
        timings = {}

        for size in batch_sizes:
            start = asyncio.get_event_loop().time()
            
            for tensor in sample_input_batch[:size]:
                request = {
                    "input": tensor,
                    "function": "model(input)",
                    "id": f"req_{id(tensor)}"
                }
                await manager.add_request(request, gpu_id=0)

            results = await manager.process_batch(gpu_id=0)
            end = asyncio.get_event_loop().time()
            
            timings[size] = end - start

        # Larger batches should be more efficient per item
        assert timings[32]/32 < timings[8]/8

    @pytest.mark.asyncio
    async def test_multi_gpu_batching(self, setup_batch_manager, sample_input_batch):
        """Test batch processing across multiple GPUs"""
        if torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 GPUs")

        manager = setup_batch_manager
        
        # Split requests between GPUs
        for i, tensor in enumerate(sample_input_batch):
            gpu_id = i % 2  # Alternate between GPUs
            request = {
                "input": tensor,
                "function": "model(input)",
                "id": f"req_{id(tensor)}"
            }
            await manager.add_request(request, gpu_id=gpu_id)

        # Process on both GPUs
        results_gpu0 = await manager.process_batch(gpu_id=0)
        results_gpu1 = await manager.process_batch(gpu_id=1)
        
        assert len(results_gpu0) + len(results_gpu1) == len(sample_input_batch)

    @pytest.mark.asyncio
    async def test_batch_priority(self, setup_batch_manager):
        """Test batch priority handling"""
        manager = setup_batch_manager
        
        # Add requests with different priorities
        priorities = [0, 2, 1]  # Low, High, Medium
        for priority in priorities:
            request = {
                "input": torch.randn(1, 10),
                "function": "model(input)",
                "id": f"req_p{priority}",
                "priority": priority
            }
            await manager.add_request(request, gpu_id=0)

        results = await manager.process_batch(gpu_id=0)
        result_order = [int(r["request_id"].split('p')[1]) for r in results]
        
        # Higher priority should be processed first
        assert result_order[0] == 2  # High priority

    @pytest.mark.asyncio
    async def test_error_handling(self, setup_batch_manager):
        """Test error handling in batch processing"""
        manager = setup_batch_manager
        
        # Add request with invalid function
        request = {
            "input": torch.randn(1, 10),
            "function": "invalid_function(input)",
            "id": "error_req"
        }
        await manager.add_request(request, gpu_id=0)
        
        results = await manager.process_batch(gpu_id=0)
        assert results[0]["status"] == "error"
        assert "error" in results[0]

    @pytest.mark.asyncio
    async def test_batch_timeout(self, setup_batch_manager):
        """Test batch timeout handling"""
        manager = setup_batch_manager
        
        # Add slow request
        request = {
            "input": torch.randn(1, 10),
            "function": "time.sleep(2); model(input)",
            "id": "slow_req"
        }
        await manager.add_request(request, gpu_id=0)
        
        with pytest.raises(BatchError):
            await manager.process_batch(gpu_id=0)

    @pytest.mark.asyncio
    async def test_stream_management(self, setup_batch_manager, sample_input_batch):
        """Test CUDA stream management"""
        manager = setup_batch_manager
        
        # Add requests
        for tensor in sample_input_batch:
            request = {
                "input": tensor,
                "function": "model(input)",
                "id": f"req_{id(tensor)}"
            }
            await manager.add_request(request, gpu_id=0)

        # Process with stream recording
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        results = await manager.process_batch(gpu_id=0)
        end_event.record()
        
        end_event.synchronize()
        assert start_event.elapsed_time(end_event) > 0

    @pytest.mark.asyncio
    async def test_memory_management(self, setup_batch_manager, sample_model):
        """Test memory management during batching"""
        manager = setup_batch_manager
        initial_memory = torch.cuda.memory_allocated()
        
        # Process large batch
        large_batch = [torch.randn(100, 100) for _ in range(32)]
        for tensor in large_batch:
            request = {
                "input": tensor,
                "function": "model(input)",
                "id": f"req_{id(tensor)}"
            }
            await manager.add_request(request, gpu_id=0)

        await manager.process_batch(gpu_id=0)
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        assert final_memory <= initial_memory * 1.1  # Allow small overhead

    @pytest.mark.asyncio
    async def test_batch_statistics(self, setup_batch_manager, sample_input_batch):
        """Test batch statistics collection"""
        manager = setup_batch_manager
        
        # Process batch
        for tensor in sample_input_batch:
            request = {
                "input": tensor,
                "function": "model(input)",
                "id": f"req_{id(tensor)}"
            }
            await manager.add_request(request, gpu_id=0)

        results = await manager.process_batch(gpu_id=0)
        
        # Check statistics
        stats = manager.get_stats()
        assert stats["total_batches"] > 0
        assert stats["average_batch_size"] > 0
        assert stats["total_processed"] == len(sample_input_batch)

    @pytest.mark.asyncio
    async def test_queue_management(self, setup_batch_manager):
        """Test request queue management"""
        manager = setup_batch_manager
        
        # Fill queue to capacity
        for i in range(manager.max_queue_size + 1):
            request = {
                "input": torch.randn(1, 10),
                "function": "model(input)",
                "id": f"req_{i}"
            }
            if i < manager.max_queue_size:
                await manager.add_request(request, gpu_id=0)
            else:
                with pytest.raises(BatchError):
                    await manager.add_request(request, gpu_id=0)

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, setup_batch_manager):
        """Test concurrent batch processing"""
        manager = setup_batch_manager
        
        async def process_batch(gpu_id):
            for _ in range(5):
                request = {
                    "input": torch.randn(1, 10),
                    "function": "model(input)",
                    "id": f"req_{gpu_id}_{_}"
                }
                await manager.add_request(request, gpu_id=gpu_id)
            return await manager.process_batch(gpu_id=gpu_id)

        # Process on multiple GPUs concurrently
        tasks = [process_batch(i) for i in range(min(2, torch.cuda.device_count()))]
        results = await asyncio.gather(*tasks)
        
        assert all(len(r) == 5 for r in results)



    @pytest.mark.asyncio
    async def test_adaptive_batching(self, setup_batch_manager):
        """Test adaptive batch size adjustment"""
        manager = setup_batch_manager
        
        # Test with different load conditions
        load_levels = [0.3, 0.7, 0.9]  # Low, Medium, High
        for load in load_levels:
            manager._current_load = load
            optimal_batch = await manager._get_optimal_batch_size()
            if load > 0.8:
                assert optimal_batch < manager.max_batch_size
            else:
                assert optimal_batch == manager.max_batch_size

    @pytest.mark.asyncio
    async def test_priority_queue(self, setup_batch_manager):
        """Test priority-based queue management"""
        manager = setup_batch_manager
        priorities = [0, 2, 1]  # Low, High, Medium
        
        for priority in priorities:
            await manager.add_request(
                {
                    "input": torch.randn(1, 10),
                    "priority": priority,
                    "id": f"req_{priority}"
                },
                gpu_id=0
            )
        
        batch = await manager._get_next_batch()
        assert batch[0]["id"] == "req_2"  # High priority first