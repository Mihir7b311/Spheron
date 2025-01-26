# tests/execution_engine/runtime/test_python_executor.py

import pytest
import torch
import asyncio
from src.execution_engine.runtime.python_executor import PythonExecutor
from src.execution_engine.cuda import CUDAContextManager, CUDAStreamManager
from src.execution_engine.batch_processor import BatchManager
from src.common.exceptions import RuntimeError

class TestPythonExecutor:
    @pytest.fixture
    async def setup_executor(self):
        """Setup test executor with dependencies"""
        cuda_context_manager = CUDAContextManager()
        stream_manager = CUDAStreamManager()
        batch_manager = BatchManager(stream_manager)
        executor = PythonExecutor(cuda_context_manager, batch_manager)
        
        yield executor
        
        await executor.cleanup()

    @pytest.fixture
    def sample_function(self):
        """Create sample ML function"""
        return """
        import torch
        
        def inference(x):
            model = torch.nn.Linear(10, 2)
            return model(x)
        """

    @pytest.fixture
    def cuda_profiler():
        """Setup CUDA profiler"""
        with torch.profiler.profile(use_cuda=True) as prof:
            yield prof

    @pytest.fixture
    async def cuda_streams():
        """Setup multiple CUDA streams"""
        streams = [torch.cuda.Stream() for _ in range(3)]
        yield streams
        torch.cuda.synchronize()
        for stream in streams:
            stream.synchronize()

    @pytest.fixture
    def large_test_model():
        """Create large test model"""
        return torch.nn.Sequential(
            *[torch.nn.Linear(1000, 1000) for _ in range(5)]
        )
    @pytest.mark.asyncio
    async def test_function_execution(self, setup_executor, sample_function):
        """Test basic function execution"""
        executor = setup_executor
        input_data = {"x": torch.randn(1, 10)}
        gpu_context = {"gpu_id": 0}
        
        result = await executor.execute_function(
            function_code=sample_function,
            input_data=input_data,
            gpu_context=gpu_context
        )
        
        assert result["results"] is not None
        assert "metrics" in result

    @pytest.mark.asyncio
    async def test_execution_with_model(self, setup_executor):
        """Test execution with pre-trained model"""
        model_code = """
        import torch
        
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        """
        
        input_data = {"input": torch.randn(1, 10)}
        gpu_context = {"gpu_id": 0}
        
        result = await setup_executor.execute_function(
            model_code,
            input_data,
            gpu_context
        )
        
        assert result["results"] is not None
        assert isinstance(result["results"][0]["output"], torch.Tensor)
        assert result["results"][0]["output"].shape == (1, 2)

    @pytest.mark.asyncio
    async def test_batch_execution(self, setup_executor, sample_function):
        """Test batch execution capability"""
        batch_size = 32
        inputs = [torch.randn(1, 10) for _ in range(batch_size)]
        input_data = {"inputs": inputs}
        gpu_context = {"gpu_id": 0}

        result = await setup_executor.execute_function(
            sample_function,
            input_data,
            gpu_context
        )
        
        assert len(result["results"]) == batch_size
        assert all(isinstance(r["output"], torch.Tensor) for r in result["results"])

    @pytest.mark.asyncio
    async def test_error_handling(self, setup_executor):
        """Test error handling in execution"""
        # Invalid function code
        bad_code = "invalid python code"
        with pytest.raises(RuntimeError):
            await setup_executor.execute_function(
                bad_code,
                {},
                {"gpu_id": 0}
            )

        # Invalid input
        with pytest.raises(RuntimeError):
            await setup_executor.execute_function(
                "def f(x): return x",
                {"x": "invalid input"},
                {"gpu_id": 0}
            )

    @pytest.mark.asyncio
    async def test_gpu_context_usage(self, setup_executor, sample_function):
        """Test GPU context handling"""
        input_data = {"x": torch.randn(1, 10)}
        gpu_context = {"gpu_id": 0}

        result = await setup_executor.execute_function(
            sample_function,
            input_data,
            gpu_context
        )
        
        # Verify correct GPU usage
        assert result["metrics"]["gpu_id"] == gpu_context["gpu_id"]
        assert "cuda_time" in result["metrics"]

    @pytest.mark.asyncio
    async def test_concurrent_execution(self, setup_executor, sample_function):
        """Test concurrent function execution"""
        inputs = [{"x": torch.randn(1, 10)} for _ in range(5)]
        gpu_context = {"gpu_id": 0}

        # Execute functions concurrently
        tasks = [
            setup_executor.execute_function(
                sample_function,
                input_data,
                gpu_context
            )
            for input_data in inputs
        ]
        
        results = await asyncio.gather(*tasks)
        assert len(results) == len(inputs)
        assert all("results" in r for r in results)

    @pytest.mark.asyncio
    async def test_memory_cleanup(self, setup_executor, sample_function):
        """Test memory cleanup after execution"""
        input_data = {"x": torch.randn(1, 10)}
        gpu_context = {"gpu_id": 0}

        # Get initial memory usage
        initial_memory = torch.cuda.memory_allocated()

        # Execute function
        await setup_executor.execute_function(
            sample_function,
            input_data,
            gpu_context
        )

        # Wait for cleanup
        await asyncio.sleep(0.1)
        torch.cuda.empty_cache()

        # Check memory cleaned up
        final_memory = torch.cuda.memory_allocated()
        assert final_memory <= initial_memory

    @pytest.mark.asyncio
    async def test_execution_timeout(self, setup_executor):
        """Test execution timeout handling"""
        slow_code = """
        import time
        def slow_function(x):
            time.sleep(2)
            return x
        """
        
        with pytest.raises(RuntimeError):
            await setup_executor.execute_function(
                slow_code,
                {"x": torch.randn(1, 10)},
                {"gpu_id": 0}
            )

    @pytest.mark.asyncio
    async def test_execution_metrics(self, setup_executor, sample_function):
        """Test execution metrics collection"""
        input_data = {"x": torch.randn(1, 10)}
        gpu_context = {"gpu_id": 0}

        result = await setup_executor.execute_function(
            sample_function,
            input_data,
            gpu_context
        )

        metrics = result["metrics"]
        assert "execution_time" in metrics
        assert "cuda_time" in metrics
        assert "batch_size" in metrics
        assert "memory_used" in metrics
        assert "compute_used" in metrics

    @pytest.mark.asyncio
    async def test_active_executions(self, setup_executor):
        """Test active executions tracking"""
        executions = setup_executor.get_active_executions()
        assert isinstance(executions, dict)
        assert len(executions) == 0

    # Additional tests for test_python_executor.py

    @pytest.mark.asyncio
    async def test_batch_optimization(self, setup_executor):
        """Test batch size optimization"""
        # Test with different batch sizes
        batch_configs = [8, 16, 32, 64]
        results = {}
        
        for batch_size in batch_configs:
            input_data = {
                "inputs": [torch.randn(1, 10) for _ in range(batch_size)]
            }
            result = await setup_executor.execute_function(
                self.sample_function,
                input_data,
                {"gpu_id": 0, "batch_size": batch_size}
            )
            results[batch_size] = result["metrics"]

        # Verify performance scaling
        assert results[32]["throughput"] > results[8]["throughput"]

    @pytest.mark.asyncio
    async def test_model_state_handling(self, setup_executor):
        """Test handling of model state"""
        # Model with state
        stateful_model = """
        import torch
        class StatefulModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0
                self.linear = torch.nn.Linear(10, 2)
                
            def forward(self, x):
                self.counter += 1
                return self.linear(x)
                
        model = StatefulModel()
        """
        
        # Execute multiple times
        for _ in range(3):
            result = await setup_executor.execute_function(
                stateful_model,
                {"x": torch.randn(1, 10)},
                {"gpu_id": 0}
            )
        
        # State should be preserved
        assert "counter" in result["model_state"]

    @pytest.mark.asyncio
    async def test_cuda_stream_handling(self, setup_executor):
        """Test CUDA stream management"""
        # Multiple streams test
        async def run_on_stream():
            return await setup_executor.execute_function(
                self.sample_function,
                {"x": torch.randn(1, 10)},
                {"gpu_id": 0, "stream_id": f"stream_{id}"}
            )

        # Run on multiple streams
        results = await asyncio.gather(
            *[run_on_stream() for _ in range(3)]
        )
        
        # Each should have different stream
        streams = [r["metrics"]["stream_id"] for r in results]
        assert len(set(streams)) == len(streams)

    @pytest.mark.asyncio
    async def test_dynamic_batching(self, setup_executor):
        """Test dynamic batching behavior"""
        async def submit_request(delay):
            await asyncio.sleep(delay)
            return await setup_executor.execute_function(
                self.sample_function,
                {"x": torch.randn(1, 10)},
                {"gpu_id": 0}
            )

        # Submit requests with different timing
        results = await asyncio.gather(
            *[submit_request(i*0.1) for i in range(5)]
        )
        
        # Should be batched appropriately
        batch_sizes = [r["metrics"]["batch_size"] for r in results]
        assert any(size > 1 for size in batch_sizes)

    @pytest.mark.asyncio
    async def test_resource_limits(self, setup_executor):
        """Test resource limit enforcement"""
        # Large model
        large_model = """
        import torch
        model = torch.nn.Sequential(
            *[torch.nn.Linear(1000, 1000) for _ in range(10)]
        )
        """
        
        # Should fail due to resource limits
        with pytest.raises(RuntimeError) as exc:
            await setup_executor.execute_function(
                large_model,
                {"x": torch.randn(1, 1000)},
                {"gpu_id": 0}
            )
        assert "resource limit exceeded" in str(exc.value)

    @pytest.mark.asyncio
    async def test_cuda_event_synchronization(self, setup_executor):
        """Test CUDA event synchronization"""
        events = []
        for _ in range(3):
            result = await setup_executor.execute_function(
                self.sample_function,
                {"x": torch.randn(1, 10)},
                {"gpu_id": 0, "record_events": True}
            )
            events.append(result["metrics"]["cuda_events"])
        
        # Events should be properly ordered
        assert all(e["start"] < e["end"] for e in events)
        assert all(events[i]["end"] <= events[i+1]["start"] for i in range(len(events)-1))

    @pytest.mark.asyncio
    async def test_profiling_integration(self, setup_executor):
        """Test profiling integration"""
        with torch.profiler.profile(use_cuda=True) as prof:
            result = await setup_executor.execute_function(
                self.sample_function,
                {"x": torch.randn(1, 10)},
                {"gpu_id": 0, "enable_profiling": True}
            )
        
        # Check profiling data
        events = prof.key_averages()
        assert len(events) > 0
        assert any("cuda" in e.key for e in events)

    @pytest.mark.asyncio
    async def test_error_propagation(self, setup_executor):
        """Test error propagation from different components"""
        # Test CUDA errors
        with torch.cuda.device(0):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Should properly propagate CUDA errors
        error_code = """
        import torch
        def bad_cuda():
            x = torch.randn(1, device='cuda')
            x[999999] = 1  # Invalid index
        """
        
        with pytest.raises(RuntimeError) as exc:
            await setup_executor.execute_function(
                error_code,
                {},
                {"gpu_id": 0}
            )
        assert "CUDA error" in str(exc.value)




    @pytest.mark.asyncio
    async def test_model_versioning(self, setup_executor):
        """Test model version handling"""
        # Test different model versions
        model_versions = ['v1', 'v2']
        for version in model_versions:
            code = f"""
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 2)
            )
            model.version = '{version}'
            """
            result = await setup_executor.execute_function(
                code,
                {"x": torch.randn(1, 10)},
                {"gpu_id": 0}
            )
            assert result["model_info"]["version"] == version

    @pytest.mark.asyncio
    async def test_input_preprocessing(self, setup_executor):
        """Test input preprocessing capabilities"""
        code = """
        def preprocess(x):
            # Normalize
            x = (x - x.mean()) / x.std()
            # Add batch dimension if needed
            if x.dim() == 2:
                x = x.unsqueeze(0)
            return x
        
        def inference(x):
            x = preprocess(x)
            return model(x)
        """
        result = await setup_executor.execute_function(
            code,
            {"x": torch.randn(10, 10)},
            {"gpu_id": 0}
        )
        assert result["preprocessing_applied"] is True

    @pytest.mark.asyncio
    async def test_custom_metric_collection(self, setup_executor):
        """Test custom metrics collection during execution"""
        code = """
        def inference(x):
            start = time.time()
            result = model(x)
            custom_metrics = {
                'processing_time': time.time() - start,
                'input_shape': list(x.shape),
                'memory_peak': torch.cuda.max_memory_allocated()
            }
            return result, custom_metrics
        """
        result = await setup_executor.execute_function(
            code,
            {"x": torch.randn(1, 10)},
            {"gpu_id": 0}
        )
        assert "custom_metrics" in result