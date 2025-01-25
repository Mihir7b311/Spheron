# tests/execution_engine/batch_processor/test_inference_batch.py

import pytest
import torch
import asyncio
from src.execution_engine.batch_processor.inference_batch import InferenceBatch
from src.common.exceptions import BatchError

class TestInferenceBatch:
    @pytest.fixture
    def setup_batch(self):
        """Create inference batch instance"""
        return InferenceBatch(batch_size=32)

    @pytest.fixture
    def sample_model(self):
        """Create sample PyTorch model"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    @pytest.fixture
    def cuda_device(self):
        """Get CUDA device"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.mark.asyncio
    async def test_batch_initialization(self, setup_batch):
        """Test batch initialization"""
        batch = setup_batch
        assert batch.batch_size == 32
        assert len(batch.current_batch) == 0
        assert len(batch.batch_inputs) == 0

    @pytest.mark.asyncio
    async def test_add_request(self, setup_batch):
        """Test adding requests to batch"""
        batch = setup_batch
        input_tensor = torch.randn(1, 10)
        
        await batch.add_inference_request(
            model_input=input_tensor,
            request_id="test_1"
        )
        
        assert len(batch.current_batch) == 1
        assert len(batch.batch_inputs) == 1
        assert batch.current_batch[0]["request_id"] == "test_1"

    @pytest.mark.asyncio
    async def test_batch_size_limit(self, setup_batch):
        """Test batch size enforcement"""
        batch = setup_batch
        
        # Try to exceed batch size
        with pytest.raises(BatchError):
            for i in range(batch.batch_size + 1):
                await batch.add_inference_request(
                    model_input=torch.randn(1, 10),
                    request_id=f"test_{i}"
                )

    @pytest.mark.asyncio
    async def test_batch_execution(self, setup_batch, sample_model, cuda_device):
        """Test batch execution"""
        batch = setup_batch
        
        # Add requests
        for i in range(5):
            await batch.add_inference_request(
                model_input=torch.randn(1, 10),
                request_id=f"test_{i}"
            )
            
        results = await batch.execute_batch(sample_model, cuda_device)
        
        assert len(results) == 5
        assert all(isinstance(r["output"], torch.Tensor) for r in results)
        assert all(r["output"].size(1) == 2 for r in results)

    @pytest.mark.asyncio
    async def test_empty_batch(self, setup_batch, sample_model, cuda_device):
        """Test executing empty batch"""
        batch = setup_batch
        results = await batch.execute_batch(sample_model, cuda_device)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_batch_clearing(self, setup_batch, sample_model, cuda_device):
        """Test batch clearing after execution"""
        batch = setup_batch
        
        await batch.add_inference_request(
            model_input=torch.randn(1, 10),
            request_id="test_1"
        )
        
        await batch.execute_batch(sample_model, cuda_device)
        
        assert len(batch.current_batch) == 0
        assert len(batch.batch_inputs) == 0

    @pytest.mark.asyncio
    async def test_invalid_input(self, setup_batch):
        """Test handling invalid input"""
        batch = setup_batch
        
        with pytest.raises(BatchError):
            await batch.add_inference_request(
                model_input="invalid_input",  # Not a tensor
                request_id="test_1"
            )

    @pytest.mark.asyncio
    async def test_batch_performance(self, setup_batch, sample_model, cuda_device):
        """Test batch processing performance"""
        batch = setup_batch
        
        start_time = asyncio.get_event_loop().time()
        
        # Add full batch
        for i in range(batch.batch_size):
            await batch.add_inference_request(
                model_input=torch.randn(1, 10),
                request_id=f"test_{i}"
            )
            
        results = await batch.execute_batch(sample_model, cuda_device)
        end_time = asyncio.get_event_loop().time()
        
        assert len(results) == batch.batch_size
        assert end_time - start_time < 5.0  # Reasonable time limit

    @pytest.mark.asyncio
    async def test_mixed_batch_sizes(self, setup_batch, sample_model, cuda_device):
        """Test handling different input sizes in batch"""
        batch = setup_batch
        
        # Add inputs with different first dimensions
        sizes = [1, 2, 4]
        for i, size in enumerate(sizes):
            await batch.add_inference_request(
                model_input=torch.randn(size, 10),
                request_id=f"test_{i}"
            )
            
        results = await batch.execute_batch(sample_model, cuda_device)
        assert len(results) == len(sizes)
        for i, result in enumerate(results):
            assert result["output"].size(0) == sizes[i]

    @pytest.mark.asyncio
    async def test_memory_management(self, setup_batch, sample_model, cuda_device):
        """Test memory management during batching"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        batch = setup_batch
        initial_memory = torch.cuda.memory_allocated()
        
        # Process large inputs
        for i in range(10):
            await batch.add_inference_request(
                model_input=torch.randn(100, 10),
                request_id=f"test_{i}"
            )
            
        await batch.execute_batch(sample_model, cuda_device)
        
        # Force cleanup
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        # Should return to roughly initial state
        assert final_memory <= initial_memory * 1.1

    @pytest.mark.asyncio
    async def test_error_propagation(self, setup_batch, cuda_device):
        """Test error handling during execution"""
        batch = setup_batch
        
        # Create invalid model
        bad_model = torch.nn.Linear(5, 2)  # Wrong input size
        if torch.cuda.is_available():
            bad_model = bad_model.cuda()
            
        await batch.add_inference_request(
            model_input=torch.randn(1, 10),  # Size mismatch
            request_id="test_1"
        )
        
        with pytest.raises(RuntimeError):
            await batch.execute_batch(bad_model, cuda_device)

    @pytest.mark.asyncio
    async def test_device_handling(self, setup_batch, sample_model):
        """Test handling different devices"""
        batch = setup_batch
        
        # Test CPU
        await batch.add_inference_request(
            model_input=torch.randn(1, 10),
            request_id="cpu_test"
        )
        results_cpu = await batch.execute_batch(
            sample_model.cpu(),
            torch.device('cpu')
        )
        assert results_cpu[0]["output"].device.type == 'cpu'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            await batch.add_inference_request(
                model_input=torch.randn(1, 10),
                request_id="gpu_test"
            )
            results_gpu = await batch.execute_batch(
                sample_model.cuda(),
                torch.device('cuda')
            )
            assert results_gpu[0]["output"].device.type == 'cuda'

    @pytest.mark.asyncio
    async def test_concurrent_batches(self, setup_batch, sample_model, cuda_device):
        """Test processing multiple batches concurrently"""
        async def process_batch(batch_id):
            batch = InferenceBatch(batch_size=32)
            for i in range(5):
                await batch.add_inference_request(
                    model_input=torch.randn(1, 10),
                    request_id=f"batch_{batch_id}_test_{i}"
                )
            return await batch.execute_batch(sample_model, cuda_device)
            
        # Process multiple batches
        batches = [process_batch(i) for i in range(3)]
        results = await asyncio.gather(*batches)
        
        assert len(results) == 3
        assert all(len(r) == 5 for r in results)

    @pytest.mark.asyncio
    async def test_variable_batch_sizes(self, setup_batch, sample_model, cuda_device):
        """Test different batch size configurations"""
        for batch_size in [1, 8, 16, 32]:
            batch = InferenceBatch(batch_size=batch_size)
            
            # Fill to capacity
            for i in range(batch_size):
                await batch.add_inference_request(
                    model_input=torch.randn(1, 10),
                    request_id=f"test_{i}"
                )
                
            results = await batch.execute_batch(sample_model, cuda_device)
            assert len(results) == batch_size

    @pytest.mark.asyncio
    async def test_output_shapes(self, setup_batch, cuda_device):
        """Test handling different output shapes"""
        batch = setup_batch
        
        # Model with different output shapes
        class VariableOutputModel(torch.nn.Module):
            def forward(self, x):
                return {
                    'output1': x @ torch.randn(10, 2),
                    'output2': x @ torch.randn(10, 3)
                }
                
        model = VariableOutputModel()
        if torch.cuda.is_available():
            model = model.cuda()
            
        await batch.add_inference_request(
            model_input=torch.randn(1, 10),
            request_id="test_1"
        )
        
        results = await batch.execute_batch(model, cuda_device)
        assert "output1" in results[0]["output"]
        assert "output2" in results[0]["output"]
        assert results[0]["output"]["output1"].size(1) == 2
        assert results[0]["output"]["output2"].size(1) == 3

    @pytest.mark.asyncio
    async def test_state_management(self, setup_batch, cuda_device):
        """Test handling stateful models"""
        batch = setup_batch
        
        class StatefulModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0
                self.linear = torch.nn.Linear(10, 2)
                
            def forward(self, x):
                self.counter += 1
                return self.linear(x)
                
        model = StatefulModel()
        if torch.cuda.is_available():
            model = model.cuda()
            
        # Run multiple batches
        for _ in range(3):
            await batch.add_inference_request(
                model_input=torch.randn(1, 10),
                request_id=f"test_{_}"
            )
            await batch.execute_batch(model, cuda_device)
            
        assert model.counter == 3



    @pytest.mark.asyncio
    async def test_auto_padding(self, setup_batch):
        """Test automatic padding for different input sizes"""
        batch = setup_batch
        
        # Add inputs with different sizes
        sizes = [(1, 10), (1, 15), (1, 20)]
        for i, size in enumerate(sizes):
            await batch.add_inference_request(
                model_input=torch.randn(*size),
                request_id=f"test_{i}"
            )
        
        # Verify padding
        padded_inputs = batch._prepare_batch_inputs()
        max_size = max(s[1] for s in sizes)
        assert all(input.size(1) == max_size for input in padded_inputs)

    @pytest.mark.asyncio
    async def test_batch_optimization_strategies(self, setup_batch):
        """Test different batch optimization strategies"""
        batch = setup_batch
        
        # Test different strategies
        strategies = ['dynamic', 'static', 'adaptive']
        for strategy in strategies:
            batch.optimization_strategy = strategy
            optimal_size = await batch._compute_optimal_batch_size()
            assert optimal_size > 0

if __name__ == "__main__":
    pytest.main(["-v", "test_inference_batch.py"])