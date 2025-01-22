import pytest
import torch
import asyncio
from src.batch_processor.inference_batch import InferenceBatch
from src.common.exceptions import BatchError

class TestInferenceBatch:
    @pytest.fixture
    def setup_batch(self):
        """Setup test batch processor"""
        return InferenceBatch(batch_size=32)

    @pytest.fixture
    def sample_model(self):
        """Create a simple test model"""
        return torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )

    @pytest.mark.asyncio
    async def test_batch_initialization(self, setup_batch):
        """Test batch initialization"""
        batch = setup_batch  # No need to await setup_batch
        assert batch.batch_size == 32
        assert len(batch.current_batch) == 0
        assert len(batch.batch_inputs) == 0

    @pytest.mark.asyncio
    async def test_add_request(self, setup_batch):
        """Test adding requests to batch"""
        batch = setup_batch  # No need to await setup_batch
        
        # Create test input
        input_tensor = torch.randn(1, 10)
        
        await batch.add_inference_request(
            model_input=input_tensor,
            request_id="test_1"
        )
        
        assert len(batch.current_batch) == 1
        assert len(batch.batch_inputs) == 1
        assert batch.current_batch[0]["request_id"] == "test_1"

    @pytest.mark.asyncio
    async def test_batch_limit(self, setup_batch):
        """Test batch size limit"""
        batch = setup_batch  # No need to await setup_batch
        
        # Try to add more than batch_size requests
        with pytest.raises(BatchError):
            for i in range(33):  # One more than batch_size
                await batch.add_inference_request(
                    model_input=torch.randn(1, 10),
                    request_id=f"test_{i}"
                )

    @pytest.mark.asyncio
    async def test_execute_batch(self, setup_batch, sample_model):
        """Test batch execution"""
        batch = setup_batch  # No need to await setup_batch
        
        # Add multiple requests
        for i in range(5):
            await batch.add_inference_request(
                model_input=torch.randn(1, 10),
                request_id=f"test_{i}"
            )
        
        # Execute batch
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
            
        sample_model.to(device)
        results = await batch.execute_batch(sample_model, device)
        
        assert len(results) == 5
        assert all(isinstance(r["output"], torch.Tensor) for r in results)
        assert all(r["output"].size(1) == 2 for r in results)  # Output size from model

    @pytest.mark.asyncio
    async def test_empty_batch(self, setup_batch, sample_model):
        """Test executing empty batch"""
        batch = setup_batch  # No need to await setup_batch
        device = torch.device('cpu')
        
        results = await batch.execute_batch(sample_model, device)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_batch_clearing(self, setup_batch, sample_model):
        """Test batch clearing after execution"""
        batch = setup_batch  # No need to await setup_batch
        device = torch.device('cpu')
        
        # Add requests
        await batch.add_inference_request(
            model_input=torch.randn(1, 10),
            request_id="test_1"
        )
        
        # Execute batch
        await batch.execute_batch(sample_model, device)
        
        # Check if batch is cleared
        assert len(batch.current_batch) == 0
        assert len(batch.batch_inputs) == 0

    @pytest.mark.asyncio
    async def test_invalid_input(self, setup_batch):
        """Test handling invalid input"""
        batch = setup_batch  # No need to await setup_batch
        
        with pytest.raises(BatchError):
            await batch.add_inference_request(
                model_input="invalid_input",  # Not a tensor
                request_id="test_1"
            )

    @pytest.mark.asyncio
    async def test_batch_performance(self, setup_batch, sample_model):
        """Test batch processing performance"""
        batch = setup_batch  # No need to await setup_batch
        device = torch.device('cpu')
        
        # Add full batch of requests
        start_time = asyncio.get_event_loop().time()
        
        for i in range(32):
            await batch.add_inference_request(
                model_input=torch.randn(1, 10),
                request_id=f"test_{i}"
            )
            
        results = await batch.execute_batch(sample_model, device)
        end_time = asyncio.get_event_loop().time()
        
        assert len(results) == 32
        assert end_time - start_time < 5.0  # Reasonable time limit

    @pytest.mark.asyncio
    async def test_request_identification(self, setup_batch, sample_model):
        """Test correct request ID mapping"""
        batch = setup_batch  # No need to await setup_batch
        device = torch.device('cpu')
        
        # Add requests with specific IDs
        test_ids = ["A", "B", "C"]
        for request_id in test_ids:
            await batch.add_inference_request(
                model_input=torch.randn(1, 10),
                request_id=request_id
            )
            
        results = await batch.execute_batch(sample_model, device)
        result_ids = [r["request_id"] for r in results]
        
        assert all(tid in result_ids for tid in test_ids)
        assert len(results) == len(test_ids)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.asyncio
    async def test_gpu_execution(self, setup_batch, sample_model):
        """Test batch execution on GPU"""
        batch = setup_batch  # No need to await setup_batch
        device = torch.device('cuda:0')
        sample_model.to(device)
        

        
        # Add requests and ensure input tensors are moved to the GPU
        gpu_input = torch.randn(1, 10, device=device)
        await batch.add_inference_request(
            model_input=gpu_input,
            request_id="gpu_test"
        )
        
        # Execute batch
        results = await batch.execute_batch(sample_model, device)

        # Assertions
        assert len(results) == 1
        assert results[0]["output"].device.type == 'cuda', (
            f"Expected CUDA output, got {results[0]['output'].device.type}"
        )
