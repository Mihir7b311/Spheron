from unittest.mock import AsyncMock
import pytest
import torch
import asyncio
from src.runtime.python_executor import PythonExecutor
from src.cuda.context_manager import CUDAContextManager
from src.batch_processor.batch_manager import BatchManager


# Mocking the StreamManager with AsyncMock
class MockStreamManager:
    async def create_stream(self, gpu_id: int, stream_name: str):
        # Mocking the return value to simulate a valid StreamInfo object
        mock_stream_info = AsyncMock()
        mock_stream_info.stream = torch.cuda.Stream(device=gpu_id)
        return mock_stream_info


@pytest.fixture
async def setup_executor():
    context_manager = CUDAContextManager()
    
    # Initialize the MockStreamManager
    mock_stream_manager = MockStreamManager()
    
    # Create BatchManager with the mocked stream manager
    batch_manager = BatchManager(mock_stream_manager)
    
    # Create PythonExecutor with real context manager and mocked batch manager
    executor = PythonExecutor(context_manager, batch_manager)
    
    return executor


@pytest.mark.asyncio
async def test_function_execution(setup_executor):
    executor = setup_executor
    
    # Test function code to be executed
    function_code = """
    def inference(x):
        return x * 2
    """
    
    # Test input data
    input_data = {"x": torch.tensor([1.0])}
    
    # GPU context for testing
    gpu_context = {"gpu_id": 0}
    
    # Execute the function using the executor
    result = await executor.execute_function(function_code, input_data, gpu_context)
    
    # Assertions to check the results
    assert "results" in result
    assert "metrics" in result
    assert len(result["results"]) == len(input_data["x"])  # Check batch size matches input size
    assert isinstance(result["results"][0]["output"], torch.Tensor)  # Ensure output is a tensor
