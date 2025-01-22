# tests/runtime/test_executor.py

import pytest
import torch
import asyncio
from src.runtime.python_executor import PythonExecutor
from src.cuda.context_manager import CUDAContextManager
from src.batch_processor.batch_manager import BatchManager

@pytest.fixture
async def setup_executor():
    context_manager = CUDAContextManager()
    batch_manager = BatchManager(None)  # Mock stream manager
    executor = PythonExecutor(context_manager, batch_manager)
    return executor

@pytest.mark.asyncio
async def test_function_execution(setup_executor):
    executor = setup_executor
    
    # Test function
    function_code = """
    def inference(x):
        return x * 2
    """
    
    input_data = {"x": torch.tensor([1.0, 2.0, 3.0])}
    gpu_context = {"gpu_id": 0}
    
    result = await executor.execute_function(
        function_code, input_data, gpu_context
    )
    
    assert "results" in result
    assert "metrics" in result
