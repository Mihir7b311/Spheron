# tests/integration/test_execution_pipeline.py

import pytest
import torch
import asyncio
from src.runtime.python_executor import PythonExecutor
from src.cuda.context_manager import CUDAContextManager
from src.batch_processor.batch_manager import BatchManager
from src.cuda.stream_manager import CUDAStreamManager

class TestExecutionPipeline:
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.context_manager = CUDAContextManager()
        self.stream_manager = CUDAStreamManager()
        self.batch_manager = BatchManager(self.stream_manager)
        self.executor = PythonExecutor(
            self.context_manager,
            self.batch_manager
        )

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        # Test ML model execution
        model_code = """
        import torch
        
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 2)
                
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        """
        
        input_tensor = torch.randn(1, 10)
        gpu_context = {"gpu_id": 0}
        
        result = await self.executor.execute_function(
            model_code,
            {"input": input_tensor},
            gpu_context
        )
        
        assert result["results"] is not None
        assert len(result["metrics"]) > 0