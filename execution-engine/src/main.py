# src/main.py

import asyncio
import logging
from typing import Dict, Any
from runtime.python_executor import PythonExecutor
from cuda.context_manager import CUDAContextManager
from batch_processor.batch_manager import BatchManager
from cuda.stream_manager import CUDAStreamManager
from common.performance_monitor import PerformanceMonitor

class ExecutionController:
    def __init__(self):
        self.context_manager = CUDAContextManager()
        self.stream_manager = CUDAStreamManager()
        self.batch_manager = BatchManager(self.stream_manager)
        self.executor = PythonExecutor(
            self.context_manager,
            self.batch_manager
        )
        self.performance_monitor = PerformanceMonitor()
        
    async def execute_model(self, 
                          model_code: str,
                          input_data: Dict[str, Any],
                          gpu_context: Dict) -> Dict:
        """Execute ML model"""
        try:
            # Execute function
            result = await self.executor.execute_function(
                model_code,
                input_data,
                gpu_context
            )
            
            # Record performance
            await self.performance_monitor.record_metrics(
                PerformanceMetrics(**result["metrics"])
            )
            
            return {
                "status": "success",
                "results": result["results"],
                "metrics": result["metrics"]
            }
            
        except Exception as e:
            logging.error(f"Execution failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    async def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return self.performance_monitor.get_statistics()

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    controller = ExecutionController()