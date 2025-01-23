# src/execution_engine/runtime/python_executor.py

import torch
import asyncio
import logging
from typing import Dict, Any, Optional
from ..common.exceptions import RuntimeError
from ..common.metrics import ExecutionMetrics

class PythonExecutor:
    def __init__(self, cuda_context_manager, batch_manager, config: Dict = None):
        """Initialize Python Executor
        Args:
            cuda_context_manager: CUDA context manager instance
            batch_manager: Batch manager instance
            config: Configuration dictionary
        """
        self.cuda_context_manager = cuda_context_manager
        self.batch_manager = batch_manager
        self.config = config or {}
        self.active_executions: Dict[str, Dict] = {}
        self.execution_timeout = self.config.get('execution_timeout', 30.0)
        
    async def execute_function(self, 
                             function_code: str,
                             input_data: Dict[str, Any],
                             gpu_context: Dict) -> Dict:
        """Execute Python function with GPU context
        Args:
            function_code: Python code to execute
            input_data: Input data for function
            gpu_context: GPU context information
        Returns:
            Dictionary containing results and metrics
        """
        execution_id = f"exec_{id(function_code)}_{asyncio.get_event_loop().time()}"
        
        try:
            # Get or create CUDA context
            context = await self.cuda_context_manager.get_context(
                gpu_context['gpu_id']
            )
            
            # Record start time
            start_time = asyncio.get_event_loop().time()
            
            # Track execution
            self.active_executions[execution_id] = {
                "start_time": start_time,
                "gpu_id": gpu_context['gpu_id'],
                "status": "running"
            }
            
            # Prepare request
            request = {
                "id": execution_id,
                "function": function_code,
                "input": input_data,
                "context": context
            }
            
            # Add to batch
            await self.batch_manager.add_request(
                request=request,
                gpu_id=gpu_context['gpu_id']
            )
            
            # Wait for processing with timeout
            results = await self._wait_for_results(
                gpu_id=gpu_context['gpu_id'],
                timeout=self.execution_timeout
            )
            
            # Record end time
            end_time = asyncio.get_event_loop().time()
            
            # Update execution status
            self.active_executions[execution_id]["status"] = "completed"
            
            # Collect metrics
            metrics = ExecutionMetrics(
                execution_id=execution_id,
                start_time=start_time,
                end_time=end_time,
                gpu_id=gpu_context['gpu_id'],
                batch_size=len(results) if results else 0,
                status="success"
            )
            
            return {
                "execution_id": execution_id,
                "results": results,
                "metrics": metrics.to_dict()
            }
            
        except Exception as e:
            logging.error(f"Function execution failed: {e}")
            if execution_id in self.active_executions:
                self.active_executions[execution_id]["status"] = "failed"
                self.active_executions[execution_id]["error"] = str(e)
            raise RuntimeError(f"Function execution failed: {e}")
        
    async def _wait_for_results(self, gpu_id: int, timeout: float) -> Optional[List[Dict]]:
        """Wait for batch processing results
        Args:
            gpu_id: GPU ID to wait for
            timeout: Timeout in seconds
        Returns:
            List of results if successful
        """
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            # Check if batch needs processing
            if self.batch_manager.get_queue_size(gpu_id) > 0:
                results = await self.batch_manager.process_batch(gpu_id)
                if results:
                    return results
                    
            await asyncio.sleep(0.01)  # Small delay
            
        raise RuntimeError(f"Execution timeout after {timeout}s")
        
    def get_active_executions(self) -> Dict:
        """Get information about active executions"""
        return self.active_executions.copy()
        
    def get_execution_status(self, execution_id: str) -> Optional[Dict]:
        """Get status of specific execution"""
        return self.active_executions.get(execution_id)