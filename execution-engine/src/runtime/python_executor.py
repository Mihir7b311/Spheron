# src/runtime/python_executor.py

import torch
import asyncio
from typing import Dict, Any
import logging
from ..common.exceptions import RuntimeError
from ..common.metrics import ExecutionMetrics

class PythonExecutor:
    def __init__(self, cuda_context_manager, batch_manager):
        self.cuda_context_manager = cuda_context_manager
        self.batch_manager = batch_manager
        self.active_executions = {}
        
    async def execute_function(self, 
                             function_code: str,
                             input_data: Dict[str, Any],
                             gpu_context: Dict) -> Dict:
        """Execute Python function with GPU context"""
        try:
            # Setup execution
            context = await self.cuda_context_manager.get_context(
                gpu_context['gpu_id']
            )
            
            start_time = asyncio.get_event_loop().time()
            
            # Add to batch
            await self.batch_manager.add_request({
                "function": function_code,
                "input": input_data,
                "context": context
            })
            
            # Process batch
            results = await self.batch_manager.process_batch(
                gpu_context['gpu_id']
            )
            
            end_time = asyncio.get_event_loop().time()
            
            # Collect metrics
            metrics = ExecutionMetrics(
                start_time=start_time,
                end_time=end_time,
                cuda_time=0.0,  # Set by batch processor
                batch_size=len(results),
                memory_used=0,  # Set by memory tracker
                compute_used=0.0  # Set by resource monitor
            )
            
            return {
                "results": results,
                "metrics": metrics.to_dict()
            }
            
        except Exception as e:
            raise RuntimeError(f"Function execution failed: {e}")