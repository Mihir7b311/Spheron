# src/execution_engine/batch_processor/batch_manager.py

import torch
import logging
import asyncio
from typing import List, Dict, Any, Optional
from ..common.exceptions import BatchError
from ..cuda.stream_manager import StreamInfo

class BatchManager:
    def __init__(self, cuda_stream_manager, config: Dict = None):
        """Initialize Batch Manager
        Args:
            cuda_stream_manager: CUDA stream manager instance
            config: Configuration dictionary
        """
        self.stream_manager = cuda_stream_manager
        self.config = config or {}
        self.max_batch_size = self.config.get('max_batch_size', 32)
        self.wait_timeout = self.config.get('wait_timeout', 0.1)
        self.batch_queues: Dict[int, List] = {}  # GPU-specific queues
        self.processing_batches: Dict[int, bool] = {}  # Track processing status
        
    async def add_request(self, request: Dict[str, Any], gpu_id: int):
        """Add request to batch queue
        Args:
            request: Request dictionary containing function and input
            gpu_id: Target GPU ID
        """
        try:
            if gpu_id not in self.batch_queues:
                self.batch_queues[gpu_id] = []
                
            self.batch_queues[gpu_id].append(request)
            logging.info(f"Added request to GPU {gpu_id} queue. Queue size: {len(self.batch_queues[gpu_id])}")
            
            # Trigger batch processing if queue reaches max size
            if len(self.batch_queues[gpu_id]) >= self.max_batch_size:
                await self.process_batch(gpu_id)
                
        except Exception as e:
            logging.error(f"Failed to add request to batch: {e}")
            raise BatchError(f"Failed to add request: {e}")
        
    async def process_batch(self, gpu_id: int) -> List[Dict]:
        """Process a batch of requests for given GPU
        Args:
            gpu_id: GPU ID to process batch for
        Returns:
            List of processed results
        """
        if self.processing_batches.get(gpu_id):
            logging.warning(f"Batch processing already in progress for GPU {gpu_id}")
            return []
            
        try:
            self.processing_batches[gpu_id] = True
            
            if not self.batch_queues.get(gpu_id):
                return []
                
            # Get batch of requests
            batch = self.batch_queues[gpu_id][:self.max_batch_size]
            self.batch_queues[gpu_id] = self.batch_queues[gpu_id][self.max_batch_size:]
            
            # Create CUDA stream
            stream_info = await self.stream_manager.create_stream(
                gpu_id=gpu_id,
                stream_id=f"batch_{gpu_id}_{id(batch)}"
            )
            
            # Process batch
            results = await self._execute_batch(batch, stream_info)
            
            # Cleanup
            await self.stream_manager.release_stream(stream_info.stream_id)
            
            return results
            
        except Exception as e:
            logging.error(f"Batch processing failed: {e}")
            raise BatchError(f"Batch processing failed: {e}")
        finally:
            self.processing_batches[gpu_id] = False
            
    async def _execute_batch(self, batch: List[Dict], stream_info: StreamInfo) -> List[Dict]:
        """Execute batch of requests using given stream
        Args:
            batch: List of request dictionaries
            stream_info: CUDA stream information
        Returns:
            List of results
        """
        results = []
        device = stream_info.device
        
        try:
            with torch.cuda.stream(stream_info.stream):
                for request in batch:
                    try:
                        # Move input to GPU
                        input_data = request["input"]
                        if isinstance(input_data, torch.Tensor):
                            input_data = input_data.to(device)
                            
                        # Execute function
                        if callable(request["function"]):
                            output = request["function"](input_data)
                        else:
                            # Handle string function code
                            exec(request["function"])
                            output = locals()["model"](input_data)
                            
                        results.append({
                            "request_id": request.get("id", id(request)),
                            "output": output,
                            "status": "success"
                        })
                        
                    except Exception as e:
                        logging.error(f"Request execution failed: {e}")
                        results.append({
                            "request_id": request.get("id", id(request)),
                            "error": str(e),
                            "status": "failed"
                        })
                        
            # Ensure all operations are complete
            stream_info.stream.synchronize()
            return results
            
        except Exception as e:
            logging.error(f"Batch execution failed: {e}")
            raise BatchError(f"Batch execution failed: {e}")
            
    def get_queue_size(self, gpu_id: int) -> int:
        """Get current queue size for GPU"""
        return len(self.batch_queues.get(gpu_id, []))
        
    def get_stats(self) -> Dict:
        """Get batch processing statistics"""
        return {
            "queue_sizes": {
                gpu_id: len(queue) 
                for gpu_id, queue in self.batch_queues.items()
            },
            "processing_status": self.processing_batches.copy()
        }