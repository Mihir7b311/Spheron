# src/execution_engine/batch_processor/inference_batch.py

import torch
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..common.exceptions import BatchError

@dataclass
class BatchMetrics:
    """Metrics for batch execution"""
    batch_size: int
    processing_time: float
    memory_used: int
    success_count: int
    error_count: int

class InferenceBatch:
    def __init__(self, 
                 batch_size: int = 32,
                 timeout: float = 1.0,
                 max_retries: int = 3):
        """Initialize inference batch processor
        Args:
            batch_size: Maximum batch size
            timeout: Batch processing timeout
            max_retries: Maximum retries for failed requests
        """
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Batch state
        self.current_batch = []
        self.batch_inputs = []
        self.batch_metrics = []
        self.retry_queue = []
        self.is_processing = False

    async def add_inference_request(self, 
                                  model_input: torch.Tensor,
                                  request_id: str,
                                  priority: int = 0) -> bool:
        """Add request to current batch
        Args:
            model_input: Input tensor
            request_id: Unique request identifier
            priority: Request priority (higher = more important)
        Returns:
            bool: Success status
        """
        try:
            if len(self.current_batch) >= self.batch_size:
                return False

            if not isinstance(model_input, torch.Tensor):
                raise BatchError("Input must be a torch.Tensor")

            # Validate input shape
            if not model_input.dim() > 0:
                raise BatchError("Input tensor cannot be scalar")

            request = {
                "id": request_id,
                "input": model_input,
                "priority": priority,
                "retry_count": 0,
                "added_time": asyncio.get_event_loop().time()
            }

            self.current_batch.append(request)
            self.batch_inputs.append(model_input)

            logging.debug(f"Added request {request_id} to batch. Current size: {len(self.current_batch)}")
            return True

        except Exception as e:
            logging.error(f"Failed to add request {request_id}: {e}")
            raise BatchError(f"Failed to add request: {e}")

    async def execute_batch(self, 
                          model: torch.nn.Module,
                          device: torch.device,
                          stream: Optional[torch.cuda.Stream] = None) -> tuple[List[Dict], BatchMetrics]:
        """Execute inference on batched inputs
        Args:
            model: PyTorch model
            device: CUDA device
            stream: CUDA stream (optional)
        Returns:
            Tuple of (results list, batch metrics)
        """
        if not self.current_batch:
            return [], BatchMetrics(0, 0.0, 0, 0, 0)

        start_time = asyncio.get_event_loop().time()
        results = []
        success_count = 0
        error_count = 0

        try:
            self.is_processing = True

            # Prepare input batch
            input_batch = torch.stack(self.batch_inputs).to(device)
            initial_memory = torch.cuda.memory_allocated(device)

            # Execute inference
            with torch.cuda.stream(stream) if stream else torch.cuda.device(device):
                with torch.no_grad():
                    try:
                        output_batch = model(input_batch)
                        stream.synchronize() if stream else torch.cuda.current_stream().synchronize()
                    except Exception as e:
                        logging.error(f"Batch inference failed: {e}")
                        raise BatchError(f"Inference failed: {e}")

            # Process results
            for idx, request in enumerate(self.current_batch):
                try:
                    results.append({
                        "request_id": request["id"],
                        "output": output_batch[idx].cpu(),  # Move to CPU
                        "status": "success",
                        "processing_time": asyncio.get_event_loop().time() - request["added_time"]
                    })
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    results.append({
                        "request_id": request["id"],
                        "error": str(e),
                        "status": "error"
                    })
                    # Add to retry queue if retries remaining
                    if request["retry_count"] < self.max_retries:
                        request["retry_count"] += 1
                        self.retry_queue.append(request)

            # Calculate metrics
            end_time = asyncio.get_event_loop().time()
            memory_used = torch.cuda.memory_allocated(device) - initial_memory
            metrics = BatchMetrics(
                batch_size=len(self.current_batch),
                processing_time=end_time - start_time,
                memory_used=memory_used,
                success_count=success_count,
                error_count=error_count
            )

            # Clear current batch
            self._clear_batch()
            return results, metrics

        except Exception as e:
            logging.error(f"Batch execution failed: {e}")
            raise BatchError(f"Batch execution failed: {e}")
        finally:
            self.is_processing = False
            # Handle retry queue
            await self._process_retries()

    async def _process_retries(self):
        """Process retry queue"""
        if self.retry_queue:
            retry_requests = self.retry_queue.copy()
            self.retry_queue.clear()
            for request in retry_requests:
                await self.add_inference_request(
                    request["input"],
                    request["id"],
                    request["priority"]
                )

    def _clear_batch(self):
        """Clear current batch"""
        self.current_batch.clear()
        self.batch_inputs.clear()

    def get_batch_size(self) -> int:
        """Get current batch size"""
        return len(self.current_batch)

    def get_state(self) -> Dict:
        """Get current state"""
        return {
            "current_batch_size": len(self.current_batch),
            "retry_queue_size": len(self.retry_queue),
            "is_processing": self.is_processing
        }

    async def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for batch processing completion
        Args:
            timeout: Wait timeout in seconds
        Returns:
            bool: Completion status
        """
        timeout = timeout or self.timeout
        start_time = asyncio.get_event_loop().time()

        while self.is_processing:
            if asyncio.get_event_loop().time() - start_time > timeout:
                return False
            await asyncio.sleep(0.01)

        return True