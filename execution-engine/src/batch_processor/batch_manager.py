# src/batch_processor/batch_manager.py

from typing import List, Dict, Any
import torch
import logging
from ..common.exceptions import BatchError

class BatchManager:
    def __init__(self, cuda_stream_manager):
        self.stream_manager = cuda_stream_manager
        self.max_batch_size = 32
        self.waiting_requests = []
        
    async def add_request(self, request: Dict[str, Any]):
        """Add request to batch queue"""
        self.waiting_requests.append(request)
        
    async def process_batch(self, gpu_id: int) -> List[Dict]:
        """Process a batch of requests"""
        try:
            if not self.waiting_requests:
                return []
                
            # Create batch
            batch = self.waiting_requests[:self.max_batch_size]
            self.waiting_requests = self.waiting_requests[self.max_batch_size:]
            
            # Get CUDA stream
            stream = await self.stream_manager.create_stream(
                gpu_id, f"batch_{len(batch)}"
            )
            
            with torch.cuda.stream(stream):
                # Process batch
                results = await self._process_requests(batch, gpu_id)
                
            return results
            
        except Exception as e:
            raise BatchError(f"Batch processing failed: {e}")