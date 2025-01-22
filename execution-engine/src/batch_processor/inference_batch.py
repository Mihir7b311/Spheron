# src/batch_processor/inference_batch.py

import torch
import logging
from typing import List, Dict, Any
from ..common.exceptions import BatchError

class InferenceBatch:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.current_batch = []
        self.batch_inputs = []
        
    async def add_inference_request(self, 
                                  model_input: torch.Tensor,
                                  request_id: str):
        """Add request to current batch"""
        if len(self.current_batch) >= self.batch_size:
            raise BatchError("Batch is full")
            
        if not isinstance(model_input, torch.Tensor):
            raise BatchError("Input must be a torch.Tensor")
            
        self.current_batch.append({
            "request_id": request_id,
            "input": model_input
        })
        self.batch_inputs.append(model_input)
        
    async def execute_batch(self, 
                          model: torch.nn.Module,
                          device: torch.device) -> List[Dict]:
        """Execute inference on batched inputs"""
        try:
            if not self.current_batch:
                return []
                
            # Create input batch tensor and move to device
            input_batch = torch.stack(self.batch_inputs)
            input_batch = input_batch.to(device)  # Ensure input is on correct device
            
            # Execute inference
            with torch.no_grad():
                output_batch = model(input_batch)
                # Ensure output stays on same device
                output_batch = output_batch.to(device)
                
            # Process results
            results = []
            for idx, request in enumerate(self.current_batch):
                # Keep output on the same device as model
                results.append({
                    "request_id": request["request_id"],
                    "output": output_batch[idx]  # Don't move to CPU
                })
                
            # Clear batch
            self.current_batch = []
            self.batch_inputs = []
            
            return results
            
        except Exception as e:
            logging.error(f"Batch inference failed: {e}")
            raise BatchError(f"Batch inference failed: {e}")