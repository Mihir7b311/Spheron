# src/batch_processor/inference_batch.py

from typing import List, Dict, Any
import torch
import numpy as np
import logging
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
                
            # Create input batch tensor
            input_batch = torch.stack(self.batch_inputs).to(device)
            
            # Execute inference
            with torch.no_grad():
                output_batch = model(input_batch)
                
            # Process results
            results = []
            for idx, request in enumerate(self.current_batch):
                results.append({
                    "request_id": request["request_id"],
                    "output": output_batch[idx].cpu()
                })
                
            # Clear batch
            self.current_batch = []
            self.batch_inputs = []
            
            return results
            
        except Exception as e:
            raise BatchError(f"Batch inference failed: {e}")