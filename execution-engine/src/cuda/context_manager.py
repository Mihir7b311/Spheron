# src/cuda/context_manager.py

import torch
import logging
from typing import Dict, Optional
from ..common.exceptions import CUDAError

class CUDAContextManager:
    def __init__(self):
        self.contexts: Dict[int, torch.cuda.Stream] = {}
        self.active_context: Optional[int] = None
        
    async def create_context(self, gpu_id: int) -> Dict:
        """Create CUDA context for GPU"""
        try:
            torch.cuda.set_device(gpu_id)
            stream = torch.cuda.Stream(device=gpu_id)
            self.contexts[gpu_id] = stream
            
            return {
                "gpu_id": gpu_id,
                "stream": stream,
                "device": torch.device(f"cuda:{gpu_id}")
            }
        except Exception as e:
            raise CUDAError(f"Failed to create CUDA context: {e}")

    async def get_context(self, gpu_id: int):
        """Get or create CUDA context"""
        if gpu_id not in self.contexts:
            return await self.create_context(gpu_id)
        return {
            "gpu_id": gpu_id,
            "stream": self.contexts[gpu_id],
            "device": torch.device(f"cuda:{gpu_id}")
        }

# src/cuda/stream_manager.py

import torch
from typing import Dict, Optional
from ..common.exceptions import CUDAError

class CUDAStreamManager:
    def __init__(self):
        self.streams: Dict[int, Dict[str, torch.cuda.Stream]] = {}
        
    async def create_stream(self, gpu_id: int, stream_id: str) -> torch.cuda.Stream:
        """Create CUDA stream for processing"""
        try:
            if gpu_id not in self.streams:
                self.streams[gpu_id] = {}
                
            stream = torch.cuda.Stream(device=gpu_id)
            self.streams[gpu_id][stream_id] = stream
            return stream
            
        except Exception as e:
            raise CUDAError(f"Failed to create CUDA stream: {e}")