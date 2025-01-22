# src/cuda/memory_manager.py

import torch
import logging
from typing import Dict, Optional
from ..common.exceptions import CUDAError

class CUDAMemoryManager:
    def __init__(self, config: Dict[str, Any]):
        self.max_memory_fraction = config.get("max_memory_fraction", 0.8)
        self.reserved_memory = {}
        self.allocations = {}
        
    async def reserve_memory(self, 
                           gpu_id: int,
                           size_bytes: int) -> bool:
        """Reserve GPU memory"""
        try:
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            available = total_memory * self.max_memory_fraction
            
            current_reserved = self.reserved_memory.get(gpu_id, 0)
            if current_reserved + size_bytes > available:
                return False
                
            self.reserved_memory[gpu_id] = current_reserved + size_bytes
            return True
            
        except Exception as e:
            raise CUDAError(f"Memory reservation failed: {e}")
            
    async def allocate_buffer(self,
                            gpu_id: int,
                            size_bytes: int) -> Optional[torch.Tensor]:
        """Allocate CUDA memory buffer"""
        try:
            if not await self.reserve_memory(gpu_id, size_bytes):
                return None
                
            buffer = torch.cuda.ByteTensor(size_bytes)
            allocation_id = id(buffer)
            
            self.allocations[allocation_id] = {
                "gpu_id": gpu_id,
                "size": size_bytes,
                "buffer": buffer
            }
            
            return buffer
            
        except Exception as e:
            raise CUDAError(f"Buffer allocation failed: {e}")
            
    async def free_buffer(self, buffer: torch.Tensor):
        """Free allocated buffer"""
        allocation_id = id(buffer)
        if allocation_id in self.allocations:
            allocation = self.allocations.pop(allocation_id)
            gpu_id = allocation["gpu_id"]
            size = allocation["size"]
            
            self.reserved_memory[gpu_id] -= size
            del buffer
            
    def get_memory_info(self, gpu_id: int) -> Dict[str, int]:
        """Get memory usage information"""
        props = torch.cuda.get_device_properties(gpu_id)
        return {
            "total": props.total_memory,
            "reserved": self.reserved_memory.get(gpu_id, 0),
            "allocations": len([
                a for a in self.allocations.values()
                if a["gpu_id"] == gpu_id
            ])
        }