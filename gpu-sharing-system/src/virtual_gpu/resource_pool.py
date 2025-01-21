# src/virtual_gpu/resource_pool.py

from typing import Dict, Optional, List
import pynvml
import logging
import uuid
from dataclasses import dataclass
from .allocation import Allocation
from ..common.exceptions import ResourceError

class ResourcePool:
    def __init__(self, gpu_id: int, total_memory: int, handle: pynvml.c_nvmlDevice_t):
        """Initialize resource pool for a GPU"""
        self.gpu_id = gpu_id
        self.total_memory = total_memory
        self.handle = handle
        self.used_memory = 0
        self.allocations: Dict[str, Allocation] = {}
        self.compute_allocated = 0
        self.min_memory_block = 256 * 1024 * 1024  # 256MB minimum allocation
        logging.info(f"Initialized ResourcePool for GPU {gpu_id}")

    async def allocate_resources(self, 
                               memory_mb: int, 
                               compute_percentage: int,
                               priority: int = 0) -> Dict:
        """Allocate GPU resources"""
        try:
            # Validate requirements
            if not self.can_allocate(memory_mb, compute_percentage):
                raise ResourceError(
                    f"Insufficient resources. Required: {memory_mb}MB memory, "
                    f"{compute_percentage}% compute. Available: "
                    f"{self.get_available_memory()}MB memory, "
                    f"{100 - self.compute_allocated}% compute"
                )

            # Create allocation
            allocation_id = str(uuid.uuid4())
            allocation = Allocation(
                id=allocation_id,
                memory_mb=memory_mb,
                compute_percentage=compute_percentage,
                priority=priority,
                gpu_id=self.gpu_id
            )

            # Update resource tracking
            self.allocations[allocation_id] = allocation
            self.used_memory += memory_mb
            self.compute_allocated += compute_percentage

            logging.info(
                f"Allocated resources on GPU {self.gpu_id}: "
                f"{memory_mb}MB memory, {compute_percentage}% compute"
            )

            return allocation.to_dict()

        except Exception as e:
            logging.error(f"Resource allocation failed on GPU {self.gpu_id}: {e}")
            raise

    def can_allocate(self, memory_mb: int, compute_percentage: int) -> bool:
        """Check if resources can be allocated"""
        # Validate memory
        if memory_mb * 1024 * 1024 < self.min_memory_block:
            return False
            
        if self.used_memory + memory_mb > self.total_memory:
            return False

        # Validate compute percentage
        if compute_percentage < 0 or compute_percentage > 100:
            return False
            
        if self.compute_allocated + compute_percentage > 100:
            return False

        return True

    async def release_resources(self, allocation_id: str) -> bool:
        """Release allocated resources"""
        try:
            if allocation_id not in self.allocations:
                return False

            allocation = self.allocations.pop(allocation_id)
            self.used_memory -= allocation.memory_mb
            self.compute_allocated -= allocation.compute_percentage

            logging.info(
                f"Released resources on GPU {self.gpu_id}: "
                f"{allocation.memory_mb}MB memory, "
                f"{allocation.compute_percentage}% compute"
            )

            return True

        except Exception as e:
            logging.error(f"Resource release failed on GPU {self.gpu_id}: {e}")
            raise

    def get_utilization(self) -> float:
        """Get current GPU utilization percentage"""
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return util.gpu  # GPU utilization percentage

        except pynvml.NVMLError as e:
            logging.error(f"Failed to get GPU utilization: {e}")
            return 0.0

    def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return (memory.used / memory.total) * 100

        except pynvml.NVMLError as e:
            logging.error(f"Failed to get memory usage: {e}")
            return 0.0

    def get_available_memory(self) -> int:
        """Get available memory in MB"""
        return (self.total_memory - self.used_memory)

    def get_stats(self) -> Dict:
        """Get resource pool statistics"""
        return {
            "gpu_id": self.gpu_id,
            "total_memory": self.total_memory,
            "used_memory": self.used_memory,
            "available_memory": self.get_available_memory(),
            "compute_allocated": self.compute_allocated,
            "compute_available": 100 - self.compute_allocated,
            "utilization": self.get_utilization(),
            "active_allocations": len(self.allocations)
        }