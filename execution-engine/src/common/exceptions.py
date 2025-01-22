# src/common/exceptions.py

class ExecutionError(Exception):
    """Base exception for execution engine"""
    pass

class RuntimeError(ExecutionError):
    """Exception for Python runtime issues"""
    pass

class BatchError(ExecutionError):
    """Exception for batch processing issues"""
    pass

class CUDAError(ExecutionError):
    """Exception for CUDA-related issues"""
    pass

# src/common/metrics.py

from dataclasses import dataclass
from typing import Dict, Any
import time

@dataclass
class ExecutionMetrics:
    start_time: float
    end_time: float
    cuda_time: float
    batch_size: int
    memory_used: int
    compute_used: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_time": self.end_time - self.start_time,
            "cuda_time": self.cuda_time,
            "batch_size": self.batch_size,
            "memory_used": self.memory_used,
            "compute_used": self.compute_used
        }