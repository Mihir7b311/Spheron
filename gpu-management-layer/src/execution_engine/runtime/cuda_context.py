# src/execution_engine/runtime/cuda_context.py
import torch
import logging
from typing import Dict, Optional
from ..common.exceptions import CUDAError

class CUDAContext:
    def __init__(self, gpu_id: int):
        """Initialize CUDA context
        Args:
            gpu_id: GPU device ID
        """
        self.gpu_id = gpu_id
        self.device = None
        self.stream = None
        self.memory_pool = None
        self.active = False
        self.total_memory = 0
        self.memory_threshold = 0.9  # 90% memory alert threshold
        self._initialize_context()

    def _initialize_context(self):
        """Initialize CUDA context and memory tracking"""
        try:
            if not torch.cuda.is_available():
                raise CUDAError("CUDA is not available")

            # Initialize device
            self.device = torch.device(f'cuda:{self.gpu_id}')
            torch.cuda.set_device(self.gpu_id)

            # Create stream
            self.stream = torch.cuda.Stream(device=self.gpu_id)
            
            # Initialize memory tracking
            self.total_memory = torch.cuda.get_device_properties(self.device).total_memory
            self.memory_pool = torch.cuda.memory_stats(self.device)
            self.active = True

            # Initial cleanup
            torch.cuda.empty_cache()
            logging.info(f"Initialized CUDA context for GPU {self.gpu_id}")

        except Exception as e:
            logging.error(f"Failed to initialize CUDA context: {e}")
            raise CUDAError(f"Context initialization failed: {e}")

    def get_context_info(self) -> Dict:
        """Get context information and memory stats"""
        if not self.active:
            return {"status": "inactive"}

        try:
            memory_allocated = torch.cuda.memory_allocated(self.device)
            memory_reserved = torch.cuda.memory_reserved(self.device)
            memory_cached = torch.cuda.memory_cached(self.device)

            return {
                "gpu_id": self.gpu_id,
                "device": self.device,
                "stream": self.stream,
                "active": self.active,
                "memory": {
                    "total": self.total_memory,
                    "allocated": memory_allocated,
                    "reserved": memory_reserved,
                    "cached": memory_cached,
                    "utilization": memory_allocated / self.total_memory
                }
            }
        except Exception as e:
            logging.error(f"Failed to get context info: {e}")
            return {"status": "error", "error": str(e)}

    def check_memory_status(self) -> bool:
        """Check if memory usage is within threshold"""
        if not self.active:
            return False

        try:
            memory_allocated = torch.cuda.memory_allocated(self.device)
            utilization = memory_allocated / self.total_memory
            
            if utilization > self.memory_threshold:
                logging.warning(f"High memory utilization: {utilization*100:.2f}%")
                return False
            return True

        except Exception as e:
            logging.error(f"Failed to check memory status: {e}")
            return False

    def __enter__(self):
        """Context manager entry"""
        if not self.active:
            self._initialize_context()
        torch.cuda.set_device(self.gpu_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        try:
            if self.stream:
                self.stream.synchronize()
            torch.cuda.empty_cache()
            
            if exc_type is not None:
                logging.error(f"Context exit with error: {exc_type.__name__}: {exc_val}")
                return False
            return True

        except Exception as e:
            logging.error(f"Context exit failed: {e}")
            return False

    def cleanup(self):
        """Explicit cleanup of CUDA resources"""
        try:
            if self.stream:
                self.stream.synchronize()
            torch.cuda.empty_cache()
            self.active = False
            logging.info(f"Cleaned up CUDA context for GPU {self.gpu_id}")

        except Exception as e:
            logging.error(f"Cleanup failed: {e}")
            raise CUDAError(f"Context cleanup failed: {e}")