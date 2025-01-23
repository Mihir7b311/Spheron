# src/gpu_sharing/virtual_gpu.py

import pynvml
from typing import Dict, List, Optional
import logging
import asyncio
from dataclasses import dataclass
from ..common.exceptions import GPUError, ResourceError
from ..common.monitoring import ResourceMonitor
from ..cache_system import ModelCache
from ..execution_engine.cuda import CUDAContextManager

@dataclass
class VirtualGPUConfig:
    """Configuration for virtual GPU"""
    memory_mb: int
    compute_percentage: int
    priority: int = 0
    enable_mps: bool = True
    mps_percentage: int = 10
    max_processes: int = 48

class VirtualGPUManager:
    def __init__(self, config: Dict = None):
        """Initialize Virtual GPU Manager
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.initialized = False
        self.virtual_gpus: Dict[str, Dict] = {}
        self.physical_gpus: Dict[int, Dict] = {}
        self.cuda_context_manager = CUDAContextManager()
        self.model_cache = None  # Will be initialized per GPU
        self.monitor = ResourceMonitor()
        self._initialize_gpus()

    def _initialize_gpus(self) -> None:
        """Initialize GPU environment"""
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Initialize physical GPU tracking
                self.physical_gpus[i] = {
                    "handle": handle,
                    "total_memory": memory_info.total,
                    "allocated_memory": 0,
                    "allocated_compute": 0,
                    "virtual_gpus": []
                }
                
                # Initialize model cache for this GPU
                self.model_cache = ModelCache(
                    gpu_id=i,
                    capacity_gb=memory_info.total / (1024**3)
                )
                
            self.initialized = True
            logging.info(f"Initialized {device_count} GPUs for virtualization")
            
        except pynvml.NVMLError as e:
            logging.error(f"Failed to initialize GPUs: {e}")
            raise GPUError(f"GPU initialization failed: {e}")

    async def create_virtual_gpu(self, config: VirtualGPUConfig) -> Dict:
        """Create a virtual GPU instance
        Args:
            config: Virtual GPU configuration
        Returns:
            Dict containing virtual GPU details
        """
        if not self.initialized:
            raise GPUError("GPU Manager not initialized")

        try:
            # Find suitable physical GPU
            gpu_id = await self._find_suitable_gpu(
                config.memory_mb, 
                config.compute_percentage
            )
            if gpu_id is None:
                raise ResourceError("No suitable GPU available")

            # Generate virtual GPU ID
            v_gpu_id = f"vgpu_{len(self.virtual_gpus)}_{gpu_id}"

            # Set up virtual GPU
            v_gpu = {
                "id": v_gpu_id,
                "gpu_id": gpu_id,
                "config": config,
                "allocated_memory": config.memory_mb,
                "allocated_compute": config.compute_percentage,
                "active": True,
                "processes": [],
                "start_time": asyncio.get_event_loop().time()
            }

            # Update allocations
            self.physical_gpus[gpu_id]["allocated_memory"] += config.memory_mb
            self.physical_gpus[gpu_id]["allocated_compute"] += config.compute_percentage
            self.physical_gpus[gpu_id]["virtual_gpus"].append(v_gpu_id)
            self.virtual_gpus[v_gpu_id] = v_gpu

            # Set up CUDA context
            cuda_context = await self.cuda_context_manager.create_context(gpu_id)
            v_gpu["cuda_context"] = cuda_context

            logging.info(
                f"Created virtual GPU {v_gpu_id} on GPU {gpu_id} "
                f"(Memory: {config.memory_mb}MB, Compute: {config.compute_percentage}%)"
            )

            return v_gpu

        except Exception as e:
            logging.error(f"Failed to create virtual GPU: {e}")
            raise

    async def _find_suitable_gpu(self, 
                               required_memory: int,
                               required_compute: int) -> Optional[int]:
        """Find most suitable GPU for allocation
        Args:
            required_memory: Required memory in MB
            required_compute: Required compute percentage
        Returns:
            GPU ID if found, None otherwise
        """
        best_gpu = None
        best_score = float('inf')

        for gpu_id, gpu in self.physical_gpus.items():
            # Check if GPU can accommodate request
            if not self._can_allocate(gpu, required_memory, required_compute):
                continue

            # Calculate suitability score (lower is better)
            memory_usage = gpu["allocated_memory"] / (gpu["total_memory"] / (1024*1024))
            compute_usage = gpu["allocated_compute"] / 100
            
            # Consider both memory and compute utilization
            score = (memory_usage * 0.6) + (compute_usage * 0.4)

            # Prefer GPU with matching models in cache
            if self.model_cache and gpu_id == self.model_cache.gpu_id:
                score *= 0.8

            if score < best_score:
                best_score = score
                best_gpu = gpu_id

        return best_gpu

    def _can_allocate(self, 
                     gpu: Dict,
                     memory_mb: int,
                     compute_percentage: int) -> bool:
        """Check if GPU can accommodate allocation"""
        return (
            gpu["allocated_memory"] + memory_mb <= gpu["total_memory"] / (1024*1024) and
            gpu["allocated_compute"] + compute_percentage <= 100
        )

    async def release_virtual_gpu(self, v_gpu_id: str) -> bool:
        """Release a virtual GPU and its resources
        Args:
            v_gpu_id: Virtual GPU ID to release
        Returns:
            Success status
        """
        try:
            if v_gpu_id not in self.virtual_gpus:
                return False

            v_gpu = self.virtual_gpus[v_gpu_id]
            gpu_id = v_gpu["gpu_id"]

            # Release allocated resources
            self.physical_gpus[gpu_id]["allocated_memory"] -= v_gpu["allocated_memory"]
            self.physical_gpus[gpu_id]["allocated_compute"] -= v_gpu["allocated_compute"]
            self.physical_gpus[gpu_id]["virtual_gpus"].remove(v_gpu_id)

            # Clean up CUDA context
            if "cuda_context" in v_gpu:
                await self.cuda_context_manager.release_context(gpu_id)

            # Remove virtual GPU
            del self.virtual_gpus[v_gpu_id]

            logging.info(f"Released virtual GPU {v_gpu_id}")
            return True

        except Exception as e:
            logging.error(f"Failed to release virtual GPU {v_gpu_id}: {e}")
            raise

    def get_virtual_gpu_info(self, v_gpu_id: str) -> Optional[Dict]:
        """Get information about virtual GPU"""
        return self.virtual_gpus.get(v_gpu_id)

    def get_physical_gpu_info(self, gpu_id: int) -> Optional[Dict]:
        """Get information about physical GPU"""
        if gpu_id not in self.physical_gpus:
            return None

        gpu = self.physical_gpus[gpu_id]
        try:
            handle = gpu["handle"]
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

            return {
                "gpu_id": gpu_id,
                "total_memory_mb": gpu["total_memory"] / (1024*1024),
                "allocated_memory_mb": gpu["allocated_memory"],
                "allocated_compute": gpu["allocated_compute"],
                "utilization_gpu": utilization.gpu,
                "utilization_memory": utilization.memory,
                "virtual_gpus": len(gpu["virtual_gpus"]),
                "memory_info": {
                    "total": memory.total,
                    "used": memory.used,
                    "free": memory.free
                }
            }

        except Exception as e:
            logging.error(f"Failed to get GPU info: {e}")
            return None

    def get_stats(self) -> Dict:
        """Get overall statistics"""
        return {
            "total_physical_gpus": len(self.physical_gpus),
            "total_virtual_gpus": len(self.virtual_gpus),
            "gpu_utilization": {
                gpu_id: self.get_physical_gpu_info(gpu_id)
                for gpu_id in self.physical_gpus
            }
        }

    async def __aenter__(self):
        """Async context manager entry"""
        if not self.initialized:
            self._initialize_gpus()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Release all virtual GPUs
        for v_gpu_id in list(self.virtual_gpus.keys()):
            await self.release_virtual_gpu(v_gpu_id)
        
        # Cleanup NVML
        try:
            pynvml.nvmlShutdown()
        except:
            pass