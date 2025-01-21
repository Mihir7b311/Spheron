# src/virtual_gpu/manager.py

import pynvml
from typing import Dict, List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
from .mps_manager import MPSManager
from .resource_pool import ResourcePool
from ..common.exceptions import GPUError, ResourceError
from ..common.monitoring import ResourceMonitor

class VirtualGPUManager:
    def __init__(self):
        self.initialized = False
        self.gpus: Dict[int, ResourcePool] = {}
        self.mps_manager = MPSManager()
        self.monitor = ResourceMonitor()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._initialize_gpus()

    def _initialize_gpus(self) -> None:
        """Initialize GPU environment and create resource pools"""
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Create resource pool for each GPU
                self.gpus[i] = ResourcePool(
                    gpu_id=i,
                    total_memory=memory_info.total,
                    handle=handle
                )
                
                # Initialize MPS for this GPU
                self.mps_manager.setup_gpu(i)
                
            self.initialized = True
            logging.info(f"Initialized {device_count} GPUs")
            
        except pynvml.NVMLError as e:
            logging.error(f"Failed to initialize GPUs: {e}")
            raise GPUError(f"GPU initialization failed: {e}")

    async def create_virtual_gpu(self, 
                               memory_mb: int, 
                               compute_percentage: int,
                               priority: int = 0) -> Dict:
        """Create a virtual GPU instance with specified resources"""
        if not self.initialized:
            raise GPUError("GPU Manager not initialized")

        try:
            # Find best GPU for allocation
            gpu_id = self._find_suitable_gpu(memory_mb, compute_percentage)
            if gpu_id is None:
                raise ResourceError("No suitable GPU available")

            # Allocate resources
            v_gpu = await self.gpus[gpu_id].allocate_resources(
                memory_mb=memory_mb,
                compute_percentage=compute_percentage,
                priority=priority
            )

            # Setup MPS context
            mps_context = await self.mps_manager.create_context(
                gpu_id=gpu_id,
                compute_percentage=compute_percentage
            )

            return {
                "v_gpu_id": v_gpu["id"],
                "gpu_id": gpu_id,
                "allocated_memory": memory_mb,
                "compute_percentage": compute_percentage,
                "mps_context": mps_context
            }

        except Exception as e:
            logging.error(f"Failed to create virtual GPU: {e}")
            raise

    def _find_suitable_gpu(self, 
                          required_memory: int, 
                          required_compute: int) -> Optional[int]:
        """Find most suitable GPU for given requirements"""
        best_gpu = None
        best_score = float('inf')

        for gpu_id, pool in self.gpus.items():
            # Check if GPU can accommodate request
            if not pool.can_allocate(required_memory, required_compute):
                continue

            # Calculate suitability score (lower is better)
            current_load = pool.get_utilization()
            memory_usage = pool.get_memory_usage()
            
            score = (
                (current_load * 0.6) +  # Weight utilization more
                (memory_usage * 0.4)    # Weight memory usage less
            )

            if score < best_score:
                best_score = score
                best_gpu = gpu_id

        return best_gpu

    async def release_virtual_gpu(self, v_gpu_id: str) -> bool:
        """Release a virtual GPU and its resources"""
        try:
            for gpu in self.gpus.values():
                if await gpu.release_resources(v_gpu_id):
                    await self.mps_manager.release_context(v_gpu_id)
                    return True
            return False

        except Exception as e:
            logging.error(f"Failed to release virtual GPU {v_gpu_id}: {e}")
            raise

    def get_gpu_status(self, gpu_id: int) -> Dict:
        """Get current status of specified GPU"""
        if gpu_id not in self.gpus:
            raise GPUError(f"Invalid GPU ID: {gpu_id}")

        try:
            pool = self.gpus[gpu_id]
            return {
                "gpu_id": gpu_id,
                "total_memory": pool.total_memory,
                "used_memory": pool.used_memory,
                "utilization": pool.get_utilization(),
                "active_vgpus": len(pool.allocations),
                "temperature": pool.get_temperature(),
                "power_usage": pool.get_power_usage()
            }

        except Exception as e:
            logging.error(f"Failed to get GPU {gpu_id} status: {e}")
            raise

    def __del__(self):
        """Cleanup resources"""
        try:
            if self.initialized:
                self.mps_manager.cleanup_all()
                pynvml.nvmlShutdown()
                self.executor.shutdown()
        except:
            pass