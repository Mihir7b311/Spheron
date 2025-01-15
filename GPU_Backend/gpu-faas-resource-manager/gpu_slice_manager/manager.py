 # gpu_slice_manager/manager.py
import pynvml
from typing import Dict, Any, List
import uuid

class GPUSliceManager:
    def __init__(self, gpu_config: Dict[str, Any]):
        self.min_memory = self._parse_memory(gpu_config["min_memory"])
        self.max_memory = self._parse_memory(gpu_config["max_memory"])
        self.default_compute = gpu_config["default_compute"]
        self.oversubscription_limit = gpu_config["oversubscription_limit"]
        
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        self.slices = {}

    async def allocate_slice(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate a GPU slice"""
        required_memory = self._parse_memory(request.get("memory", "1Gi"))
        required_compute = request.get("compute_percentage", self.default_compute)

        # Find suitable GPU
        gpu_id = await self._find_suitable_gpu(required_memory, required_compute)
        if not gpu_id:
            raise Exception("No suitable GPU available")

        # Create slice
        slice_id = str(uuid.uuid4())
        slice_info = {
            "slice_id": slice_id,
            "gpu_id": gpu_id,
            "memory": required_memory,
            "compute_percentage": required_compute,
            "gpu_fraction": 1 / self.oversubscription_limit
        }

        self.slices[slice_id] = slice_info
        return slice_info

    async def _find_suitable_gpu(self, required_memory: int, required_compute: int) -> str:
        """Find GPU with enough resources"""
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Check memory availability
            if info.free >= required_memory:
                # Check compute availability
                used_compute = sum(
                    slice_info["compute_percentage"]
                    for slice_info in self.slices.values()
                    if slice_info["gpu_id"] == str(i)
                )
                
                if used_compute + required_compute <= 100 * self.oversubscription_limit:
                    return str(i)
                    
        return None

    def _parse_memory(self, memory_str: str) -> int:
        """Parse memory string (e.g., '1Gi') to bytes"""
        units = {
            'Ki': 1024,
            'Mi': 1024 ** 2,
            'Gi': 1024 ** 3,
            'Ti': 1024 ** 4
        }
        
        number = int(''.join(filter(str.isdigit, memory_str)))
        unit = ''.join(filter(str.isalpha, memory_str))
        
        return number * units.get(unit, 1)

    async def release_slice(self, slice_id: str):
        """Release a GPU slice"""
        if slice_id in self.slices:
            del self.slices[slice_id]
            return True
        return False

    async def get_slice_info(self, slice_id: str) -> Dict[str, Any]:
        """Get information about a GPU slice"""
        return self.slices.get(slice_id)

    def __del__(self):
        pynvml.nvmlShutdown()
