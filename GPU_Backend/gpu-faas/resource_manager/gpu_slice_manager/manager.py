 # gpu_slice_manager/manager.py
import pynvml
from typing import Dict, Any, List
import uuid

# resource_manager/gpu_slice_manager/manager.py
from typing import Dict, Any
import pynvml

class GPUSliceManager:
    def __init__(self, gpu_config: Dict[str, Any]):
        self.min_memory = self._parse_memory(gpu_config["min_memory"])
        self.max_memory = self._parse_memory(gpu_config["max_memory"])
        self.default_compute = gpu_config.get("default_compute", 20)
        self.oversubscription_limit = gpu_config.get("oversubscription_limit", 1.0)
        
        try:
            pynvml.nvmlInit()
        except Exception as e:
            print(f"Warning: Could not initialize NVML: {e}")
    
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

    async def allocate_slice(self, request: Dict[str, Any]) -> Dict[str, Any]:
        memory = self._parse_memory(request["memory"])
        if memory > self.max_memory:
            raise Exception("Requested memory exceeds maximum")
            
        return {
            "gpu_id": request["gpu_id"],
            "memory": request["memory"],
            "compute_percentage": request["compute_percentage"]
        }

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except:
            pass