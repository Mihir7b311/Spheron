import pynvml
from typing import Dict, Optional, List
import logging
from dataclasses import dataclass

@dataclass
class GPUDevice:
    """Represents physical GPU device"""
    device_id: int
    total_memory: int
    compute_capability: tuple
    name: str
    uuid: str
    temperature: int
    power_limit: int
    handle: pynvml.c_nvmlDevice_t

class GPUDeviceManager:
    """Manages physical GPU devices"""
    
    def __init__(self):
        self.devices: Dict[int, GPUDevice] = {}
        self.initialized = False
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                compute = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)
                
                self.devices[i] = GPUDevice(
                    device_id=i,
                    total_memory=info.total,
                    compute_capability=compute,
                    name=name.decode() if isinstance(name, bytes) else name,
                    uuid=uuid.decode() if isinstance(uuid, bytes) else uuid,
                    temperature=temp,
                    power_limit=power,
                    handle=handle
                )
            
            self.initialized = True
            logging.info(f"Initialized {len(self.devices)} GPU devices")
            
        except Exception as e:
            logging.error(f"Failed to initialize NVML: {e}")
            raise

    def get_device(self, device_id: int) -> Optional[GPUDevice]:
        """Get GPU device by ID"""
        return self.devices.get(device_id)

    def get_device_status(self, device_id: int) -> Dict:
        """Get current device status"""
        try:
            device = self.devices[device_id]
            info = pynvml.nvmlDeviceGetMemoryInfo(device.handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(device.handle)
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(device.handle)
            temp = pynvml.nvmlDeviceGetTemperature(device.handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(device.handle)
            
            return {
                "memory": {
                    "total": info.total,
                    "used": info.used,
                    "free": info.free
                },
                "utilization": {
                    "gpu": util.gpu,
                    "memory": util.memory
                },
                "processes": len(procs),
                "temperature": temp,
                "power_usage": power
            }
            
        except Exception as e:
            logging.error(f"Failed to get device status: {e}")
            return {}

    def get_processes(self, device_id: int) -> List[Dict]:
        """Get running processes on device"""
        try:
            device = self.devices[device_id]
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(device.handle)
            
            return [{
                "pid": p.pid,
                "used_memory": p.usedGpuMemory
            } for p in procs]
            
        except Exception as e:
            logging.error(f"Failed to get processes: {e}")
            return []

    def cleanup(self) -> None:
        """Cleanup device manager"""
        if self.initialized:
            try:
                pynvml.nvmlShutdown()
                self.initialized = False
            except:
                pass
            
    def __del__(self):
        self.cleanup()