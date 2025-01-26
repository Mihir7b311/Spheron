from typing import Dict, List
import pynvml
import subprocess
import logging
import asyncio
from dataclasses import dataclass
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def synchronized(lock):
    """Synchronization decorator for thread safety."""
    def wrapper(func):
        def wrapped(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)
        return wrapped
    return wrapper

@dataclass
class GPUSlice:
    """Represents a slice of GPU resources."""
    slice_id: str
    gpu_id: int
    memory_mb: int
    compute_percentage: int
    active: bool = False
    processes: List[int] = None

class MPSDaemon:
    """Manages NVIDIA Multi-Process Service Daemon."""
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.active = False
        self.lock = Lock()
        self._init_nvml()

    def _init_nvml(self):
        """Initialize NVIDIA Management Library."""
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        except Exception as e:
            logging.error(f"Failed to initialize NVML for GPU {self.gpu_id}: {e}")
            raise

    @synchronized(Lock())
    async def start_daemon(self):
        """Start MPS Control Daemon."""
        try:
            await self._run_command(f"nvidia-smi -i {self.gpu_id} -c EXCLUSIVE_PROCESS")
            await self._run_command("nvidia-cuda-mps-control -d")
            self.active = True
            logging.info(f"Started MPS daemon for GPU {self.gpu_id}")
        except Exception as e:
            logging.error(f"Failed to start MPS daemon for GPU {self.gpu_id}: {e}")
            raise

    @synchronized(Lock())
    async def stop_daemon(self):
        """Stop MPS Control Daemon."""
        try:
            if self.active:
                await self._run_command("echo quit | nvidia-cuda-mps-control")
                await self._run_command(f"nvidia-smi -i {self.gpu_id} -c DEFAULT")
                self.active = False
                logging.info(f"Stopped MPS daemon for GPU {self.gpu_id}")
        except Exception as e:
            logging.error(f"Failed to stop MPS daemon for GPU {self.gpu_id}: {e}")
            raise

    async def _run_command(self, cmd: str) -> str:
        """Run shell command asynchronously."""
        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise Exception(f"Command failed: {stderr.decode()}")
            return stdout.decode()
        except Exception as e:
            logging.error(f"Error running command '{cmd}': {e}")
            raise

class GPUInfrastructure:
    """Main GPU Infrastructure Manager."""
    def __init__(self):
        self.mps_daemons: Dict[int, MPSDaemon] = {}
        self.gpu_slices: Dict[str, GPUSlice] = {}
        self.devices: Dict[int, Dict] = {}
        self.lock = Lock()
        self._init_infrastructure()

    def _init_infrastructure(self):
        """Initialize GPU infrastructure."""
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()

            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

                self.devices[i] = {
                    "handle": handle,
                    "total_memory": memory.total,
                    "available_memory": memory.free
                }

                self.mps_daemons[i] = MPSDaemon(i)

            logging.info(f"Initialized GPU infrastructure with {self.device_count} devices")
        except Exception as e:
            logging.error(f"Failed to initialize GPU infrastructure: {e}")
            raise

    @synchronized(Lock())
    async def create_gpu_slice(self, gpu_id: int, memory_mb: int, compute_percentage: int) -> GPUSlice:
        """Create a new GPU slice."""
        try:
            if gpu_id not in self.devices:
                raise ValueError(f"Invalid GPU ID: {gpu_id}")

            if not self._validate_resources(gpu_id, memory_mb, compute_percentage):
                raise ValueError("Insufficient resources")

            slice_id = f"slice_{len(self.gpu_slices)}_{gpu_id}"
            slice = GPUSlice(
                slice_id=slice_id,
                gpu_id=gpu_id,
                memory_mb=memory_mb,
                compute_percentage=compute_percentage,
                processes=[]
            )

            self.gpu_slices[slice_id] = slice
            self.devices[gpu_id]["available_memory"] -= memory_mb * 1024 * 1024

            logging.info(
                f"Created GPU slice {slice_id} on GPU {gpu_id} ({memory_mb}MB, {compute_percentage}% compute)"
            )

            return slice
        except Exception as e:
            logging.error(f"Failed to create GPU slice: {e}")
            raise

    def _validate_resources(self, gpu_id: int, memory_mb: int, compute_percentage: int) -> bool:
        """Validate resource availability."""
        try:
            device = self.devices[gpu_id]
            available_mb = device["available_memory"] / (1024 * 1024)

            if memory_mb > available_mb:
                return False

            used_compute = sum(
                slice.compute_percentage
                for slice in self.gpu_slices.values()
                if slice.gpu_id == gpu_id
            )

            if used_compute + compute_percentage > 100:
                return False

            return True
        except Exception as e:
            logging.error(f"Resource validation failed: {e}")
            return False

    async def start_infrastructure(self):
        """Start GPU infrastructure."""
        try:
            for daemon in self.mps_daemons.values():
                await daemon.start_daemon()
            logging.info("Started GPU infrastructure")
        except Exception as e:
            logging.error(f"Failed to start infrastructure: {e}")
            raise

    async def stop_infrastructure(self):
        """Stop GPU infrastructure."""
        try:
            for daemon in self.mps_daemons.values():
                await daemon.stop_daemon()
            self.gpu_slices.clear()
            pynvml.nvmlShutdown()
            logging.info("Stopped GPU infrastructure")
        except Exception as e:
            logging.error(f"Failed to stop infrastructure: {e}")
            raise

    def get_infrastructure_status(self) -> Dict:
        """Get infrastructure status."""
        try:
            status = {
                "device_count": self.device_count,
                "devices": {},
                "slices": len(self.gpu_slices)
            }

            for gpu_id, device in self.devices.items():
                memory = pynvml.nvmlDeviceGetMemoryInfo(device["handle"])
                utilization = pynvml.nvmlDeviceGetUtilizationRates(device["handle"])

                status["devices"][gpu_id] = {
                    "total_memory": device["total_memory"],
                    "available_memory": device["available_memory"],
                    "utilization": utilization.gpu,
                    "memory_used": memory.used,
                    "active_slices": len([
                        s for s in self.gpu_slices.values()
                        if s.gpu_id == gpu_id
                    ])
                }

            return status
        except Exception as e:
            logging.error(f"Failed to get infrastructure status: {e}")
            return {"error": str(e)}

    async def __aenter__(self):
        await self.start_infrastructure()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop_infrastructure()
