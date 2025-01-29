import os
import asyncio
import logging
import pynvml
from typing import Dict, List, Optional
from .exceptions import MPSDaemonError

class MPSDaemon:
    """NVIDIA Multi-Process Service Daemon Manager"""
    
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.active = False
        self.max_processes = 48  # Maximum MPS client processes
        self.compute_percentage = 0  # Current compute percentage
        self.clients: List[int] = []  # Client process IDs
        self.initialized = False
        self._init_nvml()

    def _init_nvml(self) -> None:
        """Initialize NVIDIA Management Library"""
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            self.initialized = True
            logging.info(f"Initialized NVML for GPU {self.gpu_id}")
        except Exception as e:
            logging.error(f"Failed to initialize NVML: {e}")
            raise MPSDaemonError(f"NVML initialization failed: {e}")

    async def start_daemon(self) -> None:
        """Start MPS Control Daemon"""
        try:
            if not self.initialized:
                raise MPSDaemonError("NVML not initialized")

            # Set GPU to exclusive process mode
            await self._run_command(
                f"nvidia-smi -i {self.gpu_id} -c EXCLUSIVE_PROCESS"
            )
            
            # Set environment variables
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            
            # Start MPS daemon
            await self._run_command("nvidia-cuda-mps-control -d")
            
            self.active = True
            logging.info(f"Started MPS daemon for GPU {self.gpu_id}")
            
        except Exception as e:
            logging.error(f"Failed to start MPS daemon: {e}")
            raise MPSDaemonError(f"Failed to start MPS daemon: {e}")

    async def stop_daemon(self) -> None:
        """Stop MPS Control Daemon"""
        try:
            if self.active:
                # Stop MPS daemon
                await self._run_command("echo quit | nvidia-cuda-mps-control")
                
                # Reset GPU mode
                await self._run_command(
                    f"nvidia-smi -i {self.gpu_id} -c DEFAULT"
                )
                
                self.active = False
                self.clients.clear()
                logging.info(f"Stopped MPS daemon for GPU {self.gpu_id}")
                
        except Exception as e:
            logging.error(f"Failed to stop MPS daemon: {e}")
            raise MPSDaemonError(f"Failed to stop MPS daemon: {e}")

    async def set_compute_percentage(self, percentage: int) -> None:
        """Set compute percentage for MPS clients"""
        try:
            if not 0 <= percentage <= 100:
                raise ValueError("Percentage must be between 0 and 100")

            if self.active:
                await self._run_command(
                    f"echo set_active_thread_percentage {percentage} | "
                    "nvidia-cuda-mps-control"
                )
                self.compute_percentage = percentage
                logging.info(
                    f"Set compute percentage to {percentage}% for GPU {self.gpu_id}"
                )
                
        except Exception as e:
            logging.error(f"Failed to set compute percentage: {e}")
            raise MPSDaemonError(f"Failed to set compute percentage: {e}")

    async def register_client(self, pid: int) -> None:
        """Register MPS client process"""
        if pid not in self.clients:
            if len(self.clients) >= self.max_processes:
                raise MPSDaemonError("Maximum MPS clients reached")
            
            self.clients.append(pid)
            logging.info(f"Registered MPS client {pid} on GPU {self.gpu_id}")

    async def unregister_client(self, pid: int) -> None:
        """Unregister MPS client process"""
        if pid in self.clients:
            self.clients.remove(pid)
            logging.info(f"Unregistered MPS client {pid} from GPU {self.gpu_id}")

    async def _run_command(self, cmd: str) -> str:
        """Run shell command asynchronously"""
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise MPSDaemonError(f"Command failed: {stderr.decode()}")
            
        return stdout.decode()

    def get_status(self) -> Dict:
        """Get MPS daemon status"""
        try:
            if not self.initialized:
                return {"status": "uninitialized"}

            memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            
            return {
                "active": self.active,
                "gpu_id": self.gpu_id,
                "compute_percentage": self.compute_percentage,
                "client_count": len(self.clients),
                "memory_used": memory.used,
                "memory_total": memory.total,
                "gpu_utilization": utilization.gpu,
                "memory_utilization": utilization.memory
            }
            
        except Exception as e:
            logging.error(f"Failed to get MPS status: {e}")
            return {"status": "error", "error": str(e)}

    async def cleanup(self) -> None:
        """Cleanup MPS daemon resources"""
        try:
            if self.active:
                await self.stop_daemon()
            if self.initialized:
                pynvml.nvmlShutdown()
                self.initialized = False
            logging.info(f"Cleaned up MPS daemon for GPU {self.gpu_id}")
        except Exception as e:
            logging.error(f"Failed to cleanup MPS daemon: {e}")
            raise MPSDaemonError(f"Cleanup failed: {e}")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_daemon()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()