# mps_controller/controller.py
import subprocess
import os
from typing import Dict, Any
import asyncio

class MPSController:
    def __init__(self, mps_config: Dict[str, Any]):
        self.max_processes = mps_config["max_processes"]
        self.compute_percentage = mps_config["compute_percentage"]
        self.default_memory = mps_config["default_memory"]
        self.max_memory = mps_config["max_memory"]
        self.active_gpus = set()

    async def setup_mps(self, gpu_id: str):
        """Setup NVIDIA MPS for a GPU"""
        if gpu_id in self.active_gpus:
            return True

        try:
            # Set GPU to exclusive process mode
            await self._run_command(f"nvidia-smi -i {gpu_id} -c EXCLUSIVE_PROCESS")
            
            # Start MPS daemon
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            await self._run_command("nvidia-cuda-mps-control -d")
            
            self.active_gpus.add(gpu_id)
            return True
        except Exception as e:
            raise Exception(f"Failed to setup MPS for GPU {gpu_id}: {str(e)}")

    async def stop_mps(self, gpu_id: str):
        """Stop MPS for a GPU"""
        if gpu_id not in self.active_gpus:
            return True

        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            await self._run_command("echo quit | nvidia-cuda-mps-control")
            
            # Reset GPU mode
            await self._run_command(f"nvidia-smi -i {gpu_id} -c DEFAULT")
            
            self.active_gpus.remove(gpu_id)
            return True
        except Exception as e:
            raise Exception(f"Failed to stop MPS for GPU {gpu_id}: {str(e)}")

    async def set_compute_percentage(self, gpu_id: str, percentage: int):
        """Set compute percentage for MPS"""
        if percentage < self.compute_percentage or percentage > 100:
            raise ValueError("Invalid compute percentage")

        try:
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(percentage)
            return True
        except Exception as e:
            raise Exception(f"Failed to set compute percentage: {str(e)}")

    async def _run_command(self, command: str):
        """Run shell command asynchronously"""
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Command failed: {stderr.decode()}")
        
        return stdout.decode() 
