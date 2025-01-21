# src/virtual_gpu/mps_manager.py

import subprocess
import os
import logging
import asyncio
from typing import Dict, Optional, List
from ..common.exceptions import MPSError

class MPSManager:
    def __init__(self):
        self.active_gpus = set()
        self.contexts: Dict[str, Dict] = {}
        self.mps_pipe_dir = "/tmp/nvidia-mps"
        self.max_processes = 48  # Maximum MPS processes per GPU
        self.min_compute = 10    # Minimum compute percentage per process

    async def setup_gpu(self, gpu_id: int) -> bool:
        """Setup MPS for specific GPU"""
        try:
            if gpu_id in self.active_gpus:
                return True

            # Set GPU to exclusive mode
            await self._run_command(
                f"nvidia-smi -i {gpu_id} -c EXCLUSIVE_PROCESS"
            )

            # Start MPS daemon
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            await self._run_command("nvidia-cuda-mps-control -d")

            self.active_gpus.add(gpu_id)
            logging.info(f"MPS setup complete for GPU {gpu_id}")
            return True

        except Exception as e:
            logging.error(f"MPS setup failed for GPU {gpu_id}: {e}")
            raise MPSError(f"MPS setup failed: {e}")

    async def create_context(self, gpu_id: int, compute_percentage: int) -> Dict:
        """Create MPS context with specified compute percentage"""
        try:
            if gpu_id not in self.active_gpus:
                raise MPSError(f"MPS not active for GPU {gpu_id}")

            # Validate compute percentage
            if compute_percentage < self.min_compute:
                raise MPSError(f"Compute percentage must be at least {self.min_compute}%")

            # Check active contexts count
            gpu_contexts = [ctx for ctx in self.contexts.values() if ctx["gpu_id"] == gpu_id]
            if len(gpu_contexts) >= self.max_processes:
                raise MPSError(f"Maximum MPS processes ({self.max_processes}) reached for GPU {gpu_id}")

            context_id = f"ctx_{len(self.contexts)}_{gpu_id}"
            
            # Set compute percentage
            await self._run_command(
                f"echo set_active_thread_percentage {context_id} "
                f"{compute_percentage} | nvidia-cuda-mps-control"
            )

            # Create context record
            context = {
                "id": context_id,
                "gpu_id": gpu_id,
                "compute_percentage": compute_percentage,
                "created_time": asyncio.get_event_loop().time()
            }
            self.contexts[context_id] = context
            
            logging.info(f"Created MPS context {context_id} on GPU {gpu_id}")
            return context

        except Exception as e:
            logging.error(f"Failed to create MPS context: {e}")
            raise MPSError(f"Context creation failed: {e}")

    async def release_context(self, context_id: str) -> bool:
        """Release MPS context"""
        try:
            if context_id not in self.contexts:
                return False

            context = self.contexts.pop(context_id)
            
            # Reset compute percentage
            await self._run_command(
                f"echo set_active_thread_percentage {context_id} "
                f"0 | nvidia-cuda-mps-control"
            )

            logging.info(f"Released MPS context {context_id} from GPU {context['gpu_id']}")
            return True

        except Exception as e:
            logging.error(f"Failed to release context {context_id}: {e}")
            raise MPSError(f"Context release failed: {e}")

    async def cleanup_gpu(self, gpu_id: int) -> bool:
        """Cleanup MPS for specific GPU"""
        try:
            if gpu_id not in self.active_gpus:
                return True

            # Remove all contexts for this GPU
            contexts_to_remove = [
                ctx_id for ctx_id, ctx in self.contexts.items() 
                if ctx["gpu_id"] == gpu_id
            ]
            for ctx_id in contexts_to_remove:
                await self.release_context(ctx_id)

            # Stop MPS daemon
            await self._run_command(
                "echo quit | nvidia-cuda-mps-control"
            )

            # Reset GPU mode
            await self._run_command(
                f"nvidia-smi -i {gpu_id} -c DEFAULT"
            )

            self.active_gpus.remove(gpu_id)
            logging.info(f"Cleaned up MPS for GPU {gpu_id}")
            return True

        except Exception as e:
            logging.error(f"MPS cleanup failed for GPU {gpu_id}: {e}")
            raise MPSError(f"MPS cleanup failed: {e}")

    async def cleanup_all(self):
        """Cleanup all MPS resources"""
        for gpu_id in list(self.active_gpus):
            await self.cleanup_gpu(gpu_id)

    async def get_context_info(self, context_id: str) -> Optional[Dict]:
        """Get information about MPS context"""
        return self.contexts.get(context_id)

    async def list_gpu_contexts(self, gpu_id: int) -> List[Dict]:
        """List all contexts for a specific GPU"""
        return [
            ctx for ctx in self.contexts.values() 
            if ctx["gpu_id"] == gpu_id
        ]

    async def _run_command(self, cmd: str) -> str:
        """Run shell command asynchronously"""
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise MPSError(f"Command failed: {stderr.decode()}")
        
        return stdout.decode()

    async def get_mps_stats(self, gpu_id: int) -> Dict:
        """Get MPS statistics for GPU"""
        if gpu_id not in self.active_gpus:
            return {
                "active": False,
                "contexts": 0,
                "total_compute": 0
            }

        contexts = await self.list_gpu_contexts(gpu_id)
        return {
            "active": True,
            "contexts": len(contexts),
            "total_compute": sum(ctx["compute_percentage"] for ctx in contexts)
        }