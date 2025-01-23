# src/execution_engine/cuda/context_manager.py

import torch
import logging
from typing import Dict, Optional
from dataclasses import dataclass
import asyncio
from ..common.exceptions import CUDAError

@dataclass
class ContextInfo:
    """Information about a CUDA context"""
    gpu_id: int
    device: torch.device
    stream: torch.cuda.Stream
    total_memory: int
    is_active: bool = False
    last_used: float = 0.0

class CUDAContextManager:
    def __init__(self):
        """Initialize CUDA Context Manager"""
        self.contexts: Dict[int, ContextInfo] = {}
        self.active_context: Optional[int] = None
        self.initialized = False
        self._initialize()

    def _initialize(self):
        """Initialize CUDA environment"""
        try:
            if not torch.cuda.is_available():
                raise CUDAError("CUDA is not available")
            
            self.device_count = torch.cuda.device_count()
            self.initialized = True
            logging.info(f"Initialized CUDA Context Manager. Available devices: {self.device_count}")
            
        except Exception as e:
            logging.error(f"Failed to initialize CUDA Context Manager: {e}")
            raise CUDAError(f"Initialization failed: {e}")

    async def create_context(self, gpu_id: int) -> Dict:
        """Create CUDA context for GPU
        Args:
            gpu_id: GPU device ID
        Returns:
            Dict containing context information
        """
        try:
            if not self.initialized:
                raise CUDAError("CUDA Context Manager not initialized")

            if gpu_id >= self.device_count:
                raise CUDAError(f"Invalid GPU ID: {gpu_id}. Available devices: {self.device_count}")

            # Set device and create stream
            device = torch.device(f'cuda:{gpu_id}')
            torch.cuda.set_device(device)
            stream = torch.cuda.Stream(device=gpu_id)
            
            # Get memory info
            total_memory = torch.cuda.get_device_properties(device).total_memory

            # Create context info
            context = ContextInfo(
                gpu_id=gpu_id,
                device=device,
                stream=stream,
                total_memory=total_memory,
                is_active=True,
                last_used=asyncio.get_event_loop().time()
            )
            
            self.contexts[gpu_id] = context
            logging.info(f"Created CUDA context for GPU {gpu_id}")
            
            return self.get_context_info(gpu_id)

        except Exception as e:
            logging.error(f"Failed to create context for GPU {gpu_id}: {e}")
            raise CUDAError(f"Context creation failed: {e}")

    async def get_context(self, gpu_id: int) -> Dict:
        """Get or create CUDA context
        Args:
            gpu_id: GPU device ID
        Returns:
            Dict containing context information
        """
        if gpu_id not in self.contexts:
            return await self.create_context(gpu_id)
            
        context = self.contexts[gpu_id]
        context.last_used = asyncio.get_event_loop().time()
        return self.get_context_info(gpu_id)

    def get_context_info(self, gpu_id: int) -> Dict:
        """Get context information
        Args:
            gpu_id: GPU device ID
        Returns:
            Dict containing context information
        """
        if gpu_id not in self.contexts:
            return {"status": "not_found"}
            
        context = self.contexts[gpu_id]
        
        try:
            memory_allocated = torch.cuda.memory_allocated(context.device)
            memory_reserved = torch.cuda.memory_reserved(context.device)
            
            return {
                "gpu_id": context.gpu_id,
                "device": context.device,
                "stream": context.stream,
                "is_active": context.is_active,
                "last_used": context.last_used,
                "memory": {
                    "total": context.total_memory,
                    "allocated": memory_allocated,
                    "reserved": memory_reserved,
                    "available": context.total_memory - memory_allocated
                }
            }
        except Exception as e:
            logging.error(f"Failed to get context info for GPU {gpu_id}: {e}")
            return {"status": "error", "error": str(e)}

    async def release_context(self, gpu_id: int):
        """Release CUDA context
        Args:
            gpu_id: GPU device ID
        """
        try:
            if gpu_id not in self.contexts:
                return
                
            context = self.contexts[gpu_id]
            
            # Synchronize and clean up
            context.stream.synchronize()
            torch.cuda.empty_cache()
            
            context.is_active = False
            del self.contexts[gpu_id]
            
            logging.info(f"Released CUDA context for GPU {gpu_id}")
            
        except Exception as e:
            logging.error(f"Failed to release context for GPU {gpu_id}: {e}")
            raise CUDAError(f"Context release failed: {e}")

    async def synchronize_context(self, gpu_id: int):
        """Synchronize CUDA context
        Args:
            gpu_id: GPU device ID
        """
        try:
            if gpu_id not in self.contexts:
                return
                
            context = self.contexts[gpu_id]
            context.stream.synchronize()
            
        except Exception as e:
            logging.error(f"Failed to synchronize context for GPU {gpu_id}: {e}")
            raise CUDAError(f"Context synchronization failed: {e}")

    def check_memory_availability(self, gpu_id: int, required_memory: int) -> bool:
        """Check if required memory is available
        Args:
            gpu_id: GPU device ID
            required_memory: Required memory in bytes
        Returns:
            bool indicating if memory is available
        """
        try:
            if gpu_id not in self.contexts:
                return False
                
            context = self.contexts[gpu_id]
            memory_info = self.get_context_info(gpu_id)["memory"]
            
            return memory_info["available"] >= required_memory
            
        except Exception as e:
            logging.error(f"Failed to check memory availability for GPU {gpu_id}: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get statistics for all contexts
        Returns:
            Dict containing context statistics
        """
        return {
            "total_contexts": len(self.contexts),
            "active_contexts": sum(1 for c in self.contexts.values() if c.is_active),
            "contexts": {
                gpu_id: self.get_context_info(gpu_id)
                for gpu_id in self.contexts
            }
        }

    async def cleanup(self):
        """Clean up all contexts"""
        try:
            for gpu_id in list(self.contexts.keys()):
                await self.release_context(gpu_id)
                
            logging.info("Cleaned up all CUDA contexts")
            
        except Exception as e:
            logging.error(f"Failed to clean up contexts: {e}")
            raise CUDAError(f"Cleanup failed: {e}")

    async def __aenter__(self):
        """Async context manager entry"""
        if not self.initialized:
            self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()