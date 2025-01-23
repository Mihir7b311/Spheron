# src/cache_system/model_cache.py
import torch
import logging
from typing import Dict, Any, Optional
from .lru_manager import LRUManager
from .memory_tracker import MemoryTracker

class ModelCache:
    def __init__(self, gpu_id: int, capacity_gb: float):
        """Initialize model cache with capacity in GB"""
        if capacity_gb <= 0:
            raise ValueError("Capacity must be greater than 0")
        
        self.gpu_id = gpu_id
        self.capacity_bytes = int(capacity_gb * 1024 * 1024 * 1024)
        self.used_memory = 0
        self.max_items = 10  # Maximum number of models in cache
        self.lru_manager = LRUManager(capacity=self.max_items)
        self.memory_tracker = MemoryTracker(gpu_id)
        self.model_sizes = {}
        
        logging.info(f"Initialized ModelCache for GPU {gpu_id} with capacity: {capacity_gb}GB")

    async def store_model(self, model_id: str, model: torch.nn.Module) -> bool:
        """Store model in cache"""
        try:
            # Calculate model size
            model_size = self._get_model_size(model)
            
            # Check if we have enough capacity
            if not self.memory_tracker.check_memory_availability(model_size):
                logging.warning(f"Insufficient memory for model {model_id}")
                return False

            # Handle eviction if needed
            if len(self.model_sizes) >= self.max_items:
                evicted_id = await self._evict_model()
                if not evicted_id:
                    return False

            # Store the model
            with torch.cuda.device(self.gpu_id):
                model = model.cuda()
                self.lru_manager.put(model_id, model)
                self.model_sizes[model_id] = model_size
                self.used_memory += model_size
                
            logging.info(f"Stored model {model_id} in GPU {self.gpu_id}")
            return True

        except Exception as e:
            logging.error(f"Failed to store model {model_id}: {e}")
            raise

    async def get_model(self, model_id: str) -> Optional[torch.nn.Module]:
        """Retrieve model from cache"""
        model = self.lru_manager.get(model_id)
        if model is not None:
            logging.info(f"Cache hit for model {model_id}")
        else:
            logging.info(f"Cache miss for model {model_id}")
        return model

    async def _evict_model(self) -> Optional[str]:
        """Evict least recently used model"""
        if not self.lru_manager.cache:
            return None

        # Get the least recently used model
        lru_key = next(iter(self.lru_manager.cache))
        if lru_key in self.model_sizes:
            freed_memory = self.model_sizes.pop(lru_key)
            self.used_memory -= freed_memory
            model = self.lru_manager.cache.pop(lru_key)
            del model  # Free GPU memory
            torch.cuda.empty_cache()
            logging.info(f"Evicted model {lru_key}")
            return lru_key
        return None

    def _get_model_size(self, model: torch.nn.Module) -> int:
        """Calculate model size in bytes"""
        total_params = sum(p.nelement() for p in model.parameters())
        bytes_per_param = sum(p.element_size() for p in model.parameters())
        return total_params * bytes_per_param

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        lru_stats = self.lru_manager.get_stats()
        memory_info = self.memory_tracker.get_memory_info()
        
        return {
            **lru_stats,
            "used_memory_bytes": self.used_memory,
            "total_memory_bytes": memory_info["total"],
            "free_memory_bytes": memory_info["free"],
            "memory_utilization": self.memory_tracker.get_utilization(),
            "num_models": len(self.model_sizes)
        }

    def __del__(self):
        """Cleanup cache"""
        try:
            for model in self.lru_manager.cache.values():
                del model
            torch.cuda.empty_cache()
        except:
            pass