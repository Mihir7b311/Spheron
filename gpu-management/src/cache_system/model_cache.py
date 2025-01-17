# src/cache_system/model_cache.py
from typing import Dict, Any, Optional
import torch
import logging
from .lru_manager import LRUManager
from .exceptions import CacheException

class ModelCache:
    def __init__(self, capacity_gb: float):
        """Initialize model cache with capacity in GB"""
        if capacity_gb <= 0:
            raise ValueError("Capacity must be greater than 0")
        
        self.capacity_bytes = int(capacity_gb * 1024 * 1024 * 1024)
        self.used_memory = 0
        # Set a smaller max_items to force evictions
        self.max_items = 2  # Force eviction after 2 items
        self.lru_manager = LRUManager(capacity=self.max_items)
        self.model_sizes = {}
        logging.info(f"Initialized ModelCache with capacity: {capacity_gb}GB, max items: {self.max_items}")

    async def store_model(self, model_id: str, model: torch.nn.Module) -> bool:
        """Store model in cache"""
        try:
            # Calculate model size
            model_size = self._get_model_size(model)
            logging.info(f"Attempting to store model {model_id} of size {model_size} bytes")

            # Check if we have enough capacity
            if model_size > self.capacity_bytes:
                raise CacheException(f"Model too large ({model_size} bytes) for cache ({self.capacity_bytes} bytes)")

            # Check if we need to evict based on number of items
            if len(self.model_sizes) >= self.max_items:
                logging.info("Cache full, attempting eviction")
                evicted_id = await self._evict_model()
                if evicted_id:
                    logging.info(f"Evicted model {evicted_id}")
                else:
                    raise CacheException("Failed to evict model")

            # Store the model
            old_key = self.lru_manager.put(model_id, model)
            if old_key:
                # Handle evicted model's memory
                if old_key in self.model_sizes:
                    old_size = self.model_sizes.pop(old_key)
                    self.used_memory -= old_size
                    self.lru_manager.stats["evictions"] += 1
                    logging.info(f"Evicted model {old_key}, freed {old_size} bytes")

            # Update memory usage
            self.model_sizes[model_id] = model_size
            self.used_memory += model_size
            
            logging.info(f"Successfully stored model {model_id}. Current memory: {self.used_memory}/{self.capacity_bytes}")
            return True

        except Exception as e:
            logging.error(f"Failed to store model {model_id}: {str(e)}")
            raise

    async def _evict_model(self) -> Optional[str]:
        """Evict least recently used model"""
        if not self.lru_manager.cache:
            return None

        # Get the least recently used model
        lru_key = next(iter(self.lru_manager.cache))
        if lru_key in self.model_sizes:
            freed_memory = self.model_sizes.pop(lru_key)
            self.used_memory -= freed_memory
            self.lru_manager.cache.pop(lru_key)
            self.lru_manager.stats["evictions"] += 1
            logging.info(f"Evicted model {lru_key}, freed {freed_memory} bytes")
            return lru_key
        return None

    async def get_model(self, model_id: str) -> Optional[torch.nn.Module]:
        """Retrieve model from cache"""
        model = self.lru_manager.get(model_id)
        if model is not None:
            logging.info(f"Cache hit for model {model_id}")
        else:
            logging.info(f"Cache miss for model {model_id}")
        return model

    def _get_model_size(self, model: torch.nn.Module) -> int:
        """Calculate model size in bytes"""
        total_params = sum(p.nelement() for p in model.parameters())
        bytes_per_param = sum(p.element_size() for p in model.parameters())
        total_size = total_params * bytes_per_param
        return total_size

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        lru_stats = self.lru_manager.get_stats()
        return {
            **lru_stats,
            "used_memory_bytes": self.used_memory,
            "capacity_bytes": self.capacity_bytes,
            "memory_utilization": self.used_memory / self.capacity_bytes,
            "num_models": len(self.model_sizes)
        }