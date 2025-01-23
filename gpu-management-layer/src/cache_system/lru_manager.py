# src/cache_system/lru_manager.py
from collections import OrderedDict
from typing import Any, Optional
import logging

class LRUManager:
    def __init__(self, capacity: int):
        """Initialize LRU Cache with given capacity
        
        Args:
            capacity (int): Maximum number of items in cache
            
        Raises:
            ValueError: If capacity is less than or equal to 0
        """
        if capacity <= 0:
            raise ValueError("Cache capacity must be greater than 0")
            
        self.capacity = capacity
        self.cache = OrderedDict()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache and update access order"""
        if key is None:
            raise TypeError("Key cannot be None")
            
        if key in self.cache:
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.stats["hits"] += 1
            return value
        self.stats["misses"] += 1
        return None

    def put(self, key: str, value: Any) -> Optional[str]:
        """Add item to cache, return evicted key if any"""
        if key is None:
            raise TypeError("Key cannot be None")
            
        evicted_key = None
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used item
            evicted_key, _ = self.cache.popitem(last=False)
            self.stats["evictions"] += 1
            logging.info(f"Evicted key: {evicted_key}")
        
        self.cache[key] = value
        return evicted_key

    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache"""
        return key in self.cache

    def get_stats(self):
        """Get cache statistics"""
        return {
            **self.stats,
            "size": len(self.cache),
            "capacity": self.capacity,
            "hit_rate": self._calculate_hit_rate()
        }

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total > 0 else 0.0