# tests/cache_system/test_lru_manager.py

import pytest
from src.cache_system.lru_manager import LRUManager

class TestLRUManager:
    @pytest.fixture
    def lru_cache(self):
        """Create a test LRU cache"""
        return LRUManager(capacity=3)

    def test_initialization(self, lru_cache):
        """Test LRU cache initialization"""
        assert lru_cache.capacity == 3
        assert len(lru_cache.cache) == 0
        assert lru_cache.stats["hits"] == 0
        assert lru_cache.stats["misses"] == 0
        assert lru_cache.stats["evictions"] == 0

    def test_put_and_get(self, lru_cache):
        """Test basic put and get operations"""
        lru_cache.put("key1", "value1")
        assert len(lru_cache.cache) == 1
        
        value = lru_cache.get("key1")
        assert value == "value1"
        assert lru_cache.stats["hits"] == 1

    def test_eviction(self, lru_cache):
        """Test eviction when capacity is reached"""
        # Fill cache
        lru_cache.put("key1", "value1")
        lru_cache.put("key2", "value2")
        lru_cache.put("key3", "value3")
        
        # Add another item
        evicted = lru_cache.put("key4", "value4")
        assert evicted == "key1"  # First item should be evicted
        assert len(lru_cache.cache) == 3
        assert lru_cache.stats["evictions"] == 1

    def test_cache_stats(self, lru_cache):
        """Test cache statistics"""
        lru_cache.put("key1", "value1")
        lru_cache.get("key1")  # Hit
        lru_cache.get("key2")  # Miss
        
        stats = lru_cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["capacity"] == 3



    # Additional tests for test_lru_manager.py

    def test_thread_safety(self, lru_cache):
        """Test thread-safe operations"""
        import threading
        
        def cache_operation():
            for i in range(100):
                lru_cache.put(f"key_{i}", f"value_{i}")
                lru_cache.get(f"key_{i}")
        
        # Create multiple threads
        threads = [threading.Thread(target=cache_operation) for _ in range(5)]
        
        # Start threads
        for t in threads:
            t.start()
        
        # Wait for completion    
        for t in threads:
            t.join()
            
        # Verify cache integrity
        assert len(lru_cache.cache) <= lru_cache.capacity
        
    def test_custom_eviction(self, lru_cache):
        """Test custom eviction policies"""
        # Fill cache
        items = [("key1", 1), ("key2", 2), ("key3", 3)]
        for k, v in items:
            lru_cache.put(k, v)
            
        # Custom eviction based on value
        def custom_evict(cache):
            return min(cache.items(), key=lambda x: x[1])[0]
            
        # Test custom eviction
        evicted = lru_cache.put("key4", 4, eviction_callback=custom_evict)
        assert evicted == "key1"  # Should evict smallest value

    @pytest.mark.stress
    def test_stress_test(self):
        """Stress test cache operations"""
        large_cache = LRUManager(capacity=1000)
        
        # Perform many rapid operations
        for i in range(10000):
            large_cache.put(f"key_{i}", f"value_{i}")
            
            # Random access pattern
            if i % 3 == 0:
                large_cache.get(f"key_{i//2}")
                
        # Verify cache integrity
        assert len(large_cache.cache) <= large_cache.capacity
        stats = large_cache.get_stats()
        assert stats["hits"] + stats["misses"] == 10000//3  # One access every 3 operations