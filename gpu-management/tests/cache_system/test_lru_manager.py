# tests/cache_system/test_lru_manager.py
import pytest
from src.cache_system.lru_manager import LRUManager

@pytest.fixture
def lru_cache():
    """Create a test LRU cache with capacity of 3"""
    return LRUManager(capacity=3)

def test_initialization():
    """Test LRU cache initialization"""
    cache = LRUManager(capacity=3)
    assert cache.capacity == 3
    assert len(cache.cache) == 0
    assert cache.stats["hits"] == 0
    assert cache.stats["misses"] == 0
    assert cache.stats["evictions"] == 0

def test_put_and_get(lru_cache):
    """Test basic put and get operations"""
    # Put an item
    lru_cache.put("key1", "value1")
    assert len(lru_cache.cache) == 1
    
    # Get the item
    value = lru_cache.get("key1")
    assert value == "value1"
    assert lru_cache.stats["hits"] == 1

def test_cache_miss(lru_cache):
    """Test cache miss scenario"""
    value = lru_cache.get("nonexistent")
    assert value is None
    assert lru_cache.stats["misses"] == 1

def test_cache_eviction():
    """Test cache eviction when capacity is reached"""
    cache = LRUManager(capacity=2)
    
    # Fill cache
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    assert len(cache.cache) == 2
    
    # Add one more item
    evicted = cache.put("key3", "value3")
    assert evicted == "key1"  # First item should be evicted
    assert len(cache.cache) == 2
    assert cache.stats["evictions"] == 1

def test_access_order():
    """Test that accessing an item moves it to the end"""
    cache = LRUManager(capacity=3)
    
    # Add items
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    
    # Access key1
    cache.get("key1")
    
    # Add new item
    evicted = cache.put("key4", "value4")
    assert evicted == "key2"  # key2 should be evicted as key1 was recently used

def test_cache_stats():
    """Test cache statistics"""
    cache = LRUManager(capacity=2)
    
    # Add and access items
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.get("key1")  # Hit
    cache.get("key3")  # Miss
    cache.put("key3", "value3")  # Eviction
    
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["evictions"] == 1
    assert stats["size"] == 2
    assert stats["capacity"] == 2
    assert stats["hit_rate"] == 0.5  # 1 hit out of 2 total accesses

def test_update_existing():
    """Test updating an existing key"""
    cache = LRUManager(capacity=2)
    
    cache.put("key1", "value1")
    cache.put("key1", "new_value")
    
    assert cache.get("key1") == "new_value"
    assert len(cache.cache) == 1
    assert cache.stats["evictions"] == 0

@pytest.mark.parametrize("capacity", [1, 5, 10])
def test_different_capacities(capacity):
    """Test LRU cache with different capacities"""
    cache = LRUManager(capacity=capacity)
    
    # Fill cache
    for i in range(capacity + 1):
        cache.put(f"key{i}", f"value{i}")
    
    assert len(cache.cache) == capacity
    assert cache.stats["evictions"] == 1 if capacity < capacity + 1 else 0

def test_stress_test():
    """Stress test with many operations"""
    cache = LRUManager(capacity=100)
    
    # Add many items
    for i in range(200):
        cache.put(f"key{i}", f"value{i}")
        
        # Randomly access some items
        if i % 3 == 0:
            cache.get(f"key{i//2}")
    
    stats = cache.get_stats()
    assert len(cache.cache) == 100  # Should not exceed capacity
    assert stats["evictions"] > 0
    assert stats["hits"] + stats["misses"] > 0

def test_edge_cases():
    """Test edge cases"""
    # Test zero capacity
    with pytest.raises(ValueError):
        LRUManager(capacity=0)
    
    # Test negative capacity
    with pytest.raises(ValueError):
        LRUManager(capacity=-1)
    
    cache = LRUManager(capacity=1)
    # Test None key
    with pytest.raises(TypeError):
        cache.put(None, "value")
    
    # Test None value
    cache.put("key", None)
    assert cache.get("key") is None

def test_clear_cache(lru_cache):
    """Test clearing the cache"""
    lru_cache.put("key1", "value1")
    lru_cache.put("key2", "value2")
    
    lru_cache.clear()
    assert len(lru_cache.cache) == 0
    assert lru_cache.get("key1") is None

def test_contains(lru_cache):
    """Test checking if key exists in cache"""
    lru_cache.put("key1", "value1")
    
    assert "key1" in lru_cache
    assert "nonexistent" not in lru_cache