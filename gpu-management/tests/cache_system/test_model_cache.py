# tests/cache_system/test_model_cache.py
import pytest
import torch
from src.cache_system.model_cache import ModelCache
from src.cache_system.exceptions import CacheException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

@pytest.fixture
def sample_model():
    """Create a sample PyTorch model for testing"""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU()
    )

@pytest.mark.asyncio
async def test_model_storage(sample_model):
    """Test basic model storage and retrieval"""
    cache = ModelCache(capacity_gb=1)  # 1GB cache
    model_id = "test_model"
    
    # Store model
    success = await cache.store_model(model_id, sample_model)
    assert success is True
    
    # Retrieve model
    retrieved_model = await cache.get_model(model_id)
    assert retrieved_model is not None
    
    # Check stats
    stats = cache.get_stats()
    assert stats["hits"] >= 1
    assert stats["used_memory_bytes"] > 0

@pytest.mark.asyncio
async def test_cache_eviction(sample_model):
    """Test model eviction when cache is full"""
    # Create a very small cache (1MB)
    cache = ModelCache(capacity_gb=0.001)
    
    # Get initial stats
    initial_stats = cache.get_stats()
    logging.info(f"Initial stats: {initial_stats}")
    
    # Calculate model size
    model_size = cache._get_model_size(sample_model)
    logging.info(f"Model size: {model_size} bytes")
    logging.info(f"Cache capacity: {cache.capacity_bytes} bytes")
    
    # Store models until eviction occurs
    models_stored = []
    for i in range(5):
        model_id = f"model_{i}"
        try:
            logging.info(f"\nAttempting to store model {model_id}")
            success = await cache.store_model(model_id, sample_model)
            if success:
                models_stored.append(model_id)
                logging.info(f"Successfully stored model {model_id}")
            
            # Print current stats
            current_stats = cache.get_stats()
            logging.info(f"Current stats after storing {model_id}:")
            logging.info(f"Used memory: {current_stats['used_memory_bytes']} bytes")
            logging.info(f"Evictions: {current_stats['evictions']}")
            
        except CacheException as e:
            logging.info(f"Failed to store model {model_id}: {e}")
    
    # Get final stats
    final_stats = cache.get_stats()
    logging.info(f"\nFinal stats: {final_stats}")
    logging.info(f"Models stored: {models_stored}")
    
    # Verify evictions occurred
    assert final_stats["evictions"] > 0, (
        f"Expected evictions > 0, got {final_stats['evictions']}. "
        f"Cache capacity: {cache.capacity_bytes}, "
        f"Model size: {model_size}, "
        f"Models stored: {len(models_stored)}"
    )