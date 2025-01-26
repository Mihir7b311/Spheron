# tests/cache_system/test_model_cache.py

import pytest
import torch
import logging
from src.cache_system.model_cache import ModelCache
from src.cache_system.exceptions import CacheException

@pytest.fixture
def sample_model():
    """Create a simple test model"""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU()
    )

class TestModelCache:
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test model cache initialization"""
        cache = ModelCache(gpu_id=0, capacity_gb=1.0)
        assert cache.gpu_id == 0
        assert cache.capacity_bytes == int(1.0 * 1024 * 1024 * 1024)
        assert cache.used_memory == 0
        assert cache.max_items == 10

    @pytest.mark.asyncio 
    async def test_model_storage(self, sample_model):
        """Test basic model storage and retrieval"""
        cache = ModelCache(gpu_id=0, capacity_gb=1.0)
        model_id = "test_model"

        # Store model
        success = await cache.store_model(model_id, sample_model)
        assert success is True

        # Retrieve model
        retrieved_model = await cache.get_model(model_id)
        assert retrieved_model is not None
        assert isinstance(retrieved_model, torch.nn.Module)

        # Check stats
        stats = cache.get_stats()
        assert stats["num_models"] == 1
        assert stats["used_memory_bytes"] > 0

    @pytest.mark.asyncio
    async def test_cache_eviction(self, sample_model):
        """Test eviction when cache is full"""
        cache = ModelCache(gpu_id=0, capacity_gb=0.001)  # Very small cache
        
        # Add multiple models to force eviction
        models_stored = []
        for i in range(cache.max_items + 1):
            model_id = f"model_{i}"
            try:
                success = await cache.store_model(model_id, sample_model)
                if success:
                    models_stored.append(model_id)
            except CacheException:
                pass

        # Verify eviction
        stats = cache.get_stats()
        assert stats["num_models"] <= cache.max_items
        assert len(models_stored) > 0

    @pytest.mark.asyncio
    async def test_model_size_tracking(self, sample_model):
        """Test model size tracking"""
        cache = ModelCache(gpu_id=0, capacity_gb=1.0)
        model_id = "test_model"

        # Get initial memory
        initial_memory = cache.used_memory

        # Store model
        await cache.store_model(model_id, sample_model)

        # Check memory increased
        assert cache.used_memory > initial_memory
        assert cache.model_sizes[model_id] > 0

    @pytest.mark.asyncio
    async def test_cache_stats(self, sample_model):
        """Test cache statistics"""
        cache = ModelCache(gpu_id=0, capacity_gb=1.0)
        model_id = "test_model"

        # Store and access model
        await cache.store_model(model_id, sample_model)
        await cache.get_model(model_id)  # Cache hit
        await cache.get_model("nonexistent")  # Cache miss

        stats = cache.get_stats()
        assert "used_memory_bytes" in stats
        assert "total_memory_bytes" in stats
        assert "memory_utilization" in stats
        assert "num_models" in stats

    # Additional tests for test_model_cache.py

    @pytest.mark.asyncio
    async def test_concurrent_model_access(self, sample_model):
        """Test concurrent model access"""
        cache = ModelCache(gpu_id=0, capacity_gb=1.0)
        model_id = "test_model"
        
        # Store model
        await cache.store_model(model_id, sample_model)
        
        # Simulate concurrent access
        async def access_model():
            return await cache.get_model(model_id)
            
        # Create multiple concurrent access tasks
        tasks = [access_model() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert all(result is not None for result in results)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_memory_cleanup_on_error(self, sample_model):
        """Test memory cleanup when errors occur"""
        cache = ModelCache(gpu_id=0, capacity_gb=1.0)
        initial_memory = cache.used_memory
        
        # Simulate error during model store
        with pytest.raises(Exception):
            await cache.store_model("bad_model", None)
        
        # Verify memory cleaned up
        assert cache.used_memory == initial_memory
        
    @pytest.mark.asyncio
    async def test_multi_gpu_cache_coherence(self):
        """Test cache coherence across multiple GPUs"""
        if torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 GPUs")
            
        cache1 = ModelCache(gpu_id=0, capacity_gb=1.0)
        cache2 = ModelCache(gpu_id=1, capacity_gb=1.0)
        
        model = sample_model()
        model_id = "test_model"
        
        # Store on first GPU
        await cache1.store_model(model_id, model)
        
        # Should be accessible on second GPU
        retrieved = await cache2.get_model(model_id) 
        assert retrieved is not None