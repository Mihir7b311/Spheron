# tests/virtual_gpu/test_resource_pool.py
import pytest
from src.virtual_gpu.resource_pool import ResourcePool
from src.common.exceptions import ResourceError

@pytest.mark.asyncio
class TestResourcePool:
    async def test_initialization(self, mock_nvml):
        """Test ResourcePool initialization"""
        pool = ResourcePool(
            gpu_id=0,
            total_memory=8 * 1024 * 1024 * 1024,  # 8GB
            handle=mock_nvml.return_value
        )
        
        assert pool.gpu_id == 0
        assert pool.used_memory == 0
        assert pool.compute_allocated == 0
        assert len(pool.allocations) == 0

    async def test_allocate_resources(self, mock_nvml):
        """Test allocating resources"""
        pool = ResourcePool(
            gpu_id=0,
            total_memory=8 * 1024 * 1024 * 1024,
            handle=mock_nvml.return_value
        )
        
        allocation = await pool.allocate_resources(
            memory_mb=1024,
            compute_percentage=25,
            priority=1
        )
        
        assert allocation["id"] is not None
        assert allocation["memory_mb"] == 1024
        assert allocation["compute_percentage"] == 25
        assert allocation["priority"] == 1

    async def test_allocate_invalid_memory(self, mock_nvml):
        """Test allocating invalid memory amount"""
        pool = ResourcePool(
            gpu_id=0,
            total_memory=8 * 1024 * 1024 * 1024,
            handle=mock_nvml.return_value
        )
        
        with pytest.raises(ResourceError):
            await pool.allocate_resources(
                memory_mb=10 * 1024,  # 10GB > 8GB total
                compute_percentage=25
            )

    async def test_allocate_invalid_compute(self, mock_nvml):
        """Test allocating invalid compute percentage"""
        pool = ResourcePool(
            gpu_id=0,
            total_memory=8 * 1024 * 1024 * 1024,
            handle=mock_nvml.return_value
        )
        
        with pytest.raises(ResourceError):
            await pool.allocate_resources(
                memory_mb=1024,
                compute_percentage=150  # > 100%
            )

    async def test_release_resources(self, mock_nvml):
        """Test releasing resources"""
        pool = ResourcePool(
            gpu_id=0,
            total_memory=8 * 1024 * 1024 * 1024,
            handle=mock_nvml.return_value
        )
        
        allocation = await pool.allocate_resources(1024, 25)
        success = await pool.release_resources(allocation["id"])
        
        assert success is True
        assert pool.used_memory == 0
        assert pool.compute_allocated == 0
        
        # Test releasing non-existent allocation
        success = await pool.release_resources("non-existent")
        assert success is False

    async def test_get_utilization(self, mock_nvml):
        """Test getting GPU utilization"""
        pool = ResourcePool(
            gpu_id=0,
            total_memory=8 * 1024 * 1024 * 1024,
            handle=mock_nvml.return_value
        )
        
        utilization = pool.get_utilization()
        assert isinstance(utilization, float)
        assert 0 <= utilization <= 100

    async def test_get_memory_usage(self, mock_nvml):
        """Test getting memory usage"""
        pool = ResourcePool(
            gpu_id=0,
            total_memory=8 * 1024 * 1024 * 1024,
            handle=mock_nvml.return_value
        )
        
        memory_usage = pool.get_memory_usage()
        assert isinstance(memory_usage, float)
        assert 0 <= memory_usage <= 100

    async def test_get_stats(self, mock_nvml):
        """Test getting resource pool statistics"""
        pool = ResourcePool(
            gpu_id=0,
            total_memory=8 * 1024 * 1024 * 1024,
            handle=mock_nvml.return_value
        )
        
        stats = pool.get_stats()
        assert "gpu_id" in stats
        assert "total_memory" in stats
        assert "used_memory" in stats
        assert "available_memory" in stats
        assert "compute_allocated" in stats
        assert "compute_available" in stats
        assert "utilization" in stats
        assert "active_allocations" in stats