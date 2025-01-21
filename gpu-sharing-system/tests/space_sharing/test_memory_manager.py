# tests/space_sharing/test_memory_manager.py
import pytest
from src.space_sharing.memory_manager import MemoryManager, MemoryPartition
from src.common.exceptions import MemoryError

@pytest.mark.asyncio
class TestMemoryManager:
    async def test_initialization(self, mock_nvml):
        """Test memory manager initialization"""
        manager = MemoryManager(gpu_id=0, total_memory=8*1024)  # 8GB
        assert manager.gpu_id == 0
        assert manager.total_memory == 8*1024
        assert len(manager.partitions) == 0
        assert manager.min_partition_size == 256

    async def test_create_partition(self, mock_nvml):
        """Test creating memory partition"""
        manager = MemoryManager(gpu_id=0, total_memory=8*1024)
        
        partition = await manager.create_partition(
            size_mb=1024,
            owner_id="test_owner"
        )
        
        assert partition["partition_id"] is not None
        assert partition["size_mb"] == 1024
        assert partition["owner_id"] == "test_owner"
        assert "offset" in partition

    async def test_minimum_partition_size(self, mock_nvml):
        """Test minimum partition size enforcement"""
        manager = MemoryManager(gpu_id=0, total_memory=8*1024)
        
        with pytest.raises(MemoryError, match="must be at least"):
            await manager.create_partition(
                size_mb=100,  # Below minimum
                owner_id="test_owner"
            )

    async def test_insufficient_memory(self, mock_nvml):
        """Test handling insufficient memory"""
        manager = MemoryManager(gpu_id=0, total_memory=8*1024)
        
        with pytest.raises(MemoryError, match="Insufficient memory"):
            await manager.create_partition(
                size_mb=10*1024,  # Exceeds total memory
                owner_id="test_owner"
            )

    async def test_fragmentation_handling(self, mock_nvml):
        """Test memory fragmentation handling"""
        manager = MemoryManager(gpu_id=0, total_memory=8*1024)
        
        # Create multiple partitions
        partitions = []
        for i in range(3):
            partition = await manager.create_partition(
                size_mb=1024,
                owner_id=f"owner_{i}"
            )
            partitions.append(partition)
        
        # Release middle partition
        await manager.release_partition(partitions[1]["partition_id"])
        
        # Try to allocate in fragmented space
        new_partition = await manager.create_partition(
            size_mb=512,
            owner_id="new_owner"
        )
        
        assert new_partition is not None
        assert new_partition["size_mb"] == 512

    async def test_defragmentation(self, mock_nvml):
        """Test memory defragmentation"""
        manager = MemoryManager(gpu_id=0, total_memory=8*1024)
        
        # Fill memory with fragmentated allocations
        partitions = []
        for i in range(4):
            partition = await manager.create_partition(
                size_mb=1024,
                owner_id=f"owner_{i}"
            )
            partitions.append(partition)
        
        # Release alternate partitions
        await manager.release_partition(partitions[1]["partition_id"])
        await manager.release_partition(partitions[3]["partition_id"])
        
        # Force defragmentation
        await manager._defragment()
        
        # Verify contiguous space
        new_partition = await manager.create_partition(
            size_mb=2048,  # Should fit in defragmented space
            owner_id="new_owner"
        )
        
        assert new_partition is not None
        assert new_partition["size_mb"] == 2048

    async def test_partition_release(self, mock_nvml):
        """Test releasing memory partition"""
        manager = MemoryManager(gpu_id=0, total_memory=8*1024)
        
        partition = await manager.create_partition(
            size_mb=1024,
            owner_id="test_owner"
        )
        
        success = await manager.release_partition(partition["partition_id"])
        assert success is True
        
        # Test releasing non-existent partition
        success = await manager.release_partition("non-existent")
        assert success is False

    async def test_get_memory_map(self, mock_nvml):
        """Test getting memory map"""
        manager = MemoryManager(gpu_id=0, total_memory=8*1024)
        
        # Create several partitions
        for i in range(3):
            await manager.create_partition(
                size_mb=1024,
                owner_id=f"owner_{i}"
            )
        
        memory_map = manager.get_memory_map()
        assert len(memory_map) == 3
        for partition in memory_map:
            assert "id" in partition
            assert "size_mb" in partition
            assert "offset" in partition
            assert "in_use" in partition
            assert "owner_id" in partition

    async def test_get_stats(self, mock_nvml):
        """Test getting memory statistics"""
        manager = MemoryManager(gpu_id=0, total_memory=8*1024)
        
        await manager.create_partition(
            size_mb=1024,
            owner_id="test_owner"
        )
        
        stats = manager.get_stats()
        assert "total_memory" in stats
        assert "used_memory" in stats
        assert "free_memory" in stats
        assert "num_partitions" in stats
        assert "fragmentation" in stats
        assert "largest_free_block" in stats