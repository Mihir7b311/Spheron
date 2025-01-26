# tests/gpu_sharing/test_space_sharing.py

import pytest
import asyncio
from src.gpu_sharing.space_sharing import (
    SpaceSharingManager, 
    MemoryPartition,
    IsolationConstraints
)
from src.common.exceptions import ResourceError

class TestSpaceSharingManager:
    @pytest.fixture
    async def space_manager(self):
        """Create space sharing manager instance"""
        manager = SpaceSharingManager()
        yield manager
        # Cleanup any active partitions
        for partition_id in list(manager.partitions.keys()):
            await manager.release_partition(partition_id)

    @pytest.mark.asyncio
    async def test_create_partition(self, space_manager):
        """Test creating memory partition"""
        partition = await space_manager.create_partition(
            gpu_id=0,
            size_mb=512,
            owner_id="test_owner",
            vgpu_id="test_vgpu"
        )

        assert partition["id"] is not None
        assert partition["size_mb"] == 512
        assert partition["owner_id"] == "test_owner"
        assert partition["gpu_id"] == 0
        assert not partition["is_fragmented"]

    @pytest.mark.asyncio
    async def test_partition_size_limits(self, space_manager):
        """Test partition size limits"""
        # Try too small partition
        with pytest.raises(ResourceError):
            await space_manager.create_partition(
                gpu_id=0,
                size_mb=space_manager.min_partition_size - 1,
                owner_id="test_owner",
                vgpu_id="test_vgpu"
            )

        # Try too large partition
        with pytest.raises(ResourceError):
            await space_manager.create_partition(
                gpu_id=0,
                size_mb=1024*1024,  # Very large
                owner_id="test_owner",
                vgpu_id="test_vgpu"
            )

    @pytest.mark.asyncio
    async def test_multiple_partitions(self, space_manager):
        """Test creating multiple partitions"""
        partitions = []
        
        # Create multiple partitions
        for i in range(3):
            partition = await space_manager.create_partition(
                gpu_id=0,
                size_mb=256,
                owner_id=f"owner_{i}",
                vgpu_id=f"vgpu_{i}"
            )
            partitions.append(partition)

        # Verify partitions
        assert len(partitions) == 3
        memory_map = space_manager.get_gpu_memory_map(0)
        assert len(memory_map) == 3

        # Check contiguous allocation
        offsets = [p["offset"] for p in sorted(memory_map, key=lambda x: x["offset"])]
        for i in range(1, len(offsets)):
            assert offsets[i] > offsets[i-1]

    @pytest.mark.asyncio
    async def test_isolation_constraints(self, space_manager):
        """Test isolation constraints"""
        constraints = IsolationConstraints(
            memory_limit=1024,
            compute_limit=20,
            priority=1,
            owner_id="test_owner",
            strict_isolation=True
        )

        # Set constraints
        success = await space_manager.set_isolation_constraints(
            vgpu_id="test_vgpu",
            constraints=constraints
        )
        assert success is True

        # Create partition with constraints
        partition = await space_manager.create_partition(
            gpu_id=0,
            size_mb=512,
            owner_id="test_owner",
            vgpu_id="test_vgpu"
        )

        assert partition is not None

    @pytest.mark.asyncio
    async def test_defragmentation(self, space_manager):
        """Test memory defragmentation"""
        # Create fragmented memory pattern
        partitions = []
        for i in range(4):
            partition = await space_manager.create_partition(
                gpu_id=0,
                size_mb=256,
                owner_id=f"owner_{i}",
                vgpu_id=f"vgpu_{i}"
            )
            partitions.append(partition)

        # Release alternate partitions
        for i in range(0, len(partitions), 2):
            await space_manager.release_partition(partitions[i]["id"])

        # Get fragmentation info
        frag_info = space_manager.get_fragmentation_info(0)
        initial_frag = frag_info["fragmentation_ratio"]

        # Perform defragmentation
        await space_manager._defragment_gpu(0)

        # Check fragmentation improved
        new_frag_info = space_manager.get_fragmentation_info(0)
        assert new_frag_info["fragmentation_ratio"] < initial_frag

    @pytest.mark.asyncio
    async def test_partition_movement(self, space_manager):
        """Test partition movement"""
        # Create initial partition
        partition = await space_manager.create_partition(
            gpu_id=0,
            size_mb=256,
            owner_id="test_owner",
            vgpu_id="test_vgpu"
        )

        initial_offset = partition["offset"]
        
        # Move partition
        new_offset = initial_offset + 512
        await space_manager._move_partition(
            self.partitions[partition["id"]],
            new_offset
        )

        # Verify move
        moved_partition = space_manager._get_partition_info(partition["id"])
        assert moved_partition["offset"] == new_offset
        assert not moved_partition["is_fragmented"]

    @pytest.mark.asyncio
    async def test_memory_map(self, space_manager):
        """Test GPU memory map"""
        # Create some partitions
        await space_manager.create_partition(
            gpu_id=0,
            size_mb=256,
            owner_id="owner1",
            vgpu_id="vgpu1"
        )

        await space_manager.create_partition(
            gpu_id=0,
            size_mb=512,
            owner_id="owner2",
            vgpu_id="vgpu2"
        )

        # Get memory map
        memory_map = space_manager.get_gpu_memory_map(0)
        
        assert len(memory_map) == 2
        assert sum(p["size_mb"] for p in memory_map) == 768
        
        # Verify map is sorted by offset
        offsets = [p["offset"] for p in memory_map]
        assert offsets == sorted(offsets)

    @pytest.mark.asyncio
    async def test_resource_tracking(self, space_manager):
        """Test resource usage tracking"""
        partition = await space_manager.create_partition(
            gpu_id=0,
            size_mb=512,
            owner_id="test_owner",
            vgpu_id="test_vgpu"
        )

        # Get stats
        stats = space_manager.get_stats()
        
        assert stats["total_partitions"] == 1
        assert f"gpu_partitions" in stats
        assert stats["total_allocated_memory"] == 512
        assert "fragmentation" in stats

    @pytest.mark.asyncio
    async def test_cleanup(self, space_manager):
        """Test cleanup functionality"""
        # Create partition
        partition = await space_manager.create_partition(
            gpu_id=0,
            size_mb=256,
            owner_id="test_owner",
            vgpu_id="test_vgpu"
        )

        # Release partition
        success = await space_manager.release_partition(partition["id"])
        assert success is True
        
        # Verify cleanup
        assert partition["id"] not in space_manager.partitions
        assert len(space_manager.gpu_partitions[0]) == 0

    @pytest.mark.asyncio
    async def test_error_handling(self, space_manager):
        """Test error handling scenarios"""
        # Invalid GPU ID
        with pytest.raises(ResourceError):
            await space_manager.create_partition(
                gpu_id=9999,
                size_mb=256,
                owner_id="test_owner",
                vgpu_id="test_vgpu"
            )

        # Release non-existent partition
        success = await space_manager.release_partition("nonexistent")
        assert not success