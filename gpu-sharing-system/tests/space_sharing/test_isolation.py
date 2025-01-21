# tests/space_sharing/test_isolation.py
import pytest
from src.space_sharing.isolation import ResourceIsolation, IsolationConstraints
from src.common.exceptions import IsolationError

@pytest.mark.asyncio
class TestResourceIsolation:
    async def test_initialization(self):
        """Test resource isolation initialization"""
        isolation = ResourceIsolation()
        assert len(isolation.constraints) == 0
        assert len(isolation.active_containers) == 0

    async def test_setup_isolation(self):
        """Test setting up resource isolation"""
        isolation = ResourceIsolation()
        
        constraints = IsolationConstraints(
            memory_limit=1024,
            compute_limit=25,
            priority=1,
            owner_id="test_owner"
        )
        
        container = await isolation.setup_isolation(
            "container1",
            constraints
        )
        
        assert container["id"] == "container1"
        assert container["memory_limit"] == 1024
        assert container["compute_limit"] == 25
        assert container["priority"] == 1
        assert container["owner_id"] == "test_owner"
        assert container["active"] is True

    async def test_duplicate_container(self):
        """Test handling duplicate container setup"""
        isolation = ResourceIsolation()
        
        constraints = IsolationConstraints(
            memory_limit=1024,
            compute_limit=25,
            priority=1,
            owner_id="test_owner"
        )
        
        await isolation.setup_isolation("container1", constraints)
        
        with pytest.raises(IsolationError, match="already exists"):
            await isolation.setup_isolation("container1", constraints)

    async def test_remove_isolation(self):
        """Test removing resource isolation"""
        isolation = ResourceIsolation()
        
        constraints = IsolationConstraints(
            memory_limit=1024,
            compute_limit=25,
            priority=1,
            owner_id="test_owner"
        )
        
        await isolation.setup_isolation("container1", constraints)
        success = await isolation.remove_isolation("container1")
        
        assert success is True
        assert "container1" not in isolation.constraints
        assert "container1" not in isolation.active_containers

    async def test_check_violation(self):
        """Test checking resource violations"""
        isolation = ResourceIsolation()
        
        constraints = IsolationConstraints(
            memory_limit=1024,
            compute_limit=25,
            priority=1,
            owner_id="test_owner"
        )
        
        await isolation.setup_isolation("container1", constraints)
        
        # Test within limits
        violation = isolation.check_violation(
            "container1",
            memory_usage=512,
            compute_usage=20
        )
        assert violation is False
        
        # Test exceeding limits
        violation = isolation.check_violation(
            "container1",
            memory_usage=2048,  # Exceeds limit
            compute_usage=30    # Exceeds limit
        )
        assert violation is True

    async def test_get_isolation_info(self):
        """Test getting isolation information"""
        isolation = ResourceIsolation()
        
        constraints = IsolationConstraints(
            memory_limit=1024,
            compute_limit=25,
            priority=1,
            owner_id="test_owner"
        )
        
        await isolation.setup_isolation("container1", constraints)
        
        info = isolation.get_isolation_info("container1")
        assert info is not None
        assert info["id"] == "container1"
        assert info["memory_limit"] == 1024
        assert info["compute_limit"] == 25
        
        # Test non-existent container
        info = isolation.get_isolation_info("non-existent")
        assert info is None

    async def test_list_active_containers(self):
        """Test listing active containers"""
        isolation = ResourceIsolation()
        
        # Create multiple containers
        for i in range(3):
            constraints = IsolationConstraints(
                memory_limit=1024,
                compute_limit=25,
                priority=i,
                owner_id=f"owner_{i}"
            )
            await isolation.setup_isolation(f"container_{i}", constraints)
        
        containers = isolation.list_active_containers()
        assert len(containers) == 3
        for container in containers:
            assert "id" in container
            assert "memory_limit" in container
            assert "compute_limit" in container
            assert "priority" in container
            assert "owner_id" in container
            assert "active" in container

    async def test_get_stats(self):
        """Test getting isolation statistics"""
        isolation = ResourceIsolation()
        
        # Create multiple containers
        for i in range(3):
            constraints = IsolationConstraints(
                memory_limit=1024,
                compute_limit=25,
                priority=i,
                owner_id=f"owner_{i}"
            )
            await isolation.setup_isolation(f"container_{i}", constraints)
        
        stats = isolation.get_stats()
        assert stats["active_containers"] == 3
        assert stats["total_memory_allocated"] == 3 * 1024
        assert stats["total_compute_allocated"] == 3 * 25