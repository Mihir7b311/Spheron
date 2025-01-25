# tests/integration/test_resource_manager.py

import pytest
import asyncio
from src.integration.resource_manager import ResourceManager, ResourceRequest, ResourceType
from src.common.exceptions import ResourceError

class TestResourceManager:
    @pytest.fixture
    async def resource_manager(self, test_config):
        manager = ResourceManager(test_config)
        await manager.initialize()
        yield manager
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_resource_allocation(self, resource_manager):
        """Test basic resource allocation"""
        request = ResourceRequest(
            resource_type=ResourceType.GPU_MEMORY,
            amount=1024,  # 1GB
            priority=1,
            owner_id="test_user"
        )
        
        allocation = await resource_manager.request_resources(request)
        assert allocation["status"] == "success"
        assert allocation["allocated_amount"] == 1024
        assert allocation["gpu_id"] is not None

    @pytest.mark.asyncio
    async def test_memory_constraints(self, resource_manager):
        """Test memory constraints are enforced"""
        # Request more than available
        request = ResourceRequest(
            resource_type=ResourceType.GPU_MEMORY,
            amount=1024 * 1024,  # Too much memory
            priority=1,
            owner_id="test_user"
        )
        
        with pytest.raises(ResourceError):
            await resource_manager.request_resources(request)

    @pytest.mark.asyncio
    async def test_multi_tenant_isolation(self, resource_manager):
        """Test resource isolation between tenants"""
        # Allocate for tenant 1
        req1 = ResourceRequest(
            resource_type=ResourceType.GPU_MEMORY,
            amount=512,
            priority=1,
            owner_id="tenant1"
        )
        alloc1 = await resource_manager.request_resources(req1)

        # Allocate for tenant 2
        req2 = ResourceRequest(
            resource_type=ResourceType.GPU_MEMORY,
            amount=512,
            priority=1,
            owner_id="tenant2"
        )
        alloc2 = await resource_manager.request_resources(req2)

        # Verify isolation
        tenant1_resources = resource_manager.get_tenant_resources("tenant1")
        tenant2_resources = resource_manager.get_tenant_resources("tenant2")
        
        assert tenant1_resources != tenant2_resources

    @pytest.mark.asyncio
    async def test_resource_oversubscription(self, resource_manager):
        """Test handling of resource oversubscription"""
        requests = [
            ResourceRequest(
                resource_type=ResourceType.GPU_COMPUTE,
                amount=40,  # 40% compute
                priority=1,
                owner_id=f"user_{i}"
            )
            for i in range(3)  # Total 120% requested
        ]
        
        # First two should succeed, third should fail
        await resource_manager.request_resources(requests[0])
        await resource_manager.request_resources(requests[1])
        with pytest.raises(ResourceError):
            await resource_manager.request_resources(requests[2])

    @pytest.mark.asyncio
    async def test_resource_tracking(self, resource_manager):
        """Test resource usage tracking"""
        request = ResourceRequest(
            resource_type=ResourceType.GPU_MEMORY,
            amount=512,
            priority=1,
            owner_id="test_user"
        )
        
        allocation = await resource_manager.request_resources(request)
        
        # Check usage stats
        usage = await resource_manager.get_resource_usage()
        assert usage[allocation["gpu_id"]]["memory_reserved"] >= 512
        assert usage[allocation["gpu_id"]]["compute_reserved"] > 0

    @pytest.mark.asyncio
    async def test_resource_release(self, resource_manager):
        """Test resource release"""
        request = ResourceRequest(
            resource_type=ResourceType.GPU_MEMORY,
            amount=512,
            priority=1,
            owner_id="test_user"
        )
        
        allocation = await resource_manager.request_resources(request)
        
        # Release resources
        success = await resource_manager.release_resources(allocation["request_id"])
        assert success

        # Verify released
        usage = await resource_manager.get_resource_usage()
        assert usage[allocation["gpu_id"]]["memory_reserved"] == 0

    @pytest.mark.asyncio
    async def test_quota_management(self, resource_manager):
        """Test resource quota management"""
        # Set quota for user
        await resource_manager.set_user_quota("test_user", {
            "memory_mb": 1024,
            "compute_percentage": 50
        })

        # Try to exceed quota
        request = ResourceRequest(
            resource_type=ResourceType.GPU_MEMORY,
            amount=2048,  # Exceeds quota
            priority=1,
            owner_id="test_user"
        )
        
        with pytest.raises(ResourceError):
            await resource_manager.request_resources(request)

    @pytest.mark.asyncio
    async def test_resource_optimization(self, resource_manager):
        """Test resource optimization"""
        # Allocate suboptimal resources
        requests = [
            ResourceRequest(
                resource_type=ResourceType.GPU_MEMORY,
                amount=256,
                priority=1,
                owner_id=f"user_{i}"
            )
            for i in range(4)
        ]
        
        for req in requests:
            await resource_manager.request_resources(req)

        # Optimize allocations
        improved = await resource_manager._optimize_resource_allocation()
        assert improved

        # Check fragmentation reduced
        usage = await resource_manager.get_resource_usage()
        for gpu_id in usage:
            assert usage[gpu_id]["fragmentation"]["fragmentation_ratio"] < 0.3