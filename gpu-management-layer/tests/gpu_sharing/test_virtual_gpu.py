# tests/gpu_sharing/test_virtual_gpu.py

import pytest
import torch
import asyncio
from src.gpu_sharing.virtual_gpu import VirtualGPUManager, VirtualGPUConfig
from src.common.exceptions import GPUError, ResourceError

class TestVirtualGPUManager:
    @pytest.fixture
    async def vgpu_manager(self):
        """Create VGPU manager instance"""
        manager = VirtualGPUManager()
        await manager.initialize()
        yield manager
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_initialization(self, vgpu_manager):
        """Test VGPU manager initialization"""
        assert vgpu_manager.initialized is True
        assert len(vgpu_manager.physical_gpus) > 0

    @pytest.mark.asyncio
    async def test_create_virtual_gpu(self, vgpu_manager):
        """Test creating a virtual GPU"""
        config = VirtualGPUConfig(
            memory_mb=1024,  # 1GB
            compute_percentage=20,
            priority=0,
            enable_mps=True
        )

        v_gpu = await vgpu_manager.create_virtual_gpu(config)
        assert v_gpu is not None
        assert "id" in v_gpu
        assert "gpu_id" in v_gpu
        assert "config" in v_gpu
        assert v_gpu["active"] is True

    @pytest.mark.asyncio
    async def test_resource_limits(self, vgpu_manager):
        """Test resource limit enforcement"""
        # Try to allocate more than available
        config = VirtualGPUConfig(
            memory_mb=1024*1024,  # Much more than available
            compute_percentage=150,  # Over 100%
            priority=0
        )

        with pytest.raises(ResourceError):
            await vgpu_manager.create_virtual_gpu(config)

    @pytest.mark.asyncio
    async def test_multiple_vgpus(self, vgpu_manager):
        """Test creating multiple virtual GPUs"""
        configs = [
            VirtualGPUConfig(memory_mb=512, compute_percentage=20),
            VirtualGPUConfig(memory_mb=512, compute_percentage=20)
        ]

        vgpus = []
        for config in configs:
            vgpu = await vgpu_manager.create_virtual_gpu(config)
            vgpus.append(vgpu)

        assert len(vgpus) == 2
        assert vgpus[0]["gpu_id"] == vgpus[1]["gpu_id"]  # Should be on same GPU

    @pytest.mark.asyncio
    async def test_release_virtual_gpu(self, vgpu_manager):
        """Test releasing a virtual GPU"""
        config = VirtualGPUConfig(memory_mb=512, compute_percentage=20)
        vgpu = await vgpu_manager.create_virtual_gpu(config)
        
        success = await vgpu_manager.release_virtual_gpu(vgpu["id"])
        assert success is True

        # Verify resources released
        gpu_id = vgpu["gpu_id"]
        gpu_info = vgpu_manager.get_physical_gpu_info(gpu_id)
        assert gpu_info["allocated_memory_mb"] == 0
        assert gpu_info["allocated_compute"] == 0

    @pytest.mark.asyncio
    async def test_model_cache_integration(self, vgpu_manager):
        """Test integration with model cache"""
        # Create vGPU
        config = VirtualGPUConfig(memory_mb=1024, compute_percentage=20)
        vgpu = await vgpu_manager.create_virtual_gpu(config)

        # Verify model cache initialized
        assert vgpu_manager.model_cache is not None
        
        # Should be able to get GPU info
        gpu_info = vgpu_manager.get_physical_gpu_info(vgpu["gpu_id"])
        assert gpu_info is not None

    @pytest.mark.asyncio
    async def test_resource_monitoring(self, vgpu_manager):
        """Test resource monitoring capabilities"""
        config = VirtualGPUConfig(memory_mb=512, compute_percentage=20)
        vgpu = await vgpu_manager.create_virtual_gpu(config)

        # Get stats
        stats = vgpu_manager.get_stats()
        assert "total_physical_gpus" in stats
        assert "total_virtual_gpus" in stats
        assert "gpu_utilization" in stats

    @pytest.mark.asyncio
    async def test_error_handling(self, vgpu_manager):
        """Test error handling scenarios"""
        # Invalid GPU ID
        with pytest.raises(GPUError):
            vgpu_manager.get_physical_gpu_info(9999)

        # Release non-existent vGPU
        success = await vgpu_manager.release_virtual_gpu("nonexistent")
        assert success is False

    @pytest.mark.asyncio
    async def test_cleanup(self, vgpu_manager):
        """Test cleanup functionality"""
        config = VirtualGPUConfig(memory_mb=512, compute_percentage=20)
        await vgpu_manager.create_virtual_gpu(config)

        # Cleanup should release all resources
        await vgpu_manager.cleanup()
        
        stats = vgpu_manager.get_stats()
        assert stats["total_virtual_gpus"] == 0