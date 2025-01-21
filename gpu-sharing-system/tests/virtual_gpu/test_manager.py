# tests/virtual_gpu/test_manager.py
import pytest
from unittest.mock import Mock
from src.virtual_gpu.manager import VirtualGPUManager
from src.common.exceptions import GPUError, ResourceError

@pytest.mark.asyncio
class TestVirtualGPUManager:
    async def test_initialization(self, mock_nvml):
        """Test VirtualGPUManager initialization"""
        manager = VirtualGPUManager()
        assert manager.initialized is True
        assert len(manager.gpus) == 2  # Based on mock configuration
        assert manager.mps_manager is not None

    async def test_create_virtual_gpu(self, mock_nvml):
        """Test creating virtual GPU with valid parameters"""
        manager = VirtualGPUManager()
        
        v_gpu = await manager.create_virtual_gpu(
            memory_mb=1024,
            compute_percentage=25,
            priority=1
        )

        assert v_gpu["v_gpu_id"] is not None
        assert v_gpu["gpu_id"] in [0, 1]
        assert v_gpu["allocated_memory"] == 1024
        assert v_gpu["compute_percentage"] == 25
        assert "mps_context" in v_gpu

    async def test_create_virtual_gpu_invalid_memory(self, mock_nvml):
        """Test creating virtual GPU with invalid memory"""
        manager = VirtualGPUManager()
        
        with pytest.raises(ResourceError, match="Insufficient resources"):
            await manager.create_virtual_gpu(
                memory_mb=10 * 1024,  # 10GB (exceeds capacity)
                compute_percentage=25
            )

    async def test_create_virtual_gpu_invalid_compute(self, mock_nvml):
        """Test creating virtual GPU with invalid compute percentage"""
        manager = VirtualGPUManager()
        
        with pytest.raises(ResourceError, match="Invalid compute percentage"):
            await manager.create_virtual_gpu(
                memory_mb=1024,
                compute_percentage=150  # Over 100%
            )

    async def test_find_suitable_gpu(self, mock_nvml):
        """Test finding suitable GPU for allocation"""
        manager = VirtualGPUManager()
        
        gpu_id = manager._find_suitable_gpu(1024, 25)
        assert gpu_id in [0, 1]
        
        # Test when no suitable GPU available
        manager.gpus[0].used_memory = manager.gpus[0].total_memory
        manager.gpus[1].used_memory = manager.gpus[1].total_memory
        
        gpu_id = manager._find_suitable_gpu(1024, 25)
        assert gpu_id is None

    async def test_release_virtual_gpu(self, mock_nvml):
        """Test releasing virtual GPU"""
        manager = VirtualGPUManager()
        
        v_gpu = await manager.create_virtual_gpu(
            memory_mb=1024,
            compute_percentage=25
        )
        
        success = await manager.release_virtual_gpu(v_gpu["v_gpu_id"])
        assert success is True
        
        # Test releasing non-existent GPU
        success = await manager.release_virtual_gpu("non-existent")
        assert success is False

    async def test_get_gpu_status(self, mock_nvml):
        """Test getting GPU status"""
        manager = VirtualGPUManager()
        
        status = manager.get_gpu_status(0)
        assert "gpu_id" in status
        assert "total_memory" in status
        assert "used_memory" in status
        assert "utilization" in status
        assert "active_vgpus" in status
        
        with pytest.raises(GPUError, match="Invalid GPU ID"):
            manager.get_gpu_status(999)

    async def test_cleanup(self, mock_nvml):
        """Test cleanup of resources"""
        manager = VirtualGPUManager()
        
        # Create some virtual GPUs
        v_gpu1 = await manager.create_virtual_gpu(1024, 25)
        v_gpu2 = await manager.create_virtual_gpu(1024, 25)
        
        # Cleanup should happen in __del__
        manager.__del__()
        
        # Verify cleanup
        assert manager.initialized is False