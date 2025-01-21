# tests/virtual_gpu/test_mps.py
import pytest
from src.virtual_gpu.mps_manager import MPSManager
from src.common.exceptions import MPSError

@pytest.mark.asyncio
class TestMPSManager:
    async def test_initialization(self):
        """Test MPS Manager initialization"""
        manager = MPSManager()
        assert len(manager.active_gpus) == 0
        assert len(manager.contexts) == 0

    async def test_setup_gpu(self):
        """Test setting up GPU for MPS"""
        manager = MPSManager()
        
        success = await manager.setup_gpu(0)
        assert success is True
        assert 0 in manager.active_gpus
        
        # Test setting up same GPU again
        success = await manager.setup_gpu(0)
        assert success is True  # Should return True for already setup GPU

    async def test_create_context(self):
        """Test creating MPS context"""
        manager = MPSManager()
        await manager.setup_gpu(0)
        
        context = await manager.create_context(
            gpu_id=0,
            compute_percentage=25
        )
        
        assert context["id"] is not None
        assert context["gpu_id"] == 0
        assert context["compute_percentage"] == 25

    async def test_create_context_invalid_gpu(self):
        """Test creating context on invalid GPU"""
        manager = MPSManager()
        
        with pytest.raises(MPSError, match="MPS not active"):
            await manager.create_context(999, 25)

    async def test_create_context_invalid_percentage(self):
        """Test creating context with invalid compute percentage"""
        manager = MPSManager()
        await manager.setup_gpu(0)
        
        with pytest.raises(MPSError):
            await manager.create_context(0, 150)  # Over 100%

    async def test_release_context(self):
        """Test releasing MPS context"""
        manager = MPSManager()
        await manager.setup_gpu(0)
        
        context = await manager.create_context(0, 25)
        success = await manager.release_context(context["id"])
        assert success is True
        
        # Test releasing non-existent context
        success = await manager.release_context("non-existent")
        assert success is False

    async def test_cleanup_gpu(self):
        """Test cleaning up GPU"""
        manager = MPSManager()
        await manager.setup_gpu(0)
        
        # Create some contexts
        context1 = await manager.create_context(0, 25)
        context2 = await manager.create_context(0, 25)
        
        success = await manager.cleanup_gpu(0)
        assert success is True
        assert 0 not in manager.active_gpus
        assert len(manager.contexts) == 0

    async def test_cleanup_all(self):
        """Test cleaning up all GPUs"""
        manager = MPSManager()
        
        # Setup multiple GPUs
        await manager.setup_gpu(0)
        await manager.setup_gpu(1)
        
        # Create contexts on both GPUs
        await manager.create_context(0, 25)
        await manager.create_context(1, 25)
        
        await manager.cleanup_all()
        assert len(manager.active_gpus) == 0
        assert len(manager.contexts) == 0

    @pytest.mark.asyncio
    async def test_run_command_failure(self):
        """Test handling of failed commands"""
        manager = MPSManager()
        
        with pytest.raises(MPSError, match="Command failed"):
            await manager._run_command("invalid_command")