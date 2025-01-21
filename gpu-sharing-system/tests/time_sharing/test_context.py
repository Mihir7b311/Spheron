# tests/time_sharing/test_context.py
import pytest
import asyncio
from src.time_sharing.context import ContextManager, Context
from src.common.exceptions import ContextError

@pytest.mark.asyncio
class TestContextManager:
    async def test_initialization(self):
        """Test context manager initialization"""
        manager = ContextManager()
        assert manager.active_context is None
        assert len(manager.contexts) == 0
        assert manager.switch_count == 0
        assert manager.total_switch_time == 0

    async def test_save_context(self, mocker):
        """Test saving GPU context"""
        manager = ContextManager()
        
        # Mock GPU state capture
        mocker.patch.object(
            manager, 
            '_capture_gpu_state',
            return_value={"memory": "state"}
        )
        
        # Create a test context
        manager.contexts["test_proc"] = Context(
            id="ctx1",
            process_id="test_proc",
            gpu_id=0,
            compute_percentage=25,
            state={}
        )
        
        # Save context
        result = await manager.save_context("test_proc")
        assert result["context_id"] == "ctx1"
        assert "state" in result
        assert result["state"]["memory"] == "state"

    async def test_save_nonexistent_context(self):
        """Test saving non-existent context"""
        manager = ContextManager()
        
        with pytest.raises(ContextError, match="No context found"):
            await manager.save_context("non-existent")

    async def test_restore_context(self, mocker):
        """Test restoring GPU context"""
        manager = ContextManager()
        
        # Mock GPU state restoration
        mocker.patch.object(
            manager,
            '_restore_gpu_state',
            return_value=None
        )
        
        # Create a test context
        manager.contexts["test_proc"] = Context(
            id="ctx1",
            process_id="test_proc",
            gpu_id=0,
            compute_percentage=25,
            state={"memory": "state"}
        )
        
        # Restore context
        success = await manager.restore_context("test_proc")
        assert success is True
        assert manager.active_context == "test_proc"
        assert manager.switch_count == 1
        assert manager.total_switch_time > 0

    async def test_restore_nonexistent_context(self):
        """Test restoring non-existent context"""
        manager = ContextManager()
        
        with pytest.raises(ContextError, match="No context found"):
            await manager.restore_context("non-existent")

    async def test_switch_metrics(self):
        """Test context switch metrics"""
        manager = ContextManager()
        
        # Create test contexts
        manager.contexts["proc1"] = Context(
            id="ctx1",
            process_id="proc1",
            gpu_id=0,
            compute_percentage=25,
            state={}
        )
        manager.contexts["proc2"] = Context(
            id="ctx2",
            process_id="proc2",
            gpu_id=0,
            compute_percentage=25,
            state={}
        )
        
        # Perform multiple switches
        for _ in range(3):
            await manager.restore_context("proc1")
            await manager.restore_context("proc2")
        
        stats = manager.get_switch_stats()
        assert stats["total_switches"] == 6
        assert stats["total_switch_time_ms"] > 0
        assert stats["average_switch_time_ms"] > 0
        assert stats["active_context"] == "proc2"
        assert stats["total_contexts"] == 2

    async def test_capture_restore_gpu_state(self, mocker):
        """Test GPU state capture and restoration"""
        manager = ContextManager()
        
        # Mock GPU state operations
        test_state = {
            "memory": "state",
            "registers": "state",
            "compute": "state"
        }
        mocker.patch.object(
            manager,
            '_capture_gpu_state',
            return_value=test_state
        )
        mocker.patch.object(
            manager,
            '_restore_gpu_state',
            return_value=None
        )
        
        # Create context and perform state operations
        manager.contexts["test_proc"] = Context(
            id="ctx1",
            process_id="test_proc",
            gpu_id=0,
            compute_percentage=25,
            state={}
        )
        
        # Save and restore state
        saved_state = await manager.save_context("test_proc")
        assert saved_state["state"] == test_state
        
        success = await manager.restore_context("test_proc")
        assert success is True