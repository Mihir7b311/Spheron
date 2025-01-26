# tests/integration/test_coordinator.py
import torch
import pytest
import asyncio
from src.integration.coordinator import SystemCoordinator, SystemState
from src.common.exceptions import CoordinationError
from src.common.metrics import MetricsCollector
from src.common.monitoring import ResourceMonitor

class TestSystemCoordinator:
    @pytest.fixture
    async def coordinator(self, test_config):
        """Create system coordinator instance"""
        coordinator = SystemCoordinator(config=test_config)
        await coordinator.initialize()
        yield coordinator
        await coordinator.shutdown()

    @pytest.mark.asyncio
    async def test_system_initialization(self, coordinator):
        """Test system initialization"""
        assert coordinator.state == SystemState.RUNNING
        assert coordinator.metrics is not None
        assert coordinator.monitor is not None
        assert len(coordinator.components) > 0
        
        # Check all components initialized
        status = coordinator.get_system_status()
        assert all(
            comp["state"] == "initialized" 
            for comp in status["components"].values()
        )

    @pytest.mark.asyncio
    async def test_model_execution(self, coordinator, sample_model):
        """Test model execution through coordinator"""
        task = {
            "model_id": "test_model",
            "model_data": sample_model.state_dict(),
            "input_data": {"x": torch.randn(1, 10)}
        }

        result = await coordinator.execute_model(**task)
        assert result["status"] == "success"
        assert "results" in result
        assert "metrics" in result

    @pytest.mark.asyncio
    async def test_component_coordination(self, coordinator):
        """Test coordination between components"""
        # Verify cache and execution engine coordination
        cache_status = coordinator.get_component_status("model_cache")
        executor_status = coordinator.get_component_status("executor")
        
        assert cache_status["healthy"]
        assert executor_status["healthy"]

        # Verify GPU sharing coordination
        vgpu_status = coordinator.get_component_status("vgpu_manager")
        scheduler_status = coordinator.get_component_status("scheduler")
        
        assert vgpu_status["healthy"]
        assert scheduler_status["healthy"]

    @pytest.mark.asyncio
    async def test_health_check(self, coordinator):
        """Test system health checking"""
        health = await coordinator.health_check()
        assert health["healthy"]
        assert "components" in health
        
        # Should handle component failures
        coordinator.component_status["test_component"] = {
            "state": "error",
            "healthy": False,
            "error": "Test error"
        }
        
        health = await coordinator.health_check()
        assert not health["healthy"]

    @pytest.mark.asyncio
    async def test_resource_management(self, coordinator):
        """Test resource management through coordinator"""
        # Get initial resource usage
        initial_usage = await coordinator.get_resource_usage()
        
        # Execute task to allocate resources
        task = {
            "model_id": "test_model",
            "model_data": torch.nn.Linear(10, 10).state_dict(),
            "input_data": {"x": torch.randn(1, 10)}
        }
        await coordinator.execute_model(**task)
        
        # Check resource usage changed
        current_usage = await coordinator.get_resource_usage()
        assert current_usage != initial_usage

    @pytest.mark.asyncio
    async def test_error_handling(self, coordinator):
        """Test coordinator error handling"""
        # Invalid model execution
        with pytest.raises(CoordinationError):
            await coordinator.execute_model(
                model_id="invalid",
                model_data=None,
                input_data={}
            )

        # System should remain healthy
        health = await coordinator.health_check()
        assert health["healthy"]

    @pytest.mark.asyncio
    async def test_system_pause_resume(self, coordinator):
        """Test system pause/resume functionality"""
        await coordinator.pause_system()
        assert coordinator.state == SystemState.PAUSED
        
        await coordinator.resume_system()
        assert coordinator.state == SystemState.RUNNING

    @pytest.mark.asyncio
    async def test_metrics_collection(self, coordinator):
        """Test metrics collection"""
        # Execute some tasks
        for _ in range(3):
            task = {
                "model_id": f"model_{_}",
                "model_data": torch.nn.Linear(10, 10).state_dict(),
                "input_data": {"x": torch.randn(1, 10)}
            }
            await coordinator.execute_model(**task)

        # Check metrics
        metrics = coordinator.metrics_collector.get_recent_metrics(10)
        assert len(metrics) > 0

    @pytest.mark.asyncio 
    async def test_recovery_mechanism(self, coordinator):
        """Test system recovery mechanisms"""
        # Simulate component failure
        failed_component = "model_cache"
        coordinator.component_status[failed_component]["healthy"] = False
        
        # Attempt recovery
        recovery_success = await coordinator._handle_component_failure(failed_component)
        assert recovery_success
        
        # Check component restored
        assert coordinator.component_status[failed_component]["healthy"]

    @pytest.mark.asyncio
    async def test_shutdown(self, coordinator):
        """Test system shutdown"""
        await coordinator.shutdown()
        assert coordinator.state == SystemState.SHUTDOWN
        
        # Verify cleanup
        assert len(coordinator.components) == 0
        assert len(coordinator.component_status) == 0

        # Metrics and monitoring should be stopped
        assert coordinator.monitoring_task is None
        assert coordinator.metrics_collection_task is None