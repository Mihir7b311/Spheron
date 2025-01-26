# tests/integration/test_scheduler.py

import pytest
import asyncio
import torch
from src.integration.scheduler import IntegratedScheduler, TaskRequest, SchedulingPolicy
from src.common.exceptions import SchedulerError

class TestIntegratedScheduler:
    @pytest.fixture
    async def scheduler(self, test_config):
        """Create integrated scheduler instance"""
        scheduler = IntegratedScheduler(test_config)
        await scheduler.initialize()
        yield scheduler
        await scheduler.cleanup()

    @pytest.mark.asyncio
    async def test_locality_aware_scheduling(self, scheduler):
        """Test locality-aware scheduling decisions"""
        # Create task with cached model
        model_id = "test_model"
        cached_gpu_id = 0
        
        # Pre-cache model
        scheduler.model_locations[model_id] = [cached_gpu_id]
        
        task = TaskRequest(
            task_id="locality_test",
            model_id=model_id,
            priority=1,
            gpu_preference=None
        )

        decision = await scheduler.get_scheduling_decision(task)
        assert decision.gpu_id == cached_gpu_id
        assert decision.cache_hit is True
        assert decision.policy_applied == SchedulingPolicy.LOCALITY_AWARE

    @pytest.mark.asyncio
    async def test_load_balancing(self, scheduler):
        """Test load balancing scheduling"""
        # Create multiple tasks
        tasks = [
            TaskRequest(
                task_id=f"task_{i}",
                model_id=f"model_{i}",
                priority=1
            )
            for i in range(5)
        ]

        decisions = []
        for task in tasks:
            decision = await scheduler.get_scheduling_decision(task)
            decisions.append(decision)

        # Verify load distribution
        gpu_loads = {}
        for decision in decisions:
            gpu_loads[decision.gpu_id] = gpu_loads.get(decision.gpu_id, 0) + 1

        # Check no GPU is overloaded
        max_tasks_per_gpu = max(gpu_loads.values())
        min_tasks_per_gpu = min(gpu_loads.values())
        assert max_tasks_per_gpu - min_tasks_per_gpu <= 1

    @pytest.mark.asyncio
    async def test_priority_scheduling(self, scheduler):
        """Test priority-based scheduling"""
        high_priority = TaskRequest(
            task_id="high_priority",
            model_id="model1",
            priority=2
        )
        
        low_priority = TaskRequest(
            task_id="low_priority", 
            model_id="model2",
            priority=0
        )

        await scheduler.submit_task(high_priority)
        await scheduler.submit_task(low_priority)

        # First scheduled task should be high priority
        next_task = await scheduler.schedule_next()
        assert next_task.task_id == "high_priority"

    @pytest.mark.asyncio
    async def test_batch_scheduling(self, scheduler):
        """Test batch scheduling capabilities"""
        # Create batch of similar tasks
        batch_tasks = [
            TaskRequest(
                task_id=f"batch_task_{i}",
                model_id="common_model",
                priority=1,
                batch_key="batch_1"
            )
            for i in range(scheduler.batch_size)
        ]

        for task in batch_tasks:
            await scheduler.submit_task(task)

        # Schedule batch
        decisions = await scheduler._schedule_batch("batch_1")
        
        # Verify batch scheduling
        assert len(decisions) == len(batch_tasks)
        assert all(d.gpu_id == decisions[0].gpu_id for d in decisions)
        assert all(d.policy_applied == SchedulingPolicy.BATCH_OPTIMIZED for d in decisions)

    @pytest.mark.asyncio
    async def test_out_of_order_scheduling(self, scheduler):
        """Test out-of-order scheduling"""
        # Create tasks with cached and non-cached models
        cached_task = TaskRequest(
            task_id="cached_task",
            model_id="cached_model",
            priority=1
        )
        scheduler.model_locations["cached_model"] = [0]

        normal_task = TaskRequest(
            task_id="normal_task",
            model_id="new_model",
            priority=1
        )

        # Submit normal task first
        await scheduler.submit_task(normal_task)
        await scheduler.submit_task(cached_task)

        # With O3 scheduling, cached task should be scheduled first
        next_task = await scheduler.schedule_next()
        assert next_task.task_id == "cached_task"

    @pytest.mark.asyncio
    async def test_hybrid_scheduling(self, scheduler):
        """Test hybrid scheduling policies"""
        # Set hybrid scheduling policy
        scheduler.policy = SchedulingPolicy.HYBRID
        
        task = TaskRequest(
            task_id="hybrid_test",
            model_id="test_model",
            priority=1
        )

        decision = await scheduler.get_scheduling_decision(task)
        assert decision.policy_applied == SchedulingPolicy.HYBRID
        
        # Verify both locality and load are considered
        assert hasattr(decision, "locality_score")
        assert hasattr(decision, "load_score")

    @pytest.mark.asyncio
    async def test_gpu_preference(self, scheduler):
        """Test GPU preference handling"""
        task = TaskRequest(
            task_id="gpu_pref_test",
            model_id="test_model",
            priority=1,
            gpu_preference=0
        )

        decision = await scheduler.get_scheduling_decision(task)
        assert decision.gpu_id == task.gpu_preference

    @pytest.mark.asyncio
    async def test_scheduling_constraints(self, scheduler):
        """Test scheduling constraints enforcement"""
        # Create task with constraints
        task = TaskRequest(
            task_id="constraint_test",
            model_id="test_model",
            priority=1,
            constraints={
                "min_memory": 1024,
                "min_compute": 20,
                "required_features": ["tensor_cores"]
            }
        )

        # Should select GPU meeting constraints
        decision = await scheduler.get_scheduling_decision(task)
        gpu_info = scheduler.gpu_info[decision.gpu_id]
        assert gpu_info["available_memory"] >= task.constraints["min_memory"]
        assert gpu_info["available_compute"] >= task.constraints["min_compute"]

    @pytest.mark.asyncio
    async def test_queue_management(self, scheduler):
        """Test task queue management"""
        # Fill queue to capacity
        tasks = [
            TaskRequest(
                task_id=f"queue_task_{i}",
                model_id=f"model_{i}",
                priority=1
            )
            for i in range(scheduler.max_queue_size + 1)
        ]

        # Last task should fail
        for task in tasks[:-1]:
            await scheduler.submit_task(task)
            
        with pytest.raises(SchedulerError):
            await scheduler.submit_task(tasks[-1])

    @pytest.mark.asyncio
    async def test_scheduler_stats(self, scheduler):
        """Test scheduler statistics"""
        # Submit some tasks
        task = TaskRequest(
            task_id="stats_test",
            model_id="test_model",
            priority=1
        )
        await scheduler.submit_task(task)
        await scheduler.schedule_next()

        stats = scheduler.get_stats()
        assert "queue_length" in stats
        assert "scheduled_tasks" in stats
        assert "gpu_utilization" in stats
        assert "cache_hit_rate" in stats

    @pytest.mark.asyncio
    async def test_scheduling_overhead(self, scheduler):
        """Test scheduling overhead measurement"""
        task = TaskRequest(
            task_id="overhead_test",
            model_id="test_model",
            priority=1
        )

        start_time = asyncio.get_event_loop().time()
        decision = await scheduler.get_scheduling_decision(task)
        overhead = asyncio.get_event_loop().time() - start_time

        assert overhead < 0.1  # Should be fast
        assert "scheduling_time" in decision.metrics