# tests/gpu_sharing/test_time_sharing.py

import pytest
import asyncio
from datetime import datetime
from src.gpu_sharing.time_sharing import TimeScheduler, ProcessInfo
from src.common.exceptions import SchedulerError

class TestTimeScheduler:
    @pytest.fixture
    async def scheduler(self):
        """Create time scheduler instance"""
        scheduler = TimeScheduler()
        yield scheduler
        # Cleanup any active processes
        for process_id in list(scheduler.processes.keys()):
            await scheduler.unregister_process(process_id)

    @pytest.mark.asyncio
    async def test_process_registration(self, scheduler):
        """Test process registration"""
        process = await scheduler.register_process(
            process_id="test_proc_1",
            owner_id="test_owner",
            priority=1,
            compute_percentage=20,
            time_quota=1000,
            gpu_id=0,
            vgpu_id="vgpu_1"
        )

        assert process["id"] == "test_proc_1"
        assert process["priority"] == 1
        assert process["compute_percentage"] == 20
        assert process["time_slice"] > 0

    @pytest.mark.asyncio
    async def test_scheduling_priority(self, scheduler):
        """Test priority-based scheduling"""
        # Register processes with different priorities
        await scheduler.register_process(
            process_id="low_priority",
            owner_id="owner1",
            priority=0,
            compute_percentage=20,
            time_quota=1000,
            gpu_id=0,
            vgpu_id="vgpu_1"
        )

        await scheduler.register_process(
            process_id="high_priority",
            owner_id="owner2",
            priority=2,
            compute_percentage=20,
            time_quota=1000,
            gpu_id=0,
            vgpu_id="vgpu_2"
        )

        # Schedule next process
        next_process = await scheduler.schedule_next()
        assert next_process == "high_priority"

    @pytest.mark.asyncio
    async def test_time_slice_allocation(self, scheduler):
        """Test time slice allocation"""
        process = await scheduler.register_process(
            process_id="test_proc",
            owner_id="owner",
            priority=1,
            compute_percentage=50,
            time_quota=1000,
            gpu_id=0,
            vgpu_id="vgpu_1"
        )

        # Verify time slice calculation
        assert process["time_slice"] >= scheduler.min_time_slice
        assert process["time_slice"] <= scheduler.max_time_slice

    @pytest.mark.asyncio
    async def test_process_preemption(self, scheduler):
        """Test process preemption"""
        # Register and start process
        await scheduler.register_process(
            process_id="proc_1",
            owner_id="owner",
            priority=1,
            compute_percentage=20,
            time_quota=1000,
            gpu_id=0,
            vgpu_id="vgpu_1"
        )

        # Start process
        await scheduler.schedule_next()
        
        # Preempt process
        await scheduler._preempt_active_process()
        
        assert scheduler.active_process is None
        process_state = scheduler.processes["proc_1"]
        assert process_state.used_time > 0

    @pytest.mark.asyncio
    async def test_quota_enforcement(self, scheduler):
        """Test time quota enforcement"""
        process_id = "quota_test"
        small_quota = 100  # 100ms quota

        await scheduler.register_process(
            process_id=process_id,
            owner_id="owner",
            priority=1,
            compute_percentage=20,
            time_quota=small_quota,
            gpu_id=0,
            vgpu_id="vgpu_1"
        )

        # Run until quota exhausted
        while True:
            next_proc = await scheduler.schedule_next()
            if next_proc != process_id:
                break
            await asyncio.sleep(0.01)

        process_stats = scheduler.get_process_stats(process_id)
        assert process_stats["used_time"] >= small_quota

    @pytest.mark.asyncio
    async def test_multi_gpu_scheduling(self, scheduler):
        """Test scheduling across multiple GPUs"""
        # Register processes on different GPUs
        await scheduler.register_process(
            process_id="gpu0_proc",
            owner_id="owner1",
            priority=1,
            compute_percentage=20,
            time_quota=1000,
            gpu_id=0,
            vgpu_id="vgpu_1"
        )

        await scheduler.register_process(
            process_id="gpu1_proc",
            owner_id="owner2",
            priority=1,
            compute_percentage=20,
            time_quota=1000,
            gpu_id=1,
            vgpu_id="vgpu_2"
        )

        # Both processes should be scheduled independently
        stats = scheduler.get_scheduler_stats()
        assert stats["total_allocations"] == 2

    @pytest.mark.asyncio
    async def test_process_stats(self, scheduler):
        """Test process statistics collection"""
        process_id = "stats_test"
        await scheduler.register_process(
            process_id=process_id,
            owner_id="owner",
            priority=1,
            compute_percentage=20,
            time_quota=1000,
            gpu_id=0,
            vgpu_id="vgpu_1"
        )

        # Run process
        await scheduler.schedule_next()
        await asyncio.sleep(0.1)
        await scheduler._preempt_active_process()

        # Check stats
        stats = scheduler.get_process_stats(process_id)
        assert "used_time" in stats
        assert "remaining_quota" in stats
        assert "runtime" in stats

    @pytest.mark.asyncio
    async def test_error_handling(self, scheduler):
        """Test error handling"""
        # Register same process twice
        await scheduler.register_process(
            process_id="dup_proc",
            owner_id="owner",
            priority=1,
            compute_percentage=20,
            time_quota=1000,
            gpu_id=0,
            vgpu_id="vgpu_1"
        )

        with pytest.raises(SchedulerError):
            await scheduler.register_process(
                process_id="dup_proc",
                owner_id="owner",
                priority=1,
                compute_percentage=20,
                time_quota=1000,
                gpu_id=0,
                vgpu_id="vgpu_1"
            )

    @pytest.mark.asyncio
    async def test_unregister_process(self, scheduler):
        """Test process unregistration"""
        process_id = "unreg_test"
        await scheduler.register_process(
            process_id=process_id,
            owner_id="owner",
            priority=1,
            compute_percentage=20,
            time_quota=1000,
            gpu_id=0,
            vgpu_id="vgpu_1"
        )

        success = await scheduler.unregister_process(process_id)
        assert success is True
        assert process_id not in scheduler.processes