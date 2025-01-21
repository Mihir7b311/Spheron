# tests/time_sharing/test_scheduler.py
import pytest
import asyncio
from src.time_sharing.scheduler import TimeScheduler, ProcessInfo
from src.common.exceptions import SchedulerError

@pytest.mark.asyncio
class TestTimeScheduler:
    async def test_initialization(self):
        """Test scheduler initialization"""
        scheduler = TimeScheduler()
        assert scheduler.active_process is None
        assert len(scheduler.processes) == 0
        assert scheduler.min_time_slice == 100  # 100ms
        assert scheduler.max_time_slice == 1000  # 1s

    async def test_register_process(self):
        """Test process registration"""
        scheduler = TimeScheduler()
        
        process = await scheduler.register_process(
            process_id="test_proc",
            owner_id="test_owner",
            priority=1,
            compute_percentage=25,
            time_quota=1000
        )
        
        assert process["id"] == "test_proc"
        assert process["owner_id"] == "test_owner"
        assert process["priority"] == 1
        assert process["compute_percentage"] == 25
        assert process["time_slice"] > 0
        assert process["total_quota"] == 1000

    async def test_register_duplicate_process(self):
        """Test registering duplicate process"""
        scheduler = TimeScheduler()
        
        await scheduler.register_process(
            "test_proc", "owner", 1, 25, 1000
        )
        
        with pytest.raises(SchedulerError, match="already registered"):
            await scheduler.register_process(
                "test_proc", "owner", 1, 25, 1000
            )

    async def test_calculate_time_slice(self):
        """Test time slice calculation"""
        scheduler = TimeScheduler()
        
        # Test minimum time slice
        time_slice = scheduler._calculate_time_slice(0, 10)
        assert scheduler.min_time_slice <= time_slice <= scheduler.max_time_slice
        
        # Test maximum time slice
        time_slice = scheduler._calculate_time_slice(10, 100)
        assert time_slice == scheduler.max_time_slice

    async def test_schedule_next(self):
        """Test scheduling next process"""
        scheduler = TimeScheduler()
        
        # Register two processes with different priorities
        await scheduler.register_process(
            "proc1", "owner1", 1, 25, 1000
        )
        await scheduler.register_process(
            "proc2", "owner2", 2, 25, 1000
        )
        
        # Higher priority should run first
        next_proc = await scheduler.schedule_next()
        assert next_proc == "proc2"
        
        # After preemption, should switch to lower priority
        next_proc = await scheduler.schedule_next()
        assert next_proc == "proc1"

    async def test_preempt_active_process(self):
        """Test process preemption"""
        scheduler = TimeScheduler()
        
        await scheduler.register_process(
            "test_proc", "owner", 1, 25, 1000
        )
        
        await scheduler.schedule_next()
        await asyncio.sleep(0.1)  # Simulate some execution time
        
        await scheduler._preempt_active_process()
        assert scheduler.active_process is None
        assert scheduler.processes["test_proc"].used_time > 0

    async def test_quota_enforcement(self):
        """Test time quota enforcement"""
        scheduler = TimeScheduler()
        
        # Register process with small quota
        await scheduler.register_process(
            "test_proc", "owner", 1, 25, 100  # 100ms quota
        )
        
        # Run until quota exhausted
        start_time = asyncio.get_event_loop().time()
        while await scheduler.schedule_next():
            await asyncio.sleep(0.01)
        end_time = asyncio.get_event_loop().time()
        
        process = scheduler.processes["test_proc"]
        assert process.used_time <= process.total_quota
        assert (end_time - start_time) * 1000 >= process.total_quota

    async def test_get_process_stats(self):
        """Test getting process statistics"""
        scheduler = TimeScheduler()
        
        await scheduler.register_process(
            "test_proc", "owner", 1, 25, 1000
        )
        
        stats = scheduler.get_process_stats("test_proc")
        assert stats["id"] == "test_proc"
        assert stats["owner_id"] == "owner"
        assert stats["priority"] == 1
        assert stats["compute_percentage"] == 25
        assert stats["total_quota"] == 1000
        assert stats["used_time"] == 0
        
        # Test non-existent process
        assert scheduler.get_process_stats("non-existent") is None

    async def test_get_scheduler_stats(self):
        """Test getting scheduler statistics"""
        scheduler = TimeScheduler()
        
        await scheduler.register_process(
            "proc1", "owner1", 1, 25, 1000
        )
        await scheduler.register_process(
            "proc2", "owner2", 2, 25, 1000
        )
        
        stats = scheduler.get_scheduler_stats()
        assert stats["active_processes"] == 2
        assert stats["current_process"] == scheduler.active_process
        assert stats["total_time_allocated"] == 2000
        assert stats["total_time_used"] == 0