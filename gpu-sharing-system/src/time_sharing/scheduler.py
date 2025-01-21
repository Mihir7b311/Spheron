# src/time_sharing/scheduler.py

import asyncio
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
from ..common.exceptions import SchedulerError

@dataclass
class ProcessInfo:
    """Information about a GPU process"""
    id: str
    owner_id: str
    priority: int
    compute_percentage: int
    time_slice: int  # milliseconds
    total_quota: int # milliseconds
    used_time: int = 0
    start_time: float = 0
    last_run: float = 0

class TimeScheduler:
    def __init__(self):
        self.processes: Dict[str, ProcessInfo] = {}
        self.active_process: Optional[str] = None
        self.min_time_slice = 100  # minimum 100ms time slice
        self.max_time_slice = 1000  # maximum 1s time slice
        self.context_switch_overhead = 10  # 10ms context switch overhead

    async def register_process(self, 
                             process_id: str,
                             owner_id: str,
                             priority: int,
                             compute_percentage: int,
                             time_quota: int) -> Dict:
        """Register a new process for scheduling"""
        try:
            if process_id in self.processes:
                raise SchedulerError(f"Process {process_id} already registered")

            # Calculate time slice based on priority and compute percentage
            time_slice = self._calculate_time_slice(priority, compute_percentage)
            
            process = ProcessInfo(
                id=process_id,
                owner_id=owner_id,
                priority=priority,
                compute_percentage=compute_percentage,
                time_slice=time_slice,
                total_quota=time_quota,
                start_time=asyncio.get_event_loop().time()
            )
            
            self.processes[process_id] = process
            logging.info(f"Registered process {process_id} with {time_slice}ms time slice")
            
            return process.__dict__

        except Exception as e:
            logging.error(f"Failed to register process: {e}")
            raise SchedulerError(f"Process registration failed: {e}")

    def _calculate_time_slice(self, priority: int, compute_percentage: int) -> int:
        """Calculate time slice based on priority and compute percentage"""
        # Higher priority and compute percentage get larger time slices
        base_slice = self.min_time_slice + (priority * 50)  # 50ms per priority level
        compute_bonus = (compute_percentage / 100) * 200  # up to 200ms bonus
        
        time_slice = int(base_slice + compute_bonus)
        return min(max(time_slice, self.min_time_slice), self.max_time_slice)

    async def unregister_process(self, process_id: str) -> bool:
        """Unregister a process"""
        try:
            if process_id not in self.processes:
                return False

            if self.active_process == process_id:
                await self._preempt_active_process()

            self.processes.pop(process_id)
            logging.info(f"Unregistered process {process_id}")
            return True

        except Exception as e:
            logging.error(f"Failed to unregister process: {e}")
            raise SchedulerError(f"Process unregistration failed: {e}")

    async def schedule_next(self) -> Optional[str]:
        """Schedule next process to run"""
        try:
            if not self.processes:
                return None

            # Preempt current process if any
            if self.active_process:
                await self._preempt_active_process()

            # Select next process based on priority and quota
            next_process = self._select_next_process()
            if not next_process:
                return None

            # Activate selected process
            self.active_process = next_process.id
            next_process.last_run = asyncio.get_event_loop().time()
            
            logging.info(f"Scheduled process {next_process.id}")
            return next_process.id

        except Exception as e:
            logging.error(f"Scheduling failed: {e}")
            raise SchedulerError(f"Scheduling failed: {e}")

    def _select_next_process(self) -> Optional[ProcessInfo]:
        """Select next process to run based on priority and quota"""
        eligible_processes = [
            p for p in self.processes.values()
            if p.used_time < p.total_quota
        ]
        
        if not eligible_processes:
            return None

        # Sort by priority and unused quota
        return max(
            eligible_processes,
            key=lambda p: (
                p.priority,
                p.total_quota - p.used_time,
                -p.last_run  # prefer processes that haven't run recently
            )
        )

    async def _preempt_active_process(self):
        """Preempt currently active process"""
        if not self.active_process:
            return

        process = self.processes[self.active_process]
        current_time = asyncio.get_event_loop().time()
        run_time = int((current_time - process.last_run) * 1000)  # convert to ms
        
        process.used_time += run_time
        self.active_process = None
        
        logging.info(
            f"Preempted process {process.id} "
            f"(used {run_time}ms, total {process.used_time}ms)"
        )

    def get_process_stats(self, process_id: str) -> Optional[Dict]:
        """Get statistics for a process"""
        process = self.processes.get(process_id)
        if not process:
            return None

        current_time = asyncio.get_event_loop().time()
        return {
            "id": process.id,
            "owner_id": process.owner_id,
            "priority": process.priority,
            "compute_percentage": process.compute_percentage,
            "time_slice": process.time_slice,
            "total_quota": process.total_quota,
            "used_time": process.used_time,
            "remaining_quota": process.total_quota - process.used_time,
            "runtime": int((current_time - process.start_time) * 1000)
        }

    def get_scheduler_stats(self) -> Dict:
        """Get scheduler statistics"""
        return {
            "active_processes": len(self.processes),
            "current_process": self.active_process,
            "total_time_allocated": sum(p.total_quota for p in self.processes.values()),
            "total_time_used": sum(p.used_time for p in self.processes.values())
        }