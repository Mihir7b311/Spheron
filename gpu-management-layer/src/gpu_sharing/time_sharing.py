# src/gpu_sharing/time_sharing.py

import asyncio
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
from ..common.exceptions import SchedulerError
from ..common.monitoring import ResourceMonitor
from ..execution_engine.cuda import CUDAContextManager

@dataclass
class ProcessInfo:
    """Information about a GPU process"""
    id: str
    owner_id: str
    priority: int
    compute_percentage: int
    time_slice: int      # milliseconds
    total_quota: int     # milliseconds per time window
    gpu_id: int
    vgpu_id: str
    used_time: int = 0
    start_time: float = 0
    last_run: float = 0
    state: str = "ready"  # ready, running, waiting, completed

@dataclass
class SchedulingConfig:
    """Time sharing scheduler configuration"""
    min_time_slice: int = 100     # Minimum 100ms time slice
    max_time_slice: int = 1000    # Maximum 1s time slice
    context_switch_overhead: int = 10  # 10ms overhead
    scheduling_window: int = 3600  # 1 hour scheduling window
    priority_levels: int = 5

class TimeScheduler:
    def __init__(self, config: Dict = None):
        """Initialize Time Scheduler
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scheduling_config = SchedulingConfig(
            **self.config.get("scheduling", {})
        )
        self.processes: Dict[str, ProcessInfo] = {}
        self.active_process: Optional[str] = None
        self.cuda_context_manager = CUDAContextManager()
        self.monitor = ResourceMonitor()
        self.gpu_queues: Dict[int, List[str]] = {}  # Per-GPU process queues
        self.process_states: Dict[str, Dict] = {}    # Process state tracking
        
    async def register_process(self, 
                             process_id: str,
                             owner_id: str,
                             priority: int,
                             compute_percentage: int,
                             time_quota: int,
                             gpu_id: int,
                             vgpu_id: str) -> Dict:
        """Register a new process for scheduling
        Args:
            process_id: Unique process identifier
            owner_id: Process owner identifier
            priority: Process priority (0-4, higher = more important)
            compute_percentage: Required GPU compute percentage
            time_quota: Time quota in milliseconds
            gpu_id: Physical GPU ID
            vgpu_id: Virtual GPU ID
        Returns:
            Process information dictionary
        """
        try:
            if process_id in self.processes:
                raise SchedulerError(f"Process {process_id} already registered")

            if priority >= self.scheduling_config.priority_levels:
                raise SchedulerError(f"Invalid priority level: {priority}")

            # Calculate time slice based on priority and compute percentage
            time_slice = self._calculate_time_slice(priority, compute_percentage)
            
            process = ProcessInfo(
                id=process_id,
                owner_id=owner_id,
                priority=priority,
                compute_percentage=compute_percentage,
                time_slice=time_slice,
                total_quota=time_quota,
                gpu_id=gpu_id,
                vgpu_id=vgpu_id,
                start_time=asyncio.get_event_loop().time()
            )
            
            self.processes[process_id] = process
            
            # Add to GPU queue
            if gpu_id not in self.gpu_queues:
                self.gpu_queues[gpu_id] = []
            self.gpu_queues[gpu_id].append(process_id)
            
            # Initialize process state
            self.process_states[process_id] = {
                "status": "ready",
                "context_switches": 0,
                "preemptions": 0,
                "total_runtime": 0
            }
            
            logging.info(
                f"Registered process {process_id} with {time_slice}ms time slice "
                f"on GPU {gpu_id} (vGPU {vgpu_id})"
            )
            
            return process.__dict__

        except Exception as e:
            logging.error(f"Failed to register process: {e}")
            raise SchedulerError(f"Process registration failed: {e}")

    def _calculate_time_slice(self, priority: int, compute_percentage: int) -> int:
        """Calculate time slice based on priority and compute percentage
        Args:
            priority: Process priority level
            compute_percentage: Required compute percentage
        Returns:
            Time slice in milliseconds
        """
        # Higher priority and compute percentage get larger time slices
        base_slice = self.scheduling_config.min_time_slice + (priority * 50)  
        compute_bonus = (compute_percentage / 100) * 200  # Up to 200ms bonus
        
        time_slice = int(base_slice + compute_bonus)
        return min(
            max(time_slice, self.scheduling_config.min_time_slice),
            self.scheduling_config.max_time_slice
        )

    async def unregister_process(self, process_id: str) -> bool:
        """Unregister a process
        Args:
            process_id: Process to unregister
        Returns:
            Success status
        """
        try:
            if process_id not in self.processes:
                return False

            process = self.processes[process_id]
            
            # Stop process if running
            if self.active_process == process_id:
                await self._preempt_active_process()

            # Remove from queues
            if process.gpu_id in self.gpu_queues:
                self.gpu_queues[process.gpu_id].remove(process_id)
                
            # Cleanup
            self.processes.pop(process_id)
            self.process_states.pop(process_id)
            
            logging.info(f"Unregistered process {process_id}")
            return True

        except Exception as e:
            logging.error(f"Failed to unregister process: {e}")
            raise SchedulerError(f"Process unregistration failed: {e}")

    async def schedule_next(self, gpu_id: Optional[int] = None) -> Optional[str]:
        """Schedule next process to run
        Args:
            gpu_id: Optionally specify GPU to schedule for
        Returns:
            Scheduled process ID if any
        """
        try:
            if not self.processes:
                return None

            # Handle specific GPU if requested
            if gpu_id is not None:
                if gpu_id not in self.gpu_queues:
                    return None
                processes = [
                    p for p in self.processes.values()
                    if p.gpu_id == gpu_id
                ]
            else:
                processes = list(self.processes.values())

            if not processes:
                return None

            # Preempt current process if any
            if self.active_process:
                await self._preempt_active_process()

            # Select next process based on priority and quota
            next_process = self._select_next_process(processes)
            if not next_process:
                return None

            # Activate selected process
            self.active_process = next_process.id
            next_process.last_run = asyncio.get_event_loop().time()
            self.process_states[next_process.id]["status"] = "running"
            
            # Set up CUDA context
            await self.cuda_context_manager.get_context(next_process.gpu_id)
            
            logging.info(f"Scheduled process {next_process.id}")
            return next_process.id

        except Exception as e:
            logging.error(f"Scheduling failed: {e}")
            raise SchedulerError(f"Scheduling failed: {e}")

    def _select_next_process(self, processes: List[ProcessInfo]) -> Optional[ProcessInfo]:
        """Select next process to run based on priority and quota
        Args:
            processes: List of candidate processes
        Returns:
            Selected process if any
        """
        eligible_processes = [
            p for p in processes
            if p.used_time < p.total_quota and
            self.process_states[p.id]["status"] != "completed"
        ]
        
        if not eligible_processes:
            return None

        # Sort by priority, unused quota, and waiting time
        return max(
            eligible_processes,
            key=lambda p: (
                p.priority,
                p.total_quota - p.used_time,
                asyncio.get_event_loop().time() - p.last_run
            )
        )

    async def _preempt_active_process(self):
        """Preempt currently active process"""
        if not self.active_process:
            return

        try:
            process = self.processes[self.active_process]
            current_time = asyncio.get_event_loop().time()
            run_time = int((current_time - process.last_run) * 1000)  # Convert to ms
            
            # Update process metrics
            process.used_time += run_time
            self.process_states[process.id].update({
                "status": "ready",
                "total_runtime": self.process_states[process.id]["total_runtime"] + run_time,
                "preemptions": self.process_states[process.id]["preemptions"] + 1
            })
            
            # Save CUDA context
            await self.cuda_context_manager.synchronize_context(process.gpu_id)
            
            self.active_process = None
            
            logging.info(
                f"Preempted process {process.id} "
                f"(used {run_time}ms, total {process.used_time}ms)"
            )

        except Exception as e:
            logging.error(f"Failed to preempt process: {e}")
            raise SchedulerError(f"Process preemption failed: {e}")

    def get_process_stats(self, process_id: str) -> Optional[Dict]:
        """Get statistics for a process
        Args:
            process_id: Process ID to get stats for
        Returns:
            Process statistics dictionary
        """
        process = self.processes.get(process_id)
        if not process:
            return None

        current_time = asyncio.get_event_loop().time()
        state = self.process_states[process_id]
        
        return {
            "id": process.id,
            "owner_id": process.owner_id,
            "priority": process.priority,
            "compute_percentage": process.compute_percentage,
            "time_slice": process.time_slice,
            "total_quota": process.total_quota,
            "used_time": process.used_time,
            "remaining_quota": process.total_quota - process.used_time,
            "runtime": int((current_time - process.start_time) * 1000),
            "status": state["status"],
            "context_switches": state["context_switches"],
            "preemptions": state["preemptions"],
            "total_runtime": state["total_runtime"]
        }

    def get_scheduler_stats(self) -> Dict:
        """Get scheduler statistics
        Returns:
            Scheduler statistics dictionary
        """
        return {
            "active_processes": len(self.processes),
            "current_process": self.active_process,
            "gpu_queues": {
                gpu_id: len(queue)
                for gpu_id, queue in self.gpu_queues.items()
            },
            "total_time_allocated": sum(
                p.total_quota for p in self.processes.values()
            ),
            "total_time_used": sum(
                p.used_time for p in self.processes.values()
            ),
            "process_states": {
                pid: state["status"]
                for pid, state in self.process_states.items()
            }
        }