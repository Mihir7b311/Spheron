# src/integration/scheduler.py

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from ..cache_system import ModelCache
from ..execution_engine.runtime import PythonExecutor
from ..gpu_sharing import VirtualGPUManager, TimeScheduler
from ..common.exceptions import SchedulerError
from ..common.monitoring import ResourceMonitor
from ..common.metrics import MetricsCollector

class SchedulingPolicy(Enum):
    """Available scheduling policies"""
    LOCALITY_AWARE = "locality_aware"
    LOAD_BALANCING = "load_balancing"
    HYBRID = "hybrid"
    PRIORITY_BASED = "priority_based"

@dataclass
class TaskRequest:
    """Task scheduling request"""
    task_id: str
    model_id: str
    input_data: Any
    priority: int = 0
    gpu_preference: Optional[int] = None
    execution_timeout: Optional[float] = None
    constraints: Optional[Dict] = None

@dataclass
class SchedulingDecision:
    """Scheduling decision details"""
    task_id: str
    gpu_id: int
    vgpu_id: Optional[str]
    expected_start_time: float
    expected_duration: float
    cache_hit: bool
    policy_applied: SchedulingPolicy

class IntegratedScheduler:
    def __init__(self, config: Dict = None):
        """Initialize Integrated Scheduler
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Component instances
        self.model_cache = None
        self.python_executor = None
        self.vgpu_manager = None
        self.time_scheduler = None
        
        # Monitoring
        self.resource_monitor = ResourceMonitor()
        self.metrics_collector = MetricsCollector()
        
        # Scheduling state
        self.policy = SchedulingPolicy(self.config.get('scheduling_policy', 'locality_aware'))
        self.task_queue: List[TaskRequest] = []
        self.scheduled_tasks: Dict[str, Dict] = {}
        self.gpu_queues: Dict[int, List[str]] = defaultdict(list)
        self.gpu_load: Dict[int, float] = {}
        
        # Cache awareness
        self.model_locations: Dict[str, List[int]] = defaultdict(list)  # model_id -> gpu_ids
        self.cache_stats: Dict[str, Dict] = {}  # model_id -> stats
        
        # Configuration
        self.max_queue_size = self.config.get('max_queue_size', 1000)
        self.batch_size = self.config.get('batch_size', 32)
        self.locality_weight = self.config.get('locality_weight', 0.7)
        self.load_weight = self.config.get('load_weight', 0.3)
        self.cache_threshold = self.config.get('cache_threshold', 0.2)  # 20% reuse rate threshold
        
        # Background tasks
        self.scheduling_task = None
        self.monitoring_task = None

    async def initialize(self):
        """Initialize scheduler"""
        try:
            logging.info("Initializing integrated scheduler...")
            
            # Initialize components
            self.model_cache = ModelCache(
                gpu_id=0,  # Will be updated based on scheduling
                capacity_gb=self.config.get('cache_capacity_gb', 8)
            )
            
            self.vgpu_manager = VirtualGPUManager(self.config)
            self.time_scheduler = TimeScheduler(self.config)
            
            # Start monitoring
            await self.resource_monitor.start_monitoring()
            
            # Start background tasks
            self.scheduling_task = asyncio.create_task(self._scheduling_loop())
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logging.info("Scheduler initialization completed")
            
        except Exception as e:
            logging.error(f"Scheduler initialization failed: {e}")
            raise SchedulerError(f"Initialization failed: {e}")

    async def submit_task(self, request: TaskRequest) -> str:
        """Submit task for scheduling
        Args:
            request: Task scheduling request
        Returns:
            Task ID
        """
        try:
            if len(self.task_queue) >= self.max_queue_size:
                raise SchedulerError("Task queue full")
                
            # Add to queue
            self.task_queue.append(request)
            
            # Update cache stats if model seen before
            if request.model_id in self.cache_stats:
                self.cache_stats[request.model_id]["requests"] += 1
                
            logging.info(f"Submitted task {request.task_id} for scheduling")
            return request.task_id
            
        except Exception as e:
            logging.error(f"Task submission failed: {e}")
            raise SchedulerError(f"Task submission failed: {e}")

    async def get_scheduling_decision(self, task: TaskRequest) -> SchedulingDecision:
        """Make scheduling decision for task
        Args:
            task: Task to schedule
        Returns:
            Scheduling decision
        """
        try:
            decision = None
            
            if self.policy == SchedulingPolicy.LOCALITY_AWARE:
                decision = await self._make_locality_aware_decision(task)
            elif self.policy == SchedulingPolicy.LOAD_BALANCING:
                decision = await self._make_load_balanced_decision(task)
            elif self.policy == SchedulingPolicy.HYBRID:
                decision = await self._make_hybrid_decision(task)
            elif self.policy == SchedulingPolicy.PRIORITY_BASED:
                decision = await self._make_priority_based_decision(task)
                
            if not decision:
                raise SchedulerError("Failed to make scheduling decision")
                
            return decision
            
        except Exception as e:
            logging.error(f"Scheduling decision failed: {e}")
            raise SchedulerError(f"Scheduling decision failed: {e}")

    async def _make_locality_aware_decision(self, task: TaskRequest) -> SchedulingDecision:
        """Make locality-aware scheduling decision"""
        try:
            # Check if model is cached
            cached_gpus = self.model_locations.get(task.model_id, [])
            
            if cached_gpus:
                # Find best cached GPU
                best_gpu = None
                best_score = float('inf')
                
                for gpu_id in cached_gpus:
                    # Calculate score based on load and queue length
                    load = self.gpu_load.get(gpu_id, 0)
                    queue_length = len(self.gpu_queues[gpu_id])
                    score = (load * 0.6) + (queue_length / self.max_queue_size * 0.4)
                    
                    if score < best_score:
                        best_score = score
                        best_gpu = gpu_id
                        
                if best_gpu is not None:
                    return SchedulingDecision(
                        task_id=task.task_id,
                        gpu_id=best_gpu,
                        vgpu_id=None,  # Will be assigned later
                        expected_start_time=asyncio.get_event_loop().time(),
                        expected_duration=self._estimate_duration(task),
                        cache_hit=True,
                        policy_applied=SchedulingPolicy.LOCALITY_AWARE
                    )
                    
            # No cache hit - find least loaded GPU
            return await self._make_load_balanced_decision(task)
            
        except Exception as e:
            logging.error(f"Locality-aware decision failed: {e}")
            raise SchedulerError(f"Locality-aware decision failed: {e}")

    async def _make_load_balanced_decision(self, task: TaskRequest) -> SchedulingDecision:
        """Make load-balanced scheduling decision"""
        try:
            # Find least loaded GPU
            min_load = float('inf')
            chosen_gpu = None
            
            for gpu_id, load in self.gpu_load.items():
                if load < min_load:
                    min_load = load
                    chosen_gpu = gpu_id
                    
            if chosen_gpu is None:
                raise SchedulerError("No available GPU")
                
            return SchedulingDecision(
                task_id=task.task_id,
                gpu_id=chosen_gpu,
                vgpu_id=None,  # Will be assigned later
                expected_start_time=asyncio.get_event_loop().time(),
                expected_duration=self._estimate_duration(task),
                cache_hit=False,
                policy_applied=SchedulingPolicy.LOAD_BALANCING
            )
            
        except Exception as e:
            logging.error(f"Load-balanced decision failed: {e}")
            raise SchedulerError(f"Load-balanced decision failed: {e}")

    async def _make_hybrid_decision(self, task: TaskRequest) -> SchedulingDecision:
        """Make hybrid scheduling decision considering both locality and load"""
        try:
            cached_gpus = self.model_locations.get(task.model_id, [])
            best_gpu = None
            best_score = float('inf')
            
            for gpu_id in self.gpu_load:
                # Calculate locality score
                locality_score = 0 if gpu_id in cached_gpus else 1
                
                # Calculate load score
                load_score = self.gpu_load[gpu_id]
                
                # Calculate queue length score
                queue_score = len(self.gpu_queues[gpu_id]) / self.max_queue_size
                
                # Combined weighted score
                score = (
                    (locality_score * self.locality_weight) +
                    (load_score * self.load_weight) +
                    (queue_score * 0.2)  # 20% weight to queue length
                )
                
                if score < best_score:
                    best_score = score
                    best_gpu = gpu_id
                    
            if best_gpu is None:
                raise SchedulerError("No available GPU")
                
            return SchedulingDecision(
                task_id=task.task_id,
                gpu_id=best_gpu,
                vgpu_id=None,
                expected_start_time=asyncio.get_event_loop().time(),
                expected_duration=self._estimate_duration(task),
                cache_hit=best_gpu in cached_gpus,
                policy_applied=SchedulingPolicy.HYBRID
            )
            
        except Exception as e:
            logging.error(f"Hybrid decision failed: {e}")
            raise SchedulerError(f"Hybrid decision failed: {e}")

    async def _make_priority_based_decision(self, task: TaskRequest) -> SchedulingDecision:
        """Make priority-based scheduling decision"""
        try:
            # High priority tasks get preference for cached GPUs
            if task.priority >= 2:  # High priority
                cached_gpus = self.model_locations.get(task.model_id, [])
                if cached_gpus:
                    # Choose least loaded cached GPU
                    chosen_gpu = min(
                        cached_gpus,
                        key=lambda x: self.gpu_load.get(x, 0)
                    )
                    return SchedulingDecision(
                        task_id=task.task_id,
                        gpu_id=chosen_gpu,
                        vgpu_id=None,
                        expected_start_time=asyncio.get_event_loop().time(),
                        expected_duration=self._estimate_duration(task),
                        cache_hit=True,
                        policy_applied=SchedulingPolicy.PRIORITY_BASED
                    )
                    
            # Lower priority tasks use hybrid scheduling
            return await self._make_hybrid_decision(task)
            
        except Exception as e:
            logging.error(f"Priority-based decision failed: {e}")
            raise SchedulerError(f"Priority-based decision failed: {e}")

    def _estimate_duration(self, task: TaskRequest) -> float:
        """Estimate task execution duration"""
        # Use historical data if available
        if task.model_id in self.cache_stats:
            return self.cache_stats[task.model_id].get("avg_duration", 1.0)
        return 1.0  # Default 1 second estimate

    async def _scheduling_loop(self):
        """Main scheduling loop"""
        while True:
            try:
                await asyncio.sleep(0.1)  # 100ms scheduling interval
                
                # Process tasks in queue
                while self.task_queue:
                    task = self.task_queue[0]  # Peek first task
                    
                    # Get scheduling decision
                    decision = await self.get_scheduling_decision(task)
                    
                    # Check if can schedule now
                    if await self._can_schedule(decision):
                        # Remove from queue
                        self.task_queue.pop(0)
                        
                        # Schedule task
                        await self._schedule_task(task, decision)
                    else:
                        # Can't schedule now, try next iteration
                        break
                    
            except Exception as e:
                logging.error(f"Scheduling loop error: {e}")

    async def _can_schedule(self, decision: SchedulingDecision) -> bool:
        """Check if task can be scheduled according to decision"""
        try:
            gpu_id = decision.gpu_id
            
            # Check GPU load
            if self.gpu_load.get(gpu_id, 0) >= 0.9:  # 90% load threshold
                return False
                
            # Check queue length
            if len(self.gpu_queues[gpu_id]) >= self.max_queue_size:
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Schedule check failed: {e}")
            return False

    async def _schedule_task(self, task: TaskRequest, decision: SchedulingDecision):
        """Schedule task according to decision"""
        try:
            gpu_id = decision.gpu_id
            
            # Add to GPU queue
            self.gpu_queues[gpu_id].append(task.task_id)
            
            # Update scheduling state
            self.scheduled_tasks[task.task_id] = {
                "task": task,
                "decision": decision,
                "start_time": asyncio.get_event_loop().time(),
                "status": "scheduled"
            }
            
            # Update metrics
            await self.metrics_collector.record_scheduling({
                "task_id": task.task_id,
                "gpu_id": gpu_id,
                "cache_hit": decision.cache_hit,
                "policy": decision.policy_applied.value
            })
            
            logging.info(
                f"Scheduled task {task.task_id} to GPU {gpu_id} "
                f"(cache_hit={decision.cache_hit})"
            )
            
        except Exception as e:
            logging.error(f"Task scheduling failed: {e}")
            raise SchedulerError(f"Task scheduling failed: {e}")

    async def _monitoring_loop(self):
        """Monitor scheduling performance"""
        while True:
            try:
                await asyncio.sleep(1.0)  # 1 second monitoring interval
                
                # Update GPU loads
                metrics = await self.resource_monitor.collect_metrics()
                for gpu_id, gpu_metrics in metrics.items():
                    self.gpu_load[gpu_id] = gpu_metrics.utilization / 100
                    
                # Check for completed tasks
                current_time = asyncio.get_event_loop().time()
                for task_id, task_info in list(self.scheduled_tasks.items()):
                    if task_info["status"] == "scheduled":
                        decision = task_info["decision"]
                        if current_time >= decision.expected_start_time + decision.expected_duration:
                            # Task should be complete
                            await self._handle_task_completion(task_id, task_info)
                            
                # Update cache statistics
                await self._update_cache_stats()
                
                # Adjust scheduling policy if needed
                await self._adjust_scheduling_policy()
                
                # Log performance metrics
                await self._log_performance_metrics()
                
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5.0)  # Wait before retry

    async def _handle_task_completion(self, task_id: str, task_info: Dict):
        """Handle task completion
        Args:
            task_id: Task identifier
            task_info: Task information dictionary
        """
        try:
            # Update task status
            task_info["status"] = "completed"
            task_info["end_time"] = asyncio.get_event_loop().time()
            
            # Calculate metrics
            duration = task_info["end_time"] - task_info["start_time"]
            decision = task_info["decision"]
            
            # Update model statistics
            model_id = task_info["task"].model_id
            if model_id not in self.cache_stats:
                self.cache_stats[model_id] = {
                    "requests": 1,
                    "cache_hits": 0,
                    "total_duration": 0,
                    "avg_duration": 0
                }
                
            stats = self.cache_stats[model_id]
            stats["total_duration"] += duration
            stats["avg_duration"] = stats["total_duration"] / stats["requests"]
            if decision.cache_hit:
                stats["cache_hits"] += 1
            
            # Remove from GPU queue
            if task_id in self.gpu_queues[decision.gpu_id]:
                self.gpu_queues[decision.gpu_id].remove(task_id)
                
            # Record completion metrics
            await self.metrics_collector.record_completion({
                "task_id": task_id,
                "duration": duration,
                "gpu_id": decision.gpu_id,
                "cache_hit": decision.cache_hit
            })
            
            logging.info(f"Task {task_id} completed in {duration:.2f}s")
            
        except Exception as e:
            logging.error(f"Failed to handle task completion: {e}")

    async def _update_cache_stats(self):
        """Update cache statistics and manage model locations"""
        try:
            current_time = asyncio.get_event_loop().time()
            
            for model_id, stats in self.cache_stats.items():
                # Calculate cache hit rate
                hit_rate = stats["cache_hits"] / stats["requests"] if stats["requests"] > 0 else 0
                
                # Update model locations based on usage
                if hit_rate >= self.cache_threshold:
                    # Model is frequently used - ensure it's cached on GPUs
                    if model_id not in self.model_locations:
                        # Find suitable GPU to cache model
                        chosen_gpu = min(
                            self.gpu_load.items(),
                            key=lambda x: x[1]
                        )[0]
                        self.model_locations[model_id].append(chosen_gpu)
                else:
                    # Model is infrequently used - consider evicting from cache
                    if model_id in self.model_locations:
                        await self._evict_model(model_id)
                        
        except Exception as e:
            logging.error(f"Failed to update cache stats: {e}")

    async def _evict_model(self, model_id: str):
        """Evict model from cache
        Args:
            model_id: Model to evict
        """
        try:
            if model_id in self.model_locations:
                gpu_ids = self.model_locations[model_id]
                for gpu_id in gpu_ids:
                    # Request cache eviction
                    if self.model_cache:
                        await self.model_cache.evict_model(model_id, gpu_id)
                        
                # Clear location tracking
                self.model_locations.pop(model_id)
                
            logging.info(f"Evicted model {model_id} from cache")
            
        except Exception as e:
            logging.error(f"Failed to evict model: {e}")

    async def _adjust_scheduling_policy(self):
        """Adjust scheduling policy based on performance metrics"""
        try:
            metrics = await self.metrics_collector.get_recent_metrics(100)  # Last 100 tasks
            
            if not metrics:
                return
                
            # Calculate performance indicators
            avg_duration = sum(m["duration"] for m in metrics) / len(metrics)
            cache_hit_rate = sum(1 for m in metrics if m["cache_hit"]) / len(metrics)
            load_balance = self._calculate_load_balance()
            
            # Adjust policy based on performance
            if cache_hit_rate < 0.3:  # Low cache utilization
                self.locality_weight = min(0.9, self.locality_weight + 0.1)
            elif load_balance < 0.7:  # Poor load balance
                self.locality_weight = max(0.3, self.locality_weight - 0.1)
                
            logging.info(
                f"Adjusted scheduling weights: locality={self.locality_weight:.2f}, "
                f"load={self.load_weight:.2f}"
            )
            
        except Exception as e:
            logging.error(f"Failed to adjust scheduling policy: {e}")

    def _calculate_load_balance(self) -> float:
        """Calculate GPU load balance metric"""
        try:
            if not self.gpu_load:
                return 1.0
                
            loads = list(self.gpu_load.values())
            avg_load = sum(loads) / len(loads)
            max_deviation = max(abs(load - avg_load) for load in loads)
            
            return 1.0 - (max_deviation / avg_load if avg_load > 0 else 0)
            
        except Exception as e:
            logging.error(f"Failed to calculate load balance: {e}")
            return 1.0

    async def _log_performance_metrics(self):
        """Log current performance metrics"""
        try:
            metrics = {
                "queue_length": len(self.task_queue),
                "active_tasks": len(self.scheduled_tasks),
                "gpu_loads": self.gpu_load,
                "cache_models": len(self.model_locations),
                "scheduling_policy": self.policy.value
            }
            
            # Record metrics
            await self.metrics_collector.record_scheduler_metrics(metrics)
            
        except Exception as e:
            logging.error(f"Failed to log performance metrics: {e}")

    async def cleanup(self):
        """Cleanup scheduler resources"""
        try:
            logging.info("Cleaning up scheduler resources")
            
            # Cancel background tasks
            if self.scheduling_task:
                self.scheduling_task.cancel()
            if self.monitoring_task:
                self.monitoring_task.cancel()
                
            # Clear queues and state
            self.task_queue.clear()
            self.scheduled_tasks.clear()
            self.gpu_queues.clear()
            
            # Stop monitoring
            await self.resource_monitor.stop_monitoring()
            
            logging.info("Scheduler cleanup completed")
            
        except Exception as e:
            logging.error(f"Scheduler cleanup failed: {e}")
            raise SchedulerError(f"Cleanup failed: {e}")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()