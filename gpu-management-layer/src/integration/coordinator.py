# src/integration/coordinator.py

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from ..cache_system import ModelCache
from ..execution_engine.runtime import PythonExecutor
from ..execution_engine.cuda import CUDAContextManager
from ..gpu_sharing import VirtualGPUManager, TimeScheduler, SpaceSharingManager
from ..common.exceptions import CoordinationError
from ..common.monitoring import ResourceMonitor
from ..common.metrics import MetricsCollector

class SystemState(Enum):
    """System component states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class ComponentStatus:
    """Status information for system components"""
    state: SystemState
    healthy: bool
    last_error: Optional[str] = None
    last_check: float = 0.0

class SystemCoordinator:
    def __init__(self, config: Dict = None):
        """Initialize System Coordinator
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Component instances
        self.model_cache = None
        self.python_executor = None
        self.cuda_manager = None
        self.vgpu_manager = None
        self.time_scheduler = None
        self.space_manager = None
        
        # Monitoring and metrics
        self.resource_monitor = ResourceMonitor()
        self.metrics_collector = MetricsCollector()
        
        # State tracking
        self.component_status: Dict[str, ComponentStatus] = {}
        self.system_state = SystemState.INITIALIZING
        self.active_tasks: Dict[str, Dict] = {}
        
        # Configuration
        self.health_check_interval = self.config.get('health_check_interval', 5.0)
        self.recovery_attempts = self.config.get('recovery_attempts', 3)
        self.component_timeout = self.config.get('component_timeout', 10.0)
        
        # Background tasks
        self.health_check_task = None
        self.metric_collection_task = None

    async def initialize(self):
        """Initialize all system components"""
        try:
            logging.info("Initializing system components...")

            # Initialize CUDA manager first
            self.cuda_manager = CUDAContextManager()
            self._register_component("cuda_manager", SystemState.INITIALIZING)

            # Initialize cache system
            self.model_cache = ModelCache(
                gpu_id=0,  # Will be updated based on scheduling
                capacity_gb=self.config.get('cache_capacity_gb', 8)
            )
            self._register_component("model_cache", SystemState.INITIALIZING)

            # Initialize GPU sharing components
            self.vgpu_manager = VirtualGPUManager(self.config.get('vgpu_config'))
            self.time_scheduler = TimeScheduler(self.config.get('scheduler_config'))
            self.space_manager = SpaceSharingManager(self.config.get('space_config'))
            
            self._register_component("vgpu_manager", SystemState.INITIALIZING)
            self._register_component("time_scheduler", SystemState.INITIALIZING)
            self._register_component("space_manager", SystemState.INITIALIZING)

            # Initialize execution engine
            self.python_executor = PythonExecutor(
                self.cuda_manager,
                self.config.get('executor_config')
            )
            self._register_component("python_executor", SystemState.INITIALIZING)

            # Start monitoring
            await self.resource_monitor.start_monitoring()
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            self.metric_collection_task = asyncio.create_task(self._metric_collection_loop())

            # Update system state
            self.system_state = SystemState.RUNNING
            logging.info("System initialization completed successfully")

        except Exception as e:
            self.system_state = SystemState.ERROR
            logging.error(f"System initialization failed: {e}")
            raise CoordinationError(f"System initialization failed: {e}")

    async def execute_model(self, 
                          model_id: str,
                          model_data: bytes,
                          input_data: Dict[str, Any]) -> Dict:
        """Execute ML model with coordinated resource management
        Args:
            model_id: Model identifier
            model_data: Model binary data
            input_data: Input data for model
        Returns:
            Execution results dictionary
        """
        task_id = f"task_{len(self.active_tasks)}_{model_id}"
        
        try:
            # Register active task
            self.active_tasks[task_id] = {
                "model_id": model_id,
                "state": "initializing",
                "start_time": asyncio.get_event_loop().time()
            }

            # Check system state
            if self.system_state != SystemState.RUNNING:
                raise CoordinationError("System not in running state")

            # Check cache first
            cached_model = await self.model_cache.get_model(model_id)
            cache_hit = cached_model is not None

            if not cache_hit:
                # Allocate virtual GPU resources
                vgpu_config = await self._allocate_vgpu_resources(model_data)
                vgpu = await self.vgpu_manager.create_virtual_gpu(vgpu_config)
                
                # Set up memory partition
                partition = await self.space_manager.create_partition(
                    gpu_id=vgpu["gpu_id"],
                    size_mb=vgpu_config.memory_mb,
                    owner_id=task_id,
                    vgpu_id=vgpu["id"]
                )
                
                # Store model in cache
                await self.model_cache.store_model(model_id, model_data)
                cached_model = await self.model_cache.get_model(model_id)

            # Register with scheduler
            await self.time_scheduler.register_process(
                process_id=task_id,
                owner_id=task_id,
                priority=1,  # Default priority
                compute_percentage=50,  # Can be configured
                time_quota=1000,  # 1 second quota
                gpu_id=cached_model.device.index,
                vgpu_id=vgpu["id"] if not cache_hit else None
            )

            # Update task state
            self.active_tasks[task_id]["state"] = "executing"

            # Execute model
            results = await self.python_executor.execute_function(
                model=cached_model,
                input_data=input_data,
                gpu_context={
                    "gpu_id": cached_model.device.index,
                    "vgpu_id": vgpu["id"] if not cache_hit else None
                }
            )

            # Collect metrics
            await self.metrics_collector.record_metrics({
                "task_id": task_id,
                "cache_hit": cache_hit,
                "execution_time": results["metrics"]["execution_time"],
                "gpu_id": cached_model.device.index
            })

            # Update task state
            self.active_tasks[task_id]["state"] = "completed"
            self.active_tasks[task_id]["end_time"] = asyncio.get_event_loop().time()

            return {
                "task_id": task_id,
                "results": results["results"],
                "metrics": results["metrics"],
                "cache_hit": cache_hit
            }

        except Exception as e:
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["state"] = "error"
                self.active_tasks[task_id]["error"] = str(e)
            logging.error(f"Task execution failed: {e}")
            raise CoordinationError(f"Task execution failed: {e}")

    async def _allocate_vgpu_resources(self, model_data: bytes) -> Dict:
        """Determine required resources for model
        Args:
            model_data: Model binary data
        Returns:
            Virtual GPU configuration
        """
        # Calculate model size
        model_size_mb = len(model_data) / (1024 * 1024)
        
        # Add overhead for runtime
        required_memory = int(model_size_mb * 1.5)  # 50% overhead
        
        return {
            "memory_mb": required_memory,
            "compute_percentage": 50,  # Default to 50%
            "priority": 1,
            "enable_mps": True
        }

    async def cleanup_task(self, task_id: str):
        """Clean up resources for completed task
        Args:
            task_id: Task identifier
        """
        try:
            if task_id not in self.active_tasks:
                return

            task = self.active_tasks[task_id]
            
            # Cleanup scheduler
            await self.time_scheduler.unregister_process(task_id)
            
            # Release resources if allocated
            if "vgpu_id" in task:
                await self.vgpu_manager.release_virtual_gpu(task["vgpu_id"])
                await self.space_manager.release_partition(task["partition_id"])

            # Remove task
            del self.active_tasks[task_id]

        except Exception as e:
            logging.error(f"Task cleanup failed: {e}")
            raise CoordinationError(f"Task cleanup failed: {e}")

    def _register_component(self, name: str, initial_state: SystemState):
        """Register system component"""
        self.component_status[name] = ComponentStatus(
            state=initial_state,
            healthy=True,
            last_check=asyncio.get_event_loop().time()
        )

    async def _health_check_loop(self):
        """Periodic health check of all components"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                for component, status in self.component_status.items():
                    try:
                        # Check component health
                        healthy = await self._check_component_health(component)
                        
                        status.healthy = healthy
                        status.last_check = asyncio.get_event_loop().time()
                        
                        if not healthy:
                            logging.warning(f"Component {component} health check failed")
                            await self._handle_component_failure(component)
                            
                    except Exception as e:
                        status.healthy = False
                        status.last_error = str(e)
                        logging.error(f"Health check failed for {component}: {e}")

            except Exception as e:
                logging.error(f"Health check loop error: {e}")

    async def _check_component_health(self, component: str) -> bool:
        """Check health of specific component"""
        try:
            if component == "cuda_manager":
                return all(
                    ctx["status"] == "active" 
                    for ctx in self.cuda_manager.get_stats()["contexts"].values()
                )
            elif component == "model_cache":
                stats = self.model_cache.get_stats()
                return stats["memory_utilization"] < 0.95
            elif component == "vgpu_manager":
                return all(
                    gpu["status"] == "active"
                    for gpu in self.vgpu_manager.get_stats()["gpu_utilization"].values()
                )
            elif component == "time_scheduler":
                return self.time_scheduler.get_scheduler_stats()["status"] == "running"
            elif component == "space_manager":
                for gpu_id, info in self.space_manager.get_stats()["fragmentation"].items():
                    if info["fragmentation_ratio"] > 0.5:  # High fragmentation
                        await self.space_manager._defragment_gpu(gpu_id)
                return True
            elif component == "python_executor":
                return len(self.python_executor.get_active_executions()) < 100
                
            return True

        except Exception as e:
            logging.error(f"Health check failed for {component}: {e}")
            return False

    async def _handle_component_failure(self, component: str):
        """Handle component failure"""
        try:
            status = self.component_status[component]
            
            if status.state == SystemState.ERROR:
                return  # Already handling error
                
            status.state = SystemState.ERROR
            
            # Attempt recovery
            for attempt in range(self.recovery_attempts):
                try:
                    logging.info(f"Attempting recovery of {component} (attempt {attempt + 1})")
                    
                    if component == "cuda_manager":
                        await self.cuda_manager.cleanup()
                        self.cuda_manager = CUDAContextManager()
                    elif component == "model_cache":
                        self.model_cache = ModelCache(
                            gpu_id=0,
                            capacity_gb=self.config.get('cache_capacity_gb', 8)
                        )
                    elif component == "vgpu_manager":
                        await self.vgpu_manager.cleanup()
                        self.vgpu_manager = VirtualGPUManager(self.config.get('vgpu_config'))
                    elif component == "time_scheduler":
                        self.time_scheduler = TimeScheduler(self.config.get('scheduler_config'))
                    elif component == "space_manager":
                        self.space_manager = SpaceSharingManager(self.config.get('space_config'))
                    elif component == "python_executor":
                        self.python_executor = PythonExecutor(
                            self.cuda_manager,
                            self.config.get('executor_config')
                        )
                        
                    # Check if recovery successful
                    if await self._check_component_health(component):
                        status.state = SystemState.RUNNING
                        status.healthy = True
                        logging.info(f"Successfully recovered {component}")
                        return
                        
                except Exception as e:
                    logging.error(f"Recovery attempt {attempt + 1} failed for {component}: {e}")
                    
                await asyncio.sleep(1)  # Wait before retry
                
            # Recovery failed
            logging.error(f"Failed to recover {component} after {self.recovery_attempts} attempts")
            
        except Exception as e:
            logging.error(f"Error handling component failure: {e}")

    async def _metric_collection_loop(self):
        """Periodic collection of system metrics"""
        while True:
            try:
                await asyncio.sleep(1.0)  # 1 second interval
                
                # Collect GPU metrics
                gpu_metrics = await self.resource_monitor.collect_metrics()
                
                # Collect component metrics
                component_metrics = {
                    "cache": self.model_cache.get_stats(),
                    "scheduler": self.time_scheduler.get_scheduler_stats(),
                    "memory": self.space_manager.get_stats(),
                    "tasks": len(self.active_tasks)
                }
                
                # Store metrics
                await self.metrics_collector.record_system_metrics({
                    "gpu_metrics": gpu_metrics,
                    "component_metrics": component_metrics,
                    "timestamp": asyncio.get_event_loop().time()
                })
                
            except Exception as e:
                logging.error(f"Metric collection error: {e}")

    def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            "state": self.system_state,
            "components": {
                name: {
                    "state": status.state,
                    "healthy": status.healthy,
                    "last_error": status.last_error,
                    "last_check": status.last_check
                }
                for name, status in self.component_status.items()
            },
            "active_tasks": len(self.active_tasks),
            "metrics": self.metrics_collector.get_recent_metrics(10)
        }
    async def shutdown(self):
        """Gracefully shutdown the system"""
        try:
            logging.info("Initiating system shutdown...")
            self.system_state = SystemState.SHUTDOWN

            # Stop background tasks
            if self.health_check_task:
                self.health_check_task.cancel()
            if self.metric_collection_task:
                self.metric_collection_task.cancel()

            # Cleanup active tasks
            await self._cleanup_active_tasks()

            # Shutdown components in order
            await self._shutdown_components()

            logging.info("System shutdown completed")

        except Exception as e:
            logging.error(f"Shutdown error: {e}")
            raise CoordinationError(f"Shutdown failed: {e}")

    async def _cleanup_active_tasks(self):
        """Cleanup all active tasks"""
        try:
            task_ids = list(self.active_tasks.keys())
            for task_id in task_ids:
                try:
                    logging.info(f"Cleaning up task {task_id}")
                    await self.cleanup_task(task_id)
                except Exception as e:
                    logging.error(f"Failed to cleanup task {task_id}: {e}")
        except Exception as e:
            logging.error(f"Task cleanup error: {e}")

    async def _shutdown_components(self):
        """Shutdown system components in correct order"""
        try:
            # Stop execution engine first
            if self.python_executor:
                await self._shutdown_component("python_executor")

            # Stop GPU sharing components
            if self.time_scheduler:
                await self._shutdown_component("time_scheduler")
            if self.space_manager:
                await self._shutdown_component("space_manager")
            if self.vgpu_manager:
                await self._shutdown_component("vgpu_manager")

            # Stop cache and CUDA components
            if self.model_cache:
                await self._shutdown_component("model_cache")
            if self.cuda_manager:
                await self._shutdown_component("cuda_manager")

            # Stop monitoring
            await self.resource_monitor.stop_monitoring()

        except Exception as e:
            logging.error(f"Component shutdown error: {e}")
            raise CoordinationError(f"Component shutdown failed: {e}")

    async def _shutdown_component(self, component_name: str):
        """Shutdown specific component"""
        try:
            logging.info(f"Shutting down {component_name}")
            
            if component_name == "python_executor":
                # Wait for active executions to complete
                active_execs = self.python_executor.get_active_executions()
                for exec_id in active_execs:
                    await asyncio.wait_for(
                        self._wait_execution_completion(exec_id),
                        timeout=self.component_timeout
                    )
                    
            elif component_name == "time_scheduler":
                # Unregister all processes
                stats = self.time_scheduler.get_scheduler_stats()
                for process_id in stats["process_states"]:
                    await self.time_scheduler.unregister_process(process_id)
                    
            elif component_name == "space_manager":
                # Release all partitions
                stats = self.space_manager.get_stats()
                for gpu_id in stats["gpu_partitions"]:
                    memory_map = self.space_manager.get_gpu_memory_map(gpu_id)
                    for partition in memory_map:
                        await self.space_manager.release_partition(partition["id"])
                        
            elif component_name == "vgpu_manager":
                # Release all virtual GPUs
                stats = self.vgpu_manager.get_stats()
                for gpu_id, gpu_info in stats["gpu_utilization"].items():
                    for vgpu in gpu_info.get("virtual_gpus", []):
                        await self.vgpu_manager.release_virtual_gpu(vgpu["id"])
                        
            elif component_name == "model_cache":
                # Clear cache
                await self.model_cache.clear()
                
            elif component_name == "cuda_manager":
                await self.cuda_manager.cleanup()

            # Update component status
            if component_name in self.component_status:
                self.component_status[component_name].state = SystemState.SHUTDOWN

        except Exception as e:
            logging.error(f"Failed to shutdown {component_name}: {e}")
            raise CoordinationError(f"Component shutdown failed: {e}")

    async def pause_system(self):
        """Pause system operations"""
        try:
            logging.info("Pausing system operations")
            self.system_state = SystemState.PAUSED

            # Pause scheduling
            if self.time_scheduler:
                for process in self.time_scheduler.get_scheduler_stats()["process_states"]:
                    if process["status"] == "running":
                        await self.time_scheduler._preempt_active_process()

            # Stop accepting new tasks
            self._pause_task_acceptance()

            logging.info("System operations paused")

        except Exception as e:
            logging.error(f"System pause failed: {e}")
            raise CoordinationError(f"System pause failed: {e}")

    async def resume_system(self):
        """Resume system operations"""
        try:
            logging.info("Resuming system operations")

            # Check component health before resuming
            all_healthy = True
            for component, status in self.component_status.items():
                if not await self._check_component_health(component):
                    all_healthy = False
                    logging.error(f"Component {component} not healthy")

            if not all_healthy:
                raise CoordinationError("Cannot resume - unhealthy components")

            self.system_state = SystemState.RUNNING
            self._resume_task_acceptance()

            logging.info("System operations resumed")

        except Exception as e:
            logging.error(f"System resume failed: {e}")
            raise CoordinationError(f"System resume failed: {e}")

    def _pause_task_acceptance(self):
        """Stop accepting new tasks"""
        self._accepting_tasks = False

    def _resume_task_acceptance(self):
        """Resume accepting new tasks"""
        self._accepting_tasks = True

    async def get_resource_usage(self) -> Dict:
        """Get current resource usage statistics"""
        try:
            gpu_metrics = await self.resource_monitor.collect_metrics()
            
            return {
                "gpu_utilization": {
                    gpu_id: {
                        "compute": metrics.utilization,
                        "memory": metrics.memory_used / metrics.memory_total * 100
                    }
                    for gpu_id, metrics in gpu_metrics.items()
                },
                "cache_usage": self.model_cache.get_stats(),
                "virtual_gpus": self.vgpu_manager.get_stats(),
                "memory_partitions": {
                    gpu_id: self.space_manager.get_gpu_memory_map(gpu_id)
                    for gpu_id in self.vgpu_manager.physical_gpus
                },
                "active_tasks": len(self.active_tasks)
            }

        except Exception as e:
            logging.error(f"Failed to get resource usage: {e}")
            raise CoordinationError(f"Resource usage check failed: {e}")

    async def _wait_execution_completion(self, execution_id: str, timeout: float = None):
        """Wait for execution to complete
        Args:
            execution_id: Execution to wait for
            timeout: Optional timeout in seconds
        """
        start_time = asyncio.get_event_loop().time()
        timeout = timeout or self.component_timeout

        while True:
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise CoordinationError(f"Execution {execution_id} wait timeout")

            status = self.python_executor.get_execution_status(execution_id)
            if not status or status["status"] in ["completed", "failed"]:
                break

            await asyncio.sleep(0.1)

    async def get_task_info(self, task_id: str) -> Optional[Dict]:
        """Get detailed information about a task"""
        if task_id not in self.active_tasks:
            return None

        task = self.active_tasks[task_id]
        model_id = task["model_id"]

        return {
            "task_id": task_id,
            "model_id": model_id,
            "state": task["state"],
            "start_time": task["start_time"],
            "end_time": task.get("end_time"),
            "error": task.get("error"),
            "cache_hit": model_id in self.model_cache,
            "gpu_resources": {
                "vgpu_id": task.get("vgpu_id"),
                "partition_id": task.get("partition_id")
            } if "vgpu_id" in task else None
        }

    def __del__(self):
        """Cleanup on deletion"""
        try:
            # Cancel background tasks
            if self.health_check_task:
                self.health_check_task.cancel()
            if self.metric_collection_task:
                self.metric_collection_task.cancel()
        except:
            pass