# src/integration/resource_manager.py

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from ..cache_system import ModelCache
from ..execution_engine.cuda import CUDAContextManager, CUDAMemoryManager
from ..gpu_sharing import VirtualGPUManager, SpaceSharingManager
from ..common.exceptions import ResourceError
from ..common.monitoring import ResourceMonitor
from ..common.metrics import MetricsCollector

class ResourceType(Enum):
    """Types of manageable resources"""
    GPU_COMPUTE = "gpu_compute"
    GPU_MEMORY = "gpu_memory"
    CACHE = "cache"
    VGPU = "vgpu"
    PARTITION = "partition"

@dataclass
class ResourceRequest:
    """Resource request specification"""
    resource_type: ResourceType
    amount: int  # Memory in MB or compute in percentage
    priority: int
    owner_id: str
    gpu_id: Optional[int] = None
    vgpu_id: Optional[str] = None
    constraints: Optional[Dict] = None

@dataclass
class ResourceAllocation:
    """Resource allocation details"""
    request_id: str
    resource_type: ResourceType
    allocated_amount: int
    gpu_id: Optional[int]
    vgpu_id: Optional[str]
    partition_id: Optional[str]
    start_time: float
    status: str = "active"

class ResourceManager:
    def __init__(self, config: Dict = None):
        """Initialize Resource Manager
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Component instances
        self.cuda_manager = CUDAContextManager()
        self.cuda_memory = CUDAMemoryManager(self.config)
        self.vgpu_manager = VirtualGPUManager(self.config)
        self.space_manager = SpaceSharingManager(self.config)
        self.model_cache = None  # Will be initialized per GPU
        
        # Monitoring
        self.resource_monitor = ResourceMonitor()
        self.metrics_collector = MetricsCollector()
        
        # Resource tracking
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.resource_limits = self._init_resource_limits()
        self.reserved_resources: Dict[str, Dict] = {}
        
        # Background tasks
        self.monitor_task = None
        self.cleanup_task = None
        
        # Configuration
        self.oversubscription_ratio = self.config.get('oversubscription_ratio', 1.0)
        self.allocation_timeout = self.config.get('allocation_timeout', 30.0)
        self.cleanup_interval = self.config.get('cleanup_interval', 60.0)

    def _init_resource_limits(self) -> Dict:
        """Initialize resource limits"""
        try:
            device_count = torch.cuda.device_count()
            limits = {}
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                limits[i] = {
                    "total_memory": props.total_memory,
                    "max_compute": 100,  # 100%
                    "max_partitions": self.config.get('max_partitions_per_gpu', 8),
                    "min_partition_size": self.config.get('min_partition_size', 256)  # MB
                }
                
            return limits
            
        except Exception as e:
            logging.error(f"Failed to initialize resource limits: {e}")
            raise ResourceError(f"Resource limits initialization failed: {e}")

    async def initialize(self):
        """Initialize resource management"""
        try:
            logging.info("Initializing resource management...")
            
            # Initialize CUDA components
            await self.cuda_manager._initialize()
            
            # Initialize monitoring
            await self.resource_monitor.start_monitoring()
            
            # Start background tasks
            self.monitor_task = asyncio.create_task(self._resource_monitoring_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            # Initialize cache per GPU
            for gpu_id in self.resource_limits:
                self.model_cache = ModelCache(
                    gpu_id=gpu_id,
                    capacity_gb=self.config.get('cache_capacity_gb', 8)
                )
                
            logging.info("Resource management initialization completed")
            
        except Exception as e:
            logging.error(f"Resource management initialization failed: {e}")
            raise ResourceError(f"Initialization failed: {e}")

    async def request_resources(self, request: ResourceRequest) -> Dict:
        """Request resource allocation
        Args:
            request: Resource request specification
        Returns:
            Allocation details dictionary
        """
        try:
            # Validate request
            self._validate_request(request)
            
            # Generate request ID
            request_id = f"alloc_{len(self.allocations)}_{request.owner_id}"
            
            # Find suitable GPU if not specified
            if request.gpu_id is None and request.resource_type in [
                ResourceType.GPU_COMPUTE, ResourceType.GPU_MEMORY
            ]:
                request.gpu_id = await self._find_suitable_gpu(request)
            
            # Check resource availability
            if not await self._check_resource_availability(request):
                raise ResourceError("Insufficient resources available")
                
            # Allocate resources
            allocation = await self._allocate_resources(request_id, request)
            
            # Record allocation
            self.allocations[request_id] = allocation
            
            # Update metrics
            await self.metrics_collector.record_allocation({
                "request_id": request_id,
                "resource_type": request.resource_type.value,
                "amount": request.amount,
                "gpu_id": request.gpu_id,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            return self._get_allocation_info(allocation)
            
        except Exception as e:
            logging.error(f"Resource request failed: {e}")
            raise ResourceError(f"Resource request failed: {e}")

    def _validate_request(self, request: ResourceRequest):
        """Validate resource request"""
        if request.amount <= 0:
            raise ResourceError("Invalid resource amount requested")
            
        if request.resource_type in [ResourceType.GPU_COMPUTE, ResourceType.GPU_MEMORY]:
            if request.gpu_id is not None and request.gpu_id not in self.resource_limits:
                raise ResourceError(f"Invalid GPU ID: {request.gpu_id}")
                
        if request.resource_type == ResourceType.GPU_COMPUTE:
            if request.amount > 100:
                raise ResourceError("Invalid compute percentage requested")
                
        if request.resource_type == ResourceType.PARTITION:
            if request.amount < self.config.get('min_partition_size', 256):
                raise ResourceError("Partition size below minimum")

    async def _find_suitable_gpu(self, request: ResourceRequest) -> int:
        """Find suitable GPU for allocation"""
        best_gpu = None
        best_score = float('inf')
        
        for gpu_id, limits in self.resource_limits.items():
            # Check if GPU can accommodate request
            if not await self._can_allocate(gpu_id, request):
                continue
                
            # Calculate suitability score
            score = await self._calculate_gpu_score(gpu_id, request)
            
            if score < best_score:
                best_score = score
                best_gpu = gpu_id
                
        if best_gpu is None:
            raise ResourceError("No suitable GPU found")
            
        return best_gpu

    async def _can_allocate(self, gpu_id: int, request: ResourceRequest) -> bool:
        """Check if GPU can accommodate allocation"""
        try:
            limits = self.resource_limits[gpu_id]
            reserved = self.reserved_resources.get(str(gpu_id), {})
            
            if request.resource_type == ResourceType.GPU_MEMORY:
                available = limits["total_memory"] - reserved.get("memory", 0)
                return request.amount <= available
                
            elif request.resource_type == ResourceType.GPU_COMPUTE:
                available = 100 - reserved.get("compute", 0)
                return request.amount <= available
                
            return True
            
        except Exception as e:
            logging.error(f"Allocation check failed: {e}")
            return False

    async def _calculate_gpu_score(self, gpu_id: int, request: ResourceRequest) -> float:
        """Calculate GPU suitability score (lower is better)"""
        try:
            # Get GPU metrics
            metrics = await self.resource_monitor.collect_metrics()
            gpu_metrics = metrics.get(gpu_id, {})
            
            # Calculate utilization score
            utilization = gpu_metrics.get("utilization", 0) / 100
            memory_usage = gpu_metrics.get("memory_used", 0) / gpu_metrics.get("memory_total", 1)
            
            # Calculate fragmentation score if applicable
            if request.resource_type == ResourceType.GPU_MEMORY:
                frag_info = self.space_manager.get_fragmentation_info(gpu_id)
                fragmentation = frag_info["fragmentation_ratio"]
            else:
                fragmentation = 0
                
            # Combine scores (weighted)
            score = (
                (utilization * 0.4) +
                (memory_usage * 0.4) +
                (fragmentation * 0.2)
            )
            
            return score
            
        except Exception as e:
            logging.error(f"Score calculation failed: {e}")
            return float('inf')

    async def _allocate_resources(self, request_id: str, request: ResourceRequest) -> ResourceAllocation:
        """Allocate requested resources"""
        try:
            allocation = None
            
            if request.resource_type == ResourceType.VGPU:
                # Allocate virtual GPU
                vgpu = await self.vgpu_manager.create_virtual_gpu({
                    "memory_mb": request.amount,
                    "compute_percentage": request.constraints.get("compute_percentage", 50),
                    "priority": request.priority
                })
                
                allocation = ResourceAllocation(
                    request_id=request_id,
                    resource_type=ResourceType.VGPU,
                    allocated_amount=request.amount,
                    gpu_id=vgpu["gpu_id"],
                    vgpu_id=vgpu["id"],
                    partition_id=None,
                    start_time=asyncio.get_event_loop().time()
                )
                
            elif request.resource_type == ResourceType.PARTITION:
                # Allocate memory partition
                partition = await self.space_manager.create_partition(
                    gpu_id=request.gpu_id,
                    size_mb=request.amount,
                    owner_id=request.owner_id,
                    vgpu_id=request.vgpu_id
                )
                
                allocation = ResourceAllocation(
                    request_id=request_id,
                    resource_type=ResourceType.PARTITION,
                    allocated_amount=request.amount,
                    gpu_id=request.gpu_id,
                    vgpu_id=request.vgpu_id,
                    partition_id=partition["id"],
                    start_time=asyncio.get_event_loop().time()
                )
                
            else:
                # Update reserved resources
                gpu_key = str(request.gpu_id)
                if gpu_key not in self.reserved_resources:
                    self.reserved_resources[gpu_key] = {}
                    
                if request.resource_type == ResourceType.GPU_MEMORY:
                    self.reserved_resources[gpu_key]["memory"] = (
                        self.reserved_resources[gpu_key].get("memory", 0) + request.amount
                    )
                elif request.resource_type == ResourceType.GPU_COMPUTE:
                    self.reserved_resources[gpu_key]["compute"] = (
                        self.reserved_resources[gpu_key].get("compute", 0) + request.amount
                    )
                    
                allocation = ResourceAllocation(
                    request_id=request_id,
                    resource_type=request.resource_type,
                    allocated_amount=request.amount,
                    gpu_id=request.gpu_id,
                    vgpu_id=None,
                    partition_id=None,
                    start_time=asyncio.get_event_loop().time()
                )
                
            return allocation
            
        except Exception as e:
            logging.error(f"Resource allocation failed: {e}")
            raise ResourceError(f"Resource allocation failed: {e}")

    async def release_resources(self, request_id: str) -> bool:
        """Release allocated resources
        Args:
            request_id: Resource allocation to release
        Returns:
            Success status
        """
        try:
            if request_id not in self.allocations:
                return False
                
            allocation = self.allocations[request_id]
            
            if allocation.resource_type == ResourceType.VGPU:
                # Release virtual GPU
                if allocation.vgpu_id:
                    await self.vgpu_manager.release_virtual_gpu(allocation.vgpu_id)
                    
            elif allocation.resource_type == ResourceType.PARTITION:
                # Release memory partition
                if allocation.partition_id:
                    await self.space_manager.release_partition(allocation.partition_id)
                    
            else:
                # Update reserved resources
                gpu_key = str(allocation.gpu_id)
                if gpu_key in self.reserved_resources:
                    if allocation.resource_type == ResourceType.GPU_MEMORY:
                        self.reserved_resources[gpu_key]["memory"] -= allocation.allocated_amount
                    elif allocation.resource_type == ResourceType.GPU_COMPUTE:
                        self.reserved_resources[gpu_key]["compute"] -= allocation.allocated_amount
                        
            # Remove allocation
            del self.allocations[request_id]
            
            logging.info(f"Released resources for request {request_id}")
            return True
            
        except Exception as e:
            logging.error(f"Resource release failed: {e}")
            raise ResourceError(f"Resource release failed: {e}")

    def _get_allocation_info(self, allocation: ResourceAllocation) -> Dict:
        """Get allocation information"""
        return {
            "request_id": allocation.request_id,
            "resource_type": allocation.resource_type.value,
            "allocated_amount": allocation.allocated_amount,
            "gpu_id": allocation.gpu_id,
            "vgpu_id": allocation.vgpu_id,
            "partition_id": allocation.partition_id,
            "start_time": allocation.start_time,
            "status": allocation.status
        }

    async def _resource_monitoring_loop(self):
        """Monitor resource usage and handle issues"""
        while True:
            try:
                await asyncio.sleep(1.0)  # 1 second interval
                
                # Collect metrics
                metrics = await self.resource_monitor.collect_metrics()
                
                # Check resource usage
                for gpu_id, gpu_metrics in metrics.items():
                    # Check memory usage
                    memory_usage = gpu_metrics.memory_used / gpu_metrics.memory_total
                    if memory_usage > 0.9:
                        logging.warning(f"High memory usage on GPU {gpu_id}: {memory_usage*100:.1f}%")
                        await self._handle_high_memory_usage(gpu_id)
                        
                    # Check compute utilization
                    if gpu_metrics.utilization > 90:
                        logging.warning(f"High compute utilization on GPU {gpu_id}: {gpu_metrics.utilization}%")
                        await self._handle_high_compute_usage(gpu_id)

                # Check allocation timeouts
                current_time = asyncio.get_event_loop().time()
                for alloc_id, allocation in list(self.allocations.items()):
                    if (current_time - allocation.start_time) > self.allocation_timeout:
                        logging.warning(f"Allocation {alloc_id} timeout")
                        await self._handle_allocation_timeout(alloc_id, allocation)

                # Check fragmentation
                for gpu_id in self.resource_limits:
                    frag_info = self.space_manager.get_fragmentation_info(gpu_id)
                    if frag_info["fragmentation_ratio"] > 0.3:  # 30% fragmentation
                        logging.info(f"High memory fragmentation on GPU {gpu_id}")
                        await self._handle_fragmentation(gpu_id)

                # Update metrics
                await self.metrics_collector.record_resource_metrics(metrics)

            except Exception as e:
                logging.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(5.0)  # Wait before retry

    async def _handle_high_memory_usage(self, gpu_id: int):
        """Handle high memory usage situation"""
        try:
            # Get allocations for this GPU
            gpu_allocations = [
                alloc for alloc in self.allocations.values()
                if alloc.gpu_id == gpu_id
            ]

            # Try to free up cache
            if self.model_cache:
                await self.model_cache.clear()

            # Consider evicting low-priority allocations
            for allocation in sorted(gpu_allocations, key=lambda x: x.priority):
                if allocation.resource_type in [ResourceType.PARTITION, ResourceType.VGPU]:
                    await self.release_resources(allocation.request_id)
                    logging.info(f"Evicted allocation {allocation.request_id} due to high memory usage")
                    
                    # Check if memory usage is now acceptable
                    metrics = await self.resource_monitor.collect_metrics()
                    if metrics[gpu_id].memory_used / metrics[gpu_id].memory_total < 0.8:
                        break

        except Exception as e:
            logging.error(f"Failed to handle high memory usage: {e}")

    async def _handle_high_compute_usage(self, gpu_id: int):
        """Handle high compute utilization situation"""
        try:
            # Get compute allocations for this GPU
            compute_allocations = [
                alloc for alloc in self.allocations.values()
                if alloc.gpu_id == gpu_id and 
                alloc.resource_type == ResourceType.GPU_COMPUTE
            ]

            # Reduce compute allocation for lower priority tasks
            for allocation in sorted(compute_allocations, key=lambda x: x.priority):
                original_amount = allocation.allocated_amount
                reduced_amount = max(original_amount * 0.8, 10)  # Don't go below 10%
                
                # Update allocation
                allocation.allocated_amount = reduced_amount
                
                # Update reserved resources
                gpu_key = str(gpu_id)
                self.reserved_resources[gpu_key]["compute"] -= (original_amount - reduced_amount)
                
                logging.info(
                    f"Reduced compute allocation {allocation.request_id} "
                    f"from {original_amount}% to {reduced_amount}%"
                )
                
                # Check if utilization is now acceptable
                metrics = await self.resource_monitor.collect_metrics()
                if metrics[gpu_id].utilization < 80:
                    break

        except Exception as e:
            logging.error(f"Failed to handle high compute usage: {e}")

    async def _handle_allocation_timeout(self, alloc_id: str, allocation: ResourceAllocation):
        """Handle allocation timeout"""
        try:
            # Check if allocation is still active
            if allocation.status != "active":
                return

            logging.warning(
                f"Allocation {alloc_id} ({allocation.resource_type.value}) "
                f"timed out after {self.allocation_timeout}s"
            )

            # Release resources
            await self.release_resources(alloc_id)

        except Exception as e:
            logging.error(f"Failed to handle allocation timeout: {e}")

    async def _handle_fragmentation(self, gpu_id: int):
        """Handle memory fragmentation"""
        try:
            # Get memory map
            memory_map = self.space_manager.get_gpu_memory_map(gpu_id)
            
            # Check if defragmentation is needed
            if len(memory_map) > 1:
                logging.info(f"Initiating defragmentation for GPU {gpu_id}")
                
                # Perform defragmentation
                await self.space_manager._defragment_gpu(gpu_id)
                
                # Verify results
                new_frag_info = self.space_manager.get_fragmentation_info(gpu_id)
                logging.info(
                    f"Defragmentation completed. "
                    f"New fragmentation ratio: {new_frag_info['fragmentation_ratio']:.2f}"
                )

        except Exception as e:
            logging.error(f"Failed to handle fragmentation: {e}")

    async def _cleanup_loop(self):
        """Periodic cleanup of stale allocations"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                current_time = asyncio.get_event_loop().time()
                stale_allocations = []

                # Find stale allocations
                for alloc_id, allocation in self.allocations.items():
                    if current_time - allocation.start_time > self.allocation_timeout:
                        stale_allocations.append(alloc_id)

                # Cleanup stale allocations
                for alloc_id in stale_allocations:
                    try:
                        logging.info(f"Cleaning up stale allocation {alloc_id}")
                        await self.release_resources(alloc_id)
                    except Exception as e:
                        logging.error(f"Failed to cleanup allocation {alloc_id}: {e}")

            except Exception as e:
                logging.error(f"Cleanup loop error: {e}")

    def get_resource_usage(self) -> Dict:
        """Get current resource usage statistics"""
        try:
            usage = {}
            for gpu_id in self.resource_limits:
                gpu_key = str(gpu_id)
                reserved = self.reserved_resources.get(gpu_key, {})
                
                usage[gpu_id] = {
                    "memory_reserved": reserved.get("memory", 0),
                    "compute_reserved": reserved.get("compute", 0),
                    "active_allocations": len([
                        a for a in self.allocations.values()
                        if a.gpu_id == gpu_id
                    ]),
                    "partitions": self.space_manager.get_gpu_memory_map(gpu_id),
                    "fragmentation": self.space_manager.get_fragmentation_info(gpu_id)
                }

            return {
                "gpus": usage,
                "total_allocations": len(self.allocations),
                "resource_limits": self.resource_limits
            }

        except Exception as e:
            logging.error(f"Failed to get resource usage: {e}")
            raise ResourceError(f"Resource usage check failed: {e}")

    async def cleanup(self):
        """Cleanup resource management"""
        try:
            logging.info("Cleaning up resource management")

            # Cancel background tasks
            if self.monitor_task:
                self.monitor_task.cancel()
            if self.cleanup_task:
                self.cleanup_task.cancel()

            # Release all allocations
            for alloc_id in list(self.allocations.keys()):
                await self.release_resources(alloc_id)

            # Stop monitoring
            await self.resource_monitor.stop_monitoring()

            logging.info("Resource management cleanup completed")

        except Exception as e:
            logging.error(f"Resource management cleanup failed: {e}")
            raise ResourceError(f"Cleanup failed: {e}")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()