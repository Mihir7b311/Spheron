# src/gpu_sharing/space_sharing.py

from typing import Dict, Optional, List
import logging
from dataclasses import dataclass
from ..common.exceptions import ResourceError
from ..common.monitoring import ResourceMonitor
from ..execution_engine.cuda import CUDAContextManager, CUDAMemoryManager

@dataclass
class MemoryPartition:
    """Represents a GPU memory partition"""
    id: str
    size_mb: int
    offset: int
    gpu_id: int
    owner_id: str
    vgpu_id: str
    in_use: bool = False
    is_fragmented: bool = False
    last_access: float = 0.0

@dataclass
class IsolationConstraints:
    """Resource isolation constraints"""
    memory_limit: int         # MB
    compute_limit: int        # Percentage
    priority: int            # 0-4
    owner_id: str
    strict_isolation: bool = True

class SpaceSharingManager:
    def __init__(self, config: Dict = None):
        """Initialize Space Sharing Manager
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.partitions: Dict[str, MemoryPartition] = {}
        self.constraints: Dict[str, IsolationConstraints] = {}
        self.cuda_context_manager = CUDAContextManager()
        self.cuda_memory_manager = CUDAMemoryManager(self.config)
        self.monitor = ResourceMonitor()
        
        # Configuration
        self.min_partition_size = self.config.get("min_partition_size", 256)  # 256MB
        self.max_partitions_per_gpu = self.config.get("max_partitions_per_gpu", 8)
        self.fragmentation_threshold = self.config.get("fragmentation_threshold", 0.2)
        self.enable_defrag = self.config.get("enable_defrag", True)
        
        # Tracking
        self.gpu_partitions: Dict[int, List[str]] = {}  # GPU -> partition mappings
        self.vgpu_partitions: Dict[str, List[str]] = {} # vGPU -> partition mappings

    async def create_partition(self,
                             gpu_id: int,
                             size_mb: int,
                             owner_id: str,
                             vgpu_id: str) -> Dict:
        """Create a new memory partition
        Args:
            gpu_id: Physical GPU ID
            size_mb: Partition size in MB
            owner_id: Owner identifier
            vgpu_id: Virtual GPU identifier
        Returns:
            Partition information dictionary
        """
        try:
            # Validate size
            if size_mb < self.min_partition_size:
                raise ResourceError(
                    f"Partition size must be at least {self.min_partition_size}MB"
                )

            # Check GPU capacity
            if not await self._check_gpu_capacity(gpu_id, size_mb):
                raise ResourceError("Insufficient GPU memory")

            # Check partition limit
            if len(self.gpu_partitions.get(gpu_id, [])) >= self.max_partitions_per_gpu:
                raise ResourceError(
                    f"Maximum partitions ({self.max_partitions_per_gpu}) reached for GPU {gpu_id}"
                )

            # Find suitable location
            offset = await self._find_partition_location(gpu_id, size_mb)
            if offset is None:
                if self.enable_defrag:
                    await self._defragment_gpu(gpu_id)
                    offset = await self._find_partition_location(gpu_id, size_mb)
                if offset is None:
                    raise ResourceError("No suitable space found for partition")

            # Create partition
            partition_id = f"part_{len(self.partitions)}_{gpu_id}"
            partition = MemoryPartition(
                id=partition_id,
                size_mb=size_mb,
                offset=offset,
                gpu_id=gpu_id,
                owner_id=owner_id,
                vgpu_id=vgpu_id
            )

            # Update tracking
            self.partitions[partition_id] = partition
            if gpu_id not in self.gpu_partitions:
                self.gpu_partitions[gpu_id] = []
            self.gpu_partitions[gpu_id].append(partition_id)
            
            if vgpu_id not in self.vgpu_partitions:
                self.vgpu_partitions[vgpu_id] = []
            self.vgpu_partitions[vgpu_id].append(partition_id)

            # Allocate GPU memory
            await self.cuda_memory_manager.reserve_memory(
                gpu_id=gpu_id,
                size_bytes=size_mb * 1024 * 1024
            )

            logging.info(
                f"Created {size_mb}MB partition {partition_id} "
                f"on GPU {gpu_id} for vGPU {vgpu_id}"
            )

            return self._get_partition_info(partition_id)

        except Exception as e:
            logging.error(f"Failed to create partition: {e}")
            raise ResourceError(f"Partition creation failed: {e}")

    async def _check_gpu_capacity(self, gpu_id: int, size_mb: int) -> bool:
        """Check if GPU has capacity for new partition"""
        try:
            context = await self.cuda_context_manager.get_context(gpu_id)
            memory_info = context.get_context_info()["memory"]
            available_mb = memory_info["available"] / (1024 * 1024)
            return available_mb >= size_mb
        except Exception as e:
            logging.error(f"Failed to check GPU capacity: {e}")
            return False

    async def _find_partition_location(self, gpu_id: int, size_mb: int) -> Optional[int]:
        """Find suitable location for new partition"""
        if gpu_id not in self.gpu_partitions:
            return 0

        gpu_parts = sorted(
            [self.partitions[pid] for pid in self.gpu_partitions[gpu_id]],
            key=lambda p: p.offset
        )

        # Check gaps between partitions
        prev_end = 0
        for part in gpu_parts:
            if part.offset - prev_end >= size_mb:
                return prev_end
            prev_end = part.offset + part.size_mb

        # Check space after last partition
        context = await self.cuda_context_manager.get_context(gpu_id)
        total_memory_mb = context.get_context_info()["memory"]["total"] / (1024 * 1024)
        
        if total_memory_mb - prev_end >= size_mb:
            return prev_end

        return None

    async def _defragment_gpu(self, gpu_id: int):
        """Defragment GPU memory"""
        if not self.enable_defrag:
            return

        try:
            if gpu_id not in self.gpu_partitions:
                return

            logging.info(f"Starting defragmentation for GPU {gpu_id}")

            # Sort partitions by offset
            partitions = sorted(
                [self.partitions[pid] for pid in self.gpu_partitions[gpu_id]],
                key=lambda p: p.offset
            )

            # Compact partitions
            current_offset = 0
            for partition in partitions:
                if partition.offset != current_offset:
                    await self._move_partition(partition, current_offset)
                current_offset += partition.size_mb

            logging.info(f"Completed defragmentation for GPU {gpu_id}")

        except Exception as e:
            logging.error(f"Defragmentation failed: {e}")
            raise ResourceError(f"Defragmentation failed: {e}")

    async def _move_partition(self, partition: MemoryPartition, new_offset: int):
        """Move partition to new offset"""
        try:
            # Allocate new space
            new_buffer = await self.cuda_memory_manager.allocate_buffer(
                gpu_id=partition.gpu_id,
                size_bytes=partition.size_mb * 1024 * 1024
            )

            # Update partition
            partition.offset = new_offset
            partition.is_fragmented = False

        except Exception as e:
            logging.error(f"Failed to move partition: {e}")
            raise ResourceError(f"Partition move failed: {e}")

    async def release_partition(self, partition_id: str) -> bool:
        """Release a memory partition
        Args:
            partition_id: Partition to release
        Returns:
            Success status
        """
        try:
            if partition_id not in self.partitions:
                return False

            partition = self.partitions[partition_id]

            # Release GPU memory
            await self.cuda_memory_manager.free_buffer(
                partition.gpu_id,
                partition.size_mb * 1024 * 1024
            )

            # Update tracking
            self.gpu_partitions[partition.gpu_id].remove(partition_id)
            self.vgpu_partitions[partition.vgpu_id].remove(partition_id)
            del self.partitions[partition_id]

            logging.info(f"Released partition {partition_id}")
            return True

        except Exception as e:
            logging.error(f"Failed to release partition: {e}")
            raise ResourceError(f"Partition release failed: {e}")

    async def set_isolation_constraints(self,
                                     vgpu_id: str,
                                     constraints: IsolationConstraints) -> bool:
        """Set isolation constraints for virtual GPU
        Args:
            vgpu_id: Virtual GPU identifier
            constraints: Isolation constraints
        Returns:
            Success status
        """
        try:
            self.constraints[vgpu_id] = constraints
            logging.info(f"Set isolation constraints for vGPU {vgpu_id}")
            return True
        except Exception as e:
            logging.error(f"Failed to set constraints: {e}")
            return False

    def _get_partition_info(self, partition_id: str) -> Optional[Dict]:
        """Get information about a partition"""
        partition = self.partitions.get(partition_id)
        if not partition:
            return None

        return {
            "id": partition.id,
            "size_mb": partition.size_mb,
            "offset": partition.offset,
            "gpu_id": partition.gpu_id,
            "owner_id": partition.owner_id,
            "vgpu_id": partition.vgpu_id,
            "in_use": partition.in_use,
            "is_fragmented": partition.is_fragmented
        }

    def get_gpu_memory_map(self, gpu_id: int) -> List[Dict]:
        """Get memory map for GPU
        Args:
            gpu_id: GPU to get map for
        Returns:
            List of partition information
        """
        partitions = [
            self.partitions[pid]
            for pid in self.gpu_partitions.get(gpu_id, [])
        ]
        return [
            self._get_partition_info(p.id)
            for p in sorted(partitions, key=lambda x: x.offset)
        ]

    def get_fragmentation_info(self, gpu_id: int) -> Dict:
        """Get fragmentation information for GPU
        Args:
            gpu_id: GPU to analyze
        Returns:
            Fragmentation statistics
        """
        if gpu_id not in self.gpu_partitions:
            return {
                "fragmentation_ratio": 0.0,
                "largest_free_block": 0,
                "free_blocks": 0
            }

        partitions = sorted(
            [self.partitions[pid] for pid in self.gpu_partitions[gpu_id]],
            key=lambda p: p.offset
        )

        total_gaps = 0
        largest_gap = 0
        free_blocks = 0
        prev_end = 0

        for partition in partitions:
            gap = partition.offset - prev_end
            if gap > 0:
                total_gaps += gap
                largest_gap = max(largest_gap, gap)
                free_blocks += 1
            prev_end = partition.offset + partition.size_mb

        total_space = prev_end
        fragmentation_ratio = total_gaps / total_space if total_space > 0 else 0

        return {
            "fragmentation_ratio": fragmentation_ratio,
            "largest_free_block": largest_gap,
            "free_blocks": free_blocks
        }

    def get_stats(self) -> Dict:
        """Get overall statistics"""
        return {
            "total_partitions": len(self.partitions),
            "gpu_partitions": {
                gpu_id: len(partitions)
                for gpu_id, partitions in self.gpu_partitions.items()
            },
            "vgpu_partitions": {
                vgpu_id: len(partitions)
                for vgpu_id, partitions in self.vgpu_partitions.items()
            },
            "total_allocated_memory": sum(
                p.size_mb for p in self.partitions.values()
            ),
            "fragmentation": {
                gpu_id: self.get_fragmentation_info(gpu_id)
                for gpu_id in self.gpu_partitions
            }
        }