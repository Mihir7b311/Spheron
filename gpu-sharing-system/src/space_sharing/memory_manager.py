# src/space_sharing/memory_manager.py

from typing import Dict, List, Optional
import pynvml
import logging
import asyncio
from dataclasses import dataclass
from ..common.exceptions import MemoryError

@dataclass
class MemoryPartition:
    """Represents a memory partition"""
    id: str
    size_mb: int
    offset: int
    in_use: bool
    owner_id: Optional[str] = None
    last_access: float = 0.0

class MemoryManager:
    def __init__(self, gpu_id: int, total_memory: int):
        self.gpu_id = gpu_id
        self.total_memory = total_memory
        self.partitions: Dict[str, MemoryPartition] = {}
        self.min_partition_size = 256  # 256MB minimum
        self.fragmentation_threshold = 0.2  # 20% fragmentation triggers optimization
        self._initialize_memory_tracking()

    def _initialize_memory_tracking(self):
        """Initialize memory tracking structures"""
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            logging.info(f"Initialized memory tracking for GPU {self.gpu_id}")
        except Exception as e:
            logging.error(f"Failed to initialize memory tracking: {e}")
            raise MemoryError(f"Memory tracking initialization failed: {e}")

    async def create_partition(self, size_mb: int, owner_id: str) -> Dict:
        """Create a new memory partition"""
        try:
            # Validate size
            if size_mb < self.min_partition_size:
                raise MemoryError(f"Partition size must be at least {self.min_partition_size}MB")

            # Check available memory
            if not self._has_sufficient_memory(size_mb):
                raise MemoryError("Insufficient memory available")

            # Find suitable location
            offset = self._find_free_space(size_mb)
            if offset is None:
                # Try defragmentation
                await self._defragment()
                offset = self._find_free_space(size_mb)
                if offset is None:
                    raise MemoryError("No suitable space found after defragmentation")

            # Create partition
            partition_id = f"part_{len(self.partitions)}_{self.gpu_id}"
            partition = MemoryPartition(
                id=partition_id,
                size_mb=size_mb,
                offset=offset,
                in_use=True,
                owner_id=owner_id
            )
            
            self.partitions[partition_id] = partition
            logging.info(f"Created partition {partition_id} of {size_mb}MB")

            return {
                "partition_id": partition_id,
                "size_mb": size_mb,
                "offset": offset,
                "owner_id": owner_id
            }

        except Exception as e:
            logging.error(f"Failed to create partition: {e}")
            raise

    async def release_partition(self, partition_id: str) -> bool:
        """Release memory partition"""
        try:
            if partition_id not in self.partitions:
                return False

            partition = self.partitions.pop(partition_id)
            logging.info(
                f"Released partition {partition_id} "
                f"({partition.size_mb}MB at offset {partition.offset})"
            )

            return True

        except Exception as e:
            logging.error(f"Failed to release partition: {e}")
            raise

    def _has_sufficient_memory(self, size_mb: int) -> bool:
        """Check if sufficient memory is available"""
        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return memory_info.free >= size_mb * 1024 * 1024
        except Exception as e:
            logging.error(f"Failed to check memory availability: {e}")
            return False

    def _find_free_space(self, size_mb: int) -> Optional[int]:
        """Find free space for new partition"""
        if not self.partitions:
            return 0

        # Sort partitions by offset
        sorted_parts = sorted(
            self.partitions.values(), 
            key=lambda x: x.offset
        )

        # Check space between partitions
        prev_end = 0
        for part in sorted_parts:
            if part.offset - prev_end >= size_mb:
                return prev_end
            prev_end = part.offset + part.size_mb

        # Check space after last partition
        if self.total_memory - prev_end >= size_mb:
            return prev_end

        return None

    async def _defragment(self):
        """Defragment memory partitions"""
        try:
            if not self._needs_defragmentation():
                return

            logging.info("Starting memory defragmentation")

            # Sort partitions by offset
            sorted_parts = sorted(
                self.partitions.values(), 
                key=lambda x: x.offset
            )

            # Compact partitions
            current_offset = 0
            for part in sorted_parts:
                if part.offset != current_offset:
                    await self._move_partition(part, current_offset)
                current_offset += part.size_mb

            logging.info("Defragmentation completed")

        except Exception as e:
            logging.error(f"Defragmentation failed: {e}")
            raise

    def _needs_defragmentation(self) -> bool:
        """Check if defragmentation is needed"""
        if not self.partitions:
            return False

        total_used = sum(p.size_mb for p in self.partitions.values())
        sorted_parts = sorted(
            self.partitions.values(), 
            key=lambda x: x.offset
        )
        
        last_offset = sorted_parts[-1].offset + sorted_parts[-1].size_mb
        fragmentation = 1 - (total_used / last_offset)

        return fragmentation > self.fragmentation_threshold

    async def get_partition_info(self, partition_id: str) -> Optional[Dict]:
        """Get information about a partition"""
        if partition_id not in self.partitions:
            return None

        partition = self.partitions[partition_id]
        return {
            "id": partition.id,
            "size_mb": partition.size_mb,
            "offset": partition.offset,
            "in_use": partition.in_use,
            "owner_id": partition.owner_id
        }

    def get_memory_map(self) -> List[Dict]:
        """Get current memory map"""
        return [
            {
                "id": p.id,
                "size_mb": p.size_mb,
                "offset": p.offset,
                "in_use": p.in_use,
                "owner_id": p.owner_id
            }
            for p in sorted(
                self.partitions.values(), 
                key=lambda x: x.offset
            )
        ]

    def get_stats(self) -> Dict:
        """Get memory statistics"""
        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return {
                "total_memory": self.total_memory,
                "used_memory": memory_info.used,
                "free_memory": memory_info.free,
                "num_partitions": len(self.partitions),
                "fragmentation": self._calculate_fragmentation(),
                "largest_free_block": self._get_largest_free_block()
            }
        except Exception as e:
            logging.error(f"Failed to get memory stats: {e}")
            raise