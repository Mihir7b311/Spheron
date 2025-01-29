from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time
import logging
from .exceptions import GPUSliceError

@dataclass
class GPUSlice:
    """Represents a slice of GPU resources"""
    slice_id: str
    gpu_id: int
    memory_mb: int
    compute_percentage: int
    owner_id: Optional[str] = None
    processes: List[int] = field(default_factory=list)
    creation_time: float = field(default_factory=time.time)
    active: bool = False
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate slice configuration"""
        if self.memory_mb <= 0:
            raise GPUSliceError("Memory allocation must be positive")
        if not 0 < self.compute_percentage <= 100:
            raise GPUSliceError("Compute percentage must be between 0 and 100")

    def add_process(self, pid: int) -> None:
        """Add process to slice"""
        if pid not in self.processes:
            self.processes.append(pid)
            logging.info(f"Added process {pid} to slice {self.slice_id}")

    def remove_process(self, pid: int) -> None:
        """Remove process from slice"""
        if pid in self.processes:
            self.processes.remove(pid)
            logging.info(f"Removed process {pid} from slice {self.slice_id}")

    def get_utilization(self) -> Dict[str, float]:
        """Get slice resource utilization"""
        return {
            "memory_utilization": len(self.processes) / (self.memory_mb or 1),
            "compute_utilization": len(self.processes) * (self.compute_percentage / 100),
            "process_count": len(self.processes)
        }

    def is_oversubscribed(self) -> bool:
        """Check if slice is oversubscribed"""
        utilization = self.get_utilization()
        return (utilization["memory_utilization"] > 1.0 or 
                utilization["compute_utilization"] > 1.0)

class GPUSliceManager:
    """Manages GPU slices"""
    def __init__(self):
        self.slices: Dict[str, GPUSlice] = {}
        self.next_slice_id = 0

    def create_slice(self,
                    gpu_id: int,
                    memory_mb: int,
                    compute_percentage: int,
                    owner_id: Optional[str] = None) -> GPUSlice:
        """Create new GPU slice"""
        try:
            slice_id = f"slice_{self.next_slice_id}"
            self.next_slice_id += 1

            slice = GPUSlice(
                slice_id=slice_id,
                gpu_id=gpu_id,
                memory_mb=memory_mb,
                compute_percentage=compute_percentage,
                owner_id=owner_id
            )

            self.slices[slice_id] = slice
            logging.info(
                f"Created GPU slice {slice_id} on GPU {gpu_id} "
                f"({memory_mb}MB, {compute_percentage}% compute)"
            )
            return slice

        except Exception as e:
            raise GPUSliceError(f"Failed to create slice: {e}")

    def delete_slice(self, slice_id: str) -> None:
        """Delete GPU slice"""
        if slice_id in self.slices:
            slice = self.slices.pop(slice_id)
            if slice.processes:
                logging.warning(
                    f"Deleted slice {slice_id} with active processes: {slice.processes}"
                )
            else:
                logging.info(f"Deleted slice {slice_id}")

    def get_slice(self, slice_id: str) -> Optional[GPUSlice]:
        """Get GPU slice by ID"""
        return self.slices.get(slice_id)

    def get_gpu_slices(self, gpu_id: int) -> List[GPUSlice]:
        """Get all slices for a GPU"""
        return [s for s in self.slices.values() if s.gpu_id == gpu_id]

    def get_owner_slices(self, owner_id: str) -> List[GPUSlice]:
        """Get all slices for an owner"""
        return [s for s in self.slices.values() if s.owner_id == owner_id]

    def get_stats(self) -> Dict:
        """Get slice statistics"""
        return {
            "total_slices": len(self.slices),
            "active_slices": len([s for s in self.slices.values() if s.active]),
            "total_processes": sum(len(s.processes) for s in self.slices.values()),
            "memory_allocated": sum(s.memory_mb for s in self.slices.values()),
            "compute_allocated": sum(s.compute_percentage for s in self.slices.values())
        }