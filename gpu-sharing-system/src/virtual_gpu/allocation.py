# src/virtual_gpu/allocation.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class Allocation:
    """Resource allocation details"""
    id: str
    memory_mb: int
    compute_percentage: int
    priority: int
    process_id: Optional[int] = None  # Associated process ID
    gpu_id: int = -1  # GPU where allocation exists
    start_time: float = 0.0  # When allocation started
    last_active: float = 0.0  # Last activity timestamp

    def is_active(self) -> bool:
        """Check if allocation is currently active"""
        return self.process_id is not None

    def get_duration(self, current_time: float) -> float:
        """Get allocation duration in seconds"""
        if self.start_time == 0:
            return 0
        return current_time - self.start_time

    def to_dict(self) -> dict:
        """Convert allocation to dictionary"""
        return {
            "id": self.id,
            "memory_mb": self.memory_mb,
            "compute_percentage": self.compute_percentage,
            "priority": self.priority,
            "process_id": self.process_id,
            "gpu_id": self.gpu_id,
            "start_time": self.start_time,
            "last_active": self.last_active
        }