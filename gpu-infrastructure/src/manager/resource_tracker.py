# resource_tracker.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import logging

@dataclass
class ResourceAllocation:
    allocation_id: str
    resource_type: str
    amount: int
    owner_id: str
    timestamp: float
    gpu_id: int
    active: bool = True

class ResourceTracker:
    """Tracks GPU resource allocations"""
    
    def __init__(self):
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.gpu_resources: Dict[int, Dict] = {}
        self.next_allocation_id = 0

    def track_allocation(self, 
                        resource_type: str,
                        amount: int,
                        owner_id: str,
                        gpu_id: int) -> str:
        allocation_id = f"alloc_{self.next_allocation_id}"
        self.next_allocation_id += 1

        allocation = ResourceAllocation(
            allocation_id=allocation_id,
            resource_type=resource_type,
            amount=amount,
            owner_id=owner_id,
            timestamp=time.time(),
            gpu_id=gpu_id
        )
        
        self.allocations[allocation_id] = allocation
        self._update_gpu_resources(gpu_id, resource_type, amount, True)
        
        return allocation_id

    def release_allocation(self, allocation_id: str) -> bool:
        if allocation_id not in self.allocations:
            return False
            
        allocation = self.allocations[allocation_id]
        allocation.active = False
        self._update_gpu_resources(
            allocation.gpu_id,
            allocation.resource_type,
            allocation.amount,
            False
        )
        return True

    def _update_gpu_resources(self,
                            gpu_id: int,
                            resource_type: str,
                            amount: int,
                            is_allocation: bool):
        if gpu_id not in self.gpu_resources:
            self.gpu_resources[gpu_id] = {
                "memory": 0,
                "compute": 0
            }
            
        if is_allocation:
            self.gpu_resources[gpu_id][resource_type] += amount
        else:
            self.gpu_resources[gpu_id][resource_type] -= amount

    def get_gpu_utilization(self, gpu_id: int) -> Dict:
        if gpu_id not in self.gpu_resources:
            return {"memory": 0, "compute": 0}
        return self.gpu_resources[gpu_id].copy()

    def get_owner_allocations(self, owner_id: str) -> List[ResourceAllocation]:
        return [
            alloc for alloc in self.allocations.values()
            if alloc.owner_id == owner_id and alloc.active
        ]