#Compute_manager.py

from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
import time

@dataclass
class ComputeAllocation:
    """Represents GPU compute allocation"""
    allocation_id: str
    process_id: int
    compute_percentage: int
    priority: int
    start_time: float
    active: bool = True

class ComputeManager:
    """Manages GPU compute resources"""
    
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.allocations: Dict[str, ComputeAllocation] = {}
        self.next_allocation_id = 0
        self.total_compute = 100  # 100% compute capacity
        self.allocated_compute = 0
        self.min_compute = 5  # Minimum 5% allocation

    def allocate_compute(self, 
                        process_id: int,
                        compute_percentage: int,
                        priority: int = 0) -> Optional[str]:
        """Allocate compute resources"""
        try:
            if compute_percentage < self.min_compute:
                compute_percentage = self.min_compute
                
            if self.allocated_compute + compute_percentage > self.total_compute:
                return None
                
            allocation_id = f"comp_{self.next_allocation_id}"
            self.next_allocation_id += 1
            
            allocation = ComputeAllocation(
                allocation_id=allocation_id,
                process_id=process_id,
                compute_percentage=compute_percentage,
                priority=priority,
                start_time=time.time()
            )
            
            self.allocations[allocation_id] = allocation
            self.allocated_compute += compute_percentage
            
            logging.info(
                f"Allocated {compute_percentage}% compute on GPU {self.gpu_id} "
                f"for process {process_id} (allocation {allocation_id})"
            )
            
            return allocation_id
            
        except Exception as e:
            logging.error(f"Compute allocation failed: {e}")
            return None

    def free_compute(self, allocation_id: str) -> bool:
        """Free compute allocation"""
        try:
            if allocation_id not in self.allocations:
                return False
                
            allocation = self.allocations[allocation_id]
            self.allocated_compute -= allocation.compute_percentage
            del self.allocations[allocation_id]
            
            logging.info(
                f"Freed compute allocation {allocation_id} "
                f"({allocation.compute_percentage}%) on GPU {self.gpu_id}"
            )
            return True
            
        except Exception as e:
            logging.error(f"Failed to free compute allocation: {e}")
            return False

    def adjust_allocation(self, 
                        allocation_id: str,
                        new_percentage: int) -> bool:
        """Adjust compute allocation percentage"""
        try:
            if allocation_id not in self.allocations:
                return False
                
            allocation = self.allocations[allocation_id]
            
            # Calculate new total allocation
            new_total = (
                self.allocated_compute - 
                allocation.compute_percentage +
                new_percentage
            )
            
            if new_total > self.total_compute:
                return False
                
            # Update allocation
            old_percentage = allocation.compute_percentage
            allocation.compute_percentage = new_percentage
            self.allocated_compute = new_total
            
            logging.info(
                f"Adjusted compute allocation {allocation_id} from "
                f"{old_percentage}% to {new_percentage}% on GPU {self.gpu_id}"
            )
            return True
            
        except Exception as e:
            logging.error(f"Failed to adjust compute allocation: {e}")
            return False

    def get_process_allocations(self, process_id: int) -> List[Dict]:
        """Get allocations for process"""
        return [{
            "allocation_id": a.allocation_id,
            "compute_percentage": a.compute_percentage,
            "priority": a.priority,
            "start_time": a.start_time
        } for a in self.allocations.values()
        if a.process_id == process_id and a.active]

    def get_allocation_info(self, allocation_id: str) -> Optional[Dict]:
        """Get allocation information"""
        allocation = self.allocations.get(allocation_id)
        if not allocation:
                return None
                
        return {
                "allocation_id": allocation.allocation_id,
                "process_id": allocation.process_id,
                "compute_percentage": allocation.compute_percentage,
                "priority": allocation.priority,
                "start_time": allocation.start_time,
                "active": allocation.active,
                "duration": time.time() - allocation.start_time
            }

    def rebalance_allocations(self) -> None:
        """Rebalance compute allocations based on priority"""
        try:
            if not self.allocations:
                return
                
            # Sort allocations by priority
            sorted_allocs = sorted(
                self.allocations.values(),
                key=lambda x: (-x.priority, x.start_time)
            )
            
            # Calculate fair share
            fair_share = self.total_compute / len(sorted_allocs)
            
            # Adjust allocations
            for allocation in sorted_allocs:
                if allocation.compute_percentage > fair_share:
                    self.adjust_allocation(
                        allocation.allocation_id,
                        int(fair_share)
                    )
                    
            logging.info(f"Rebalanced compute allocations on GPU {self.gpu_id}")
            
        except Exception as e:
            logging.error(f"Failed to rebalance allocations: {e}")

    def get_stats(self) -> Dict:
        """Get compute statistics"""
        return {
            "total_compute": self.total_compute,
            "allocated_compute": self.allocated_compute,
            "available_compute": self.total_compute - self.allocated_compute,
            "allocation_count": len(self.allocations),
            "utilization": self.allocated_compute / self.total_compute
        }

    def list_allocations(self) -> List[Dict]:
        """List all compute allocations"""
        return [{
            "allocation_id": a.allocation_id,
            "process_id": a.process_id,
            "compute_percentage": a.compute_percentage,
            "priority": a.priority,
            "duration": time.time() - a.start_time,
            "active": a.active
        } for a in self.allocations.values()]
