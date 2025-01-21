# src/space_sharing/isolation.py

from typing import Dict, Optional, List
import logging
from dataclasses import dataclass
from ..common.exceptions import IsolationError

@dataclass
class IsolationConstraints:
    """Resource isolation constraints"""
    memory_limit: int
    compute_limit: int
    priority: int
    owner_id: str

class ResourceIsolation:
    def __init__(self):
        self.constraints: Dict[str, IsolationConstraints] = {}
        self.active_containers: Dict[str, Dict] = {}
        
    async def setup_isolation(self, 
                            container_id: str,
                            constraints: IsolationConstraints) -> Dict:
        """Setup resource isolation for a container"""
        try:
            if container_id in self.active_containers:
                raise IsolationError(f"Container {container_id} already exists")

            self.constraints[container_id] = constraints
            
            container_config = {
                "id": container_id,
                "memory_limit": constraints.memory_limit,
                "compute_limit": constraints.compute_limit,
                "priority": constraints.priority,
                "owner_id": constraints.owner_id,
                "active": True
            }
            
            self.active_containers[container_id] = container_config
            logging.info(f"Setup isolation for container {container_id}")
            
            return container_config

        except Exception as e:
            logging.error(f"Failed to setup isolation: {e}")
            raise IsolationError(f"Isolation setup failed: {e}")

    async def remove_isolation(self, container_id: str) -> bool:
        """Remove resource isolation"""
        try:
            if container_id not in self.active_containers:
                return False

            self.constraints.pop(container_id, None)
            self.active_containers.pop(container_id, None)
            
            logging.info(f"Removed isolation for container {container_id}")
            return True

        except Exception as e:
            logging.error(f"Failed to remove isolation: {e}")
            raise IsolationError(f"Isolation removal failed: {e}")

    def get_isolation_info(self, container_id: str) -> Optional[Dict]:
        """Get isolation information for container"""
        return self.active_containers.get(container_id)

    def list_active_containers(self) -> List[Dict]:
        """List all active containers"""
        return list(self.active_containers.values())

    def check_violation(self, container_id: str, 
                       memory_usage: int, compute_usage: int) -> bool:
        """Check if container violates its constraints"""
        if container_id not in self.constraints:
            return False

        constraints = self.constraints[container_id]
        return (
            memory_usage > constraints.memory_limit or
            compute_usage > constraints.compute_limit
        )

    def get_stats(self) -> Dict:
        """Get isolation statistics"""
        return {
            "active_containers": len(self.active_containers),
            "total_memory_allocated": sum(
                c.memory_limit for c in self.constraints.values()
            ),
            "total_compute_allocated": sum(
                c.compute_limit for c in self.constraints.values()
            )
        }