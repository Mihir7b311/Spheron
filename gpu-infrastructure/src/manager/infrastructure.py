# infrastructure.py

import asyncio
from typing import Dict, Optional, List
import logging
from ..base.mps_daemon import MPSDaemon
from ..base.gpu_slice import GPUSliceManager
from .state_manager import StateManager, SystemState
from .resource_tracker import ResourceTracker

class InfrastructureManager:
    """Main GPU infrastructure manager"""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.resource_tracker = ResourceTracker()
        self.slice_manager = GPUSliceManager()
        self.mps_daemons: Dict[int, MPSDaemon] = {}
        self.initialized = False

    async def initialize(self, gpu_ids: List[int]) -> None:
        """Initialize infrastructure"""
        try:
            self.state_manager.update_state(SystemState.INITIALIZING)
            
            # Initialize MPS daemons
            for gpu_id in gpu_ids:
                self.mps_daemons[gpu_id] = MPSDaemon(gpu_id)
                
            self.initialized = True
            self.state_manager.update_state(SystemState.RUNNING)
            logging.info(f"Initialized infrastructure for GPUs: {gpu_ids}")
            
        except Exception as e:
            self.state_manager.update_state(SystemState.ERROR, str(e))
            raise

    async def request_resources(self,
                              memory_mb: int,
                              compute_percentage: int,
                              owner_id: str) -> Dict:
        """Request GPU resources"""
        try:
            # Find suitable GPU
            gpu_id = self._find_suitable_gpu(memory_mb, compute_percentage)
            
            # Create GPU slice
            slice = self.slice_manager.create_slice(
                gpu_id=gpu_id,
                memory_mb=memory_mb,
                compute_percentage=compute_percentage,
                owner_id=owner_id
            )
            
            # Track allocation
            allocation_id = self.resource_tracker.track_allocation(
                resource_type="memory",
                amount=memory_mb,
                owner_id=owner_id,
                gpu_id=gpu_id
            )
            
            return {
                "allocation_id": allocation_id,
                "slice_id": slice.slice_id,
                "gpu_id": gpu_id
            }
            
        except Exception as e:
            logging.error(f"Failed to request resources: {e}")
            raise

    def _find_suitable_gpu(self, memory_mb: int, compute_percentage: int) -> int:
        """Find suitable GPU for allocation"""
        best_gpu = None
        best_utilization = float('inf')
        
        for gpu_id in self.mps_daemons:
            utilization = self.resource_tracker.get_gpu_utilization(gpu_id)
            total_utilization = (
                utilization["memory"] / memory_mb +
                utilization["compute"] / compute_percentage
            ) / 2
            
            if total_utilization < best_utilization:
                best_utilization = total_utilization
                best_gpu = gpu_id
                
        if best_gpu is None:
            raise Exception("No suitable GPU found")
            
        return best_gpu

    async def release_resources(self, allocation_id: str) -> bool:
        """Release allocated resources"""
        try:
            if self.resource_tracker.release_allocation(allocation_id):
                # Also release GPU slice
                slice = self.slice_manager.get_slice(allocation_id)
                if slice:
                    self.slice_manager.delete_slice(slice.slice_id)
                return True
            return False
            
        except Exception as e:
            logging.error(f"Failed to release resources: {e}")
            raise

    def get_status(self) -> Dict:
        """Get infrastructure status"""
        return {
            "state": self.state_manager.get_system_state(),
            "gpus": {
                gpu_id: {
                    "utilization": self.resource_tracker.get_gpu_utilization(gpu_id),
                    "mps_status": daemon.get_status()
                }
                for gpu_id, daemon in self.mps_daemons.items()
            },
            "slices": self.slice_manager.get_stats()
        }

    async def cleanup(self) -> None:
        """Cleanup infrastructure"""
        try:
            self.state_manager.update_state(SystemState.SHUTDOWN)
            
            # Stop MPS daemons
            for daemon in self.mps_daemons.values():
                await daemon.cleanup()
                
            # Clear managers
            self.slice_manager = GPUSliceManager()
            self.resource_tracker = ResourceTracker()
            
            self.initialized = False
            logging.info("Infrastructure cleaned up")
            
        except Exception as e:
            logging.error(f"Cleanup failed: {e}")
            raise

    async def __aenter__(self):
        """Async context manager entry"""
        if not self.initialized:
            await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()