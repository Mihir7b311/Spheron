# gpu-management-layer/main.py

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Import common components
from src.common.config import ConfigurationManager
from src.common.monitoring import ResourceMonitor
from src.common.metrics import MetricsCollector, MetricType
from src.common.exceptions import *

# Import cache system components
from src.cache_system.model_cache import ModelCache
from src.cache_system.lru_manager import LRUManager
from src.cache_system.memory_tracker import MemoryTracker

# Import execution engine components
from src.execution_engine.runtime.python_executor import PythonExecutor
from src.execution_engine.runtime.cuda_context import CUDAContext
from src.execution_engine.batch_processor.batch_manager import BatchManager
from src.execution_engine.batch_processor.inference_batch import InferenceBatch
from src.execution_engine.cuda.context_manager import CUDAContextManager
from src.execution_engine.cuda.stream_manager import CUDAStreamManager

# Import GPU sharing components
from src.gpu_sharing.virtual_gpu import VirtualGPUManager
from src.gpu_sharing.time_sharing import TimeScheduler
from src.gpu_sharing.space_sharing import SpaceSharingManager

# Import integration components
from src.integration.coordinator import SystemCoordinator
from src.integration.resource_manager import ResourceManager
from src.integration.scheduler import IntegratedScheduler

class SystemState:
    """System state tracking"""
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class GPUFaaSSystem:
    """Main GPU FaaS System"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = config_dir or os.path.join(os.path.dirname(__file__), "config")
        self.components: Dict[str, Any] = {}
        self.state = SystemState.INITIALIZING
        self.component_status: Dict[str, Dict] = {}
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('gpu_faas.log')
            ]
        )
        self.logger = logging.getLogger("GPUFaaS")

    async def initialize(self):
        """Initialize system components"""
        try:
            self.logger.info("Initializing GPU FaaS System")
            
            # Initialize configuration
            self.config_manager = ConfigurationManager(self.config_dir)
            
            # Initialize metrics and monitoring
            self.metrics = MetricsCollector()
            monitor_config = self.config_manager.get_component_config("monitoring")
            self.monitor = ResourceMonitor(self.metrics, polling_interval=monitor_config.polling_interval)
            
            # Initialize CUDA components
            cuda_context = CUDAContextManager()
            stream_manager = CUDAStreamManager()
            self.components.update({
                "cuda_context": cuda_context,
                "stream_manager": stream_manager
            })
            
            # Initialize cache system
            cache_config = self.config_manager.get_component_config("cache")
            memory_tracker = MemoryTracker(device_id=0)  # Will be updated by scheduler
            lru_manager = LRUManager(capacity=cache_config.max_models)
            model_cache = ModelCache(gpu_id=0, capacity_gb=cache_config.capacity_gb)
            self.components.update({
                "memory_tracker": memory_tracker,
                "lru_manager": lru_manager,
                "model_cache": model_cache
            })
            
            # Initialize batch processing
            exec_config = self.config_manager.get_component_config("execution")
            batch_manager = BatchManager(stream_manager)
            inference_batch = InferenceBatch(batch_size=exec_config.max_batch_size)
            self.components.update({
                "batch_manager": batch_manager,
                "inference_batch": inference_batch
            })
            
            # Initialize execution engine
            python_executor = PythonExecutor(cuda_context, batch_manager)
            self.components["executor"] = python_executor
            
            # Initialize GPU sharing
            sharing_config = self.config_manager.get_component_config("sharing")
            vgpu_manager = VirtualGPUManager(sharing_config)
            time_scheduler = TimeScheduler(sharing_config)
            space_manager = SpaceSharingManager(sharing_config)
            self.components.update({
                "vgpu_manager": vgpu_manager,
                "time_scheduler": time_scheduler,
                "space_manager": space_manager
            })
            
            # Initialize integration components
            resource_manager = ResourceManager(self.config_manager)
            scheduler = IntegratedScheduler(self.config_manager)
            coordinator = SystemCoordinator(
                self.components,
                self.config_manager,
                self.monitor,
                self.metrics
            )
            self.components.update({
                "resource_manager": resource_manager,
                "scheduler": scheduler,
                "coordinator": coordinator
            })
            
            # Initialize component status
            await self._init_component_status()
            
            # Start monitoring
            await self.monitor.start_monitoring()
            
            self.state = SystemState.RUNNING
            self.logger.info("System initialization completed")
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.logger.error(f"Initialization failed: {e}")
            raise GPUFaaSError(f"System initialization failed: {e}")

    async def _init_component_status(self):
        """Initialize component status tracking"""
        for name, component in self.components.items():
            self.component_status[name] = {
                "state": "initialized",
                "healthy": True,
                "last_check": asyncio.get_event_loop().time(),
                "error": None
            }

    async def start(self):
        """Start system operation"""
        if self.state != SystemState.RUNNING:
            await self.initialize()
            
        try:
            self.state = SystemState.STARTING
            self.logger.info("Starting system operation")
            
            # Start coordinator
            await self.components["coordinator"].start()
            
            # Start resource manager
            await self.components["resource_manager"].start()
            
            # Start scheduler
            await self.components["scheduler"].start()
            
            # Start batch processor
            await self.components["batch_manager"].start()
            
            self.state = SystemState.RUNNING
            self.logger.info("System started successfully")
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.logger.error(f"System startup failed: {e}")
            await self.shutdown()
            raise GPUFaaSError(f"System startup failed: {e}")

    async def shutdown(self):
        """Shutdown system"""
        if self.state == SystemState.SHUTDOWN:
            return
            
        self.state = SystemState.STOPPING
        self.logger.info("Initiating system shutdown")
        
        try:
            # Stop coordinator first
            if "coordinator" in self.components:
                await self.components["coordinator"].stop()
            
            # Stop monitoring
            if hasattr(self, 'monitor'):
                await self.monitor.stop_monitoring()
            
            # Cleanup resources
            await self.cleanup_resources()
            
            # Clear component references
            self.components.clear()
            self.component_status.clear()
            
            self.state = SystemState.SHUTDOWN
            self.logger.info("System shutdown completed")
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.logger.error(f"Shutdown error: {e}")
            raise GPUFaaSError(f"System shutdown failed: {e}")

    async def cleanup_resources(self):
        """Cleanup system resources"""
        cleanup_order = [
            "batch_manager",
            "inference_batch",
            "executor",
            "model_cache",
            "vgpu_manager",
            "space_manager",
            "time_scheduler",
            "stream_manager",
            "cuda_context"
        ]
        
        for component in cleanup_order:
            try:
                if component in self.components:
                    comp = self.components[component]
                    if hasattr(comp, 'cleanup'):
                        await comp.cleanup()
                    elif hasattr(comp, 'close'):
                        await comp.close()
                    self.logger.info(f"Cleaned up {component}")
            except Exception as e:
                self.logger.error(f"Failed to cleanup {component}: {e}")

    def get_component_status(self, component_name: str) -> Dict:
        """Get status of specific component"""
        if component_name not in self.components:
            raise ValueError(f"Unknown component: {component_name}")
            
        status = self.component_status.get(component_name, {})
        component = self.components[component_name]
        
        # Get component-specific status
        if hasattr(component, 'get_stats'):
            status.update(component.get_stats())
        
        # Add metrics if available
        try:
            status["metrics"] = self.metrics.get_component_metrics(component_name)
        except:
            pass
            
        return status

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        try:
            return {
                "state": self.state,
                "components": {
                    name: self.get_component_status(name)
                    for name in self.components
                },
                "metrics": self.metrics.get_recent_metrics(10)
            }
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {
                "state": self.state,
                "error": str(e)
            }

    async def health_check(self) -> Dict[str, bool]:
        """Check system health"""
        try:
            # Check component health
            component_health = {}
            for name, component in self.components.items():
                try:
                    if hasattr(component, 'health_check'):
                        health = await component.health_check()
                    else:
                        health = bool(component)
                    
                    component_health[name] = health
                    self.component_status[name]["healthy"] = health
                    self.component_status[name]["last_check"] = asyncio.get_event_loop().time()
                    
                except Exception as e:
                    component_health[name] = False
                    self.component_status[name]["healthy"] = False
                    self.component_status[name]["error"] = str(e)
            
            # Overall system health
            system_healthy = all(component_health.values())
            
            return {
                "healthy": system_healthy,
                "components": component_health
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e)
            }

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.shutdown()

async def main():
    """Main entry point"""
    system = GPUFaaSSystem()
    
    try:
        async with system:
            # System is now running
            while True:
                # Periodically check system health
                health = await system.health_check()
                if not health.get("healthy", False):
                    logging.error(f"System unhealthy: {health.get('error')}")
                    break
                    
                await asyncio.sleep(10)  # Health check interval
                
    except KeyboardInterrupt:
        logging.info("Received shutdown signal")
    except Exception as e:
        logging.error(f"System error: {e}")
    finally:
        await system.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.critical(f"Fatal error: {e}")
        sys.exit(1)