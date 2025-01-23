# src/common/monitoring.py

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import pynvml
import torch

from .exceptions import MonitoringError
from .metrics import MetricsCollector, MetricType

@dataclass
class GPUMetrics:
    """GPU metrics data structure"""
    gpu_id: int
    timestamp: float
    utilization: float  # GPU utilization percentage
    memory_used: int   # Memory used in bytes
    memory_total: int  # Total memory in bytes
    temperature: int   # Temperature in Celsius
    power_usage: int   # Power usage in milliwatts
    compute_processes: int  # Number of compute processes
    memory_processes: int   # Number of memory processes

@dataclass
class ComponentStatus:
    """Component status information"""
    name: str
    healthy: bool
    last_check: float
    last_error: Optional[str] = None
    metrics: Optional[Dict] = None

class ResourceMonitor:
    """Unified resource monitoring system"""
    
    def __init__(self, 
                 metrics_collector: MetricsCollector,
                 polling_interval: float = 1.0):
        """Initialize Resource Monitor
        Args:
            metrics_collector: Metrics collection system
            polling_interval: Monitoring interval in seconds
        """
        self.metrics = metrics_collector
        self.polling_interval = polling_interval
        self.monitoring_task = None
        self.component_status: Dict[str, ComponentStatus] = {}
        self.alert_callbacks: Dict[str, List[Callable]] = {}
        
        # Configure thresholds
        self.thresholds = {
            "gpu_utilization": 90.0,  # 90% GPU utilization
            "memory_usage": 90.0,     # 90% memory usage  
            "temperature": 80,        # 80°C temperature
            "power_usage": 250        # 250W power usage
        }
        
        self._initialize_monitoring()

    def _initialize_monitoring(self):
        """Initialize monitoring system"""
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            logging.info(f"Initialized monitoring for {self.device_count} GPUs")
            
        except Exception as e:
            logging.error(f"Failed to initialize monitoring: {e}")
            raise MonitoringError(f"Monitoring initialization failed: {e}")

    async def start_monitoring(self):
        """Start continuous monitoring"""
        if self.monitoring_task is not None:
            return

        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logging.info("Started resource monitoring")

    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
            logging.info("Stopped resource monitoring")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Collect GPU metrics
                gpu_metrics = await self.collect_gpu_metrics()
                
                # Collect component status
                await self.check_component_health()
                
                # Check thresholds and trigger alerts
                await self.check_alerts(gpu_metrics)
                
                # Record metrics
                await self._record_metrics(gpu_metrics)
                
                # Wait for next interval
                await asyncio.sleep(self.polling_interval)
                
            except asyncio.CancelledError:
                logging.info("Monitoring loop cancelled")
                break
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5.0)  # Back off on error

    async def collect_gpu_metrics(self) -> Dict[int, GPUMetrics]:
        """Collect current metrics from all GPUs"""
        try:
            current_metrics = {}
            timestamp = time.time()
            
            for gpu_id in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                
                # Collect basic metrics
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                power = pynvml.nvmlDeviceGetPowerUsage(handle)
                
                # Count processes
                compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                memory_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                
                metrics = GPUMetrics(
                    gpu_id=gpu_id,
                    timestamp=timestamp,
                    utilization=utilization.gpu,
                    memory_used=memory.used,
                    memory_total=memory.total,
                    temperature=temperature,
                    power_usage=power,
                    compute_processes=len(compute_procs),
                    memory_processes=len(memory_procs)
                )
                
                current_metrics[gpu_id] = metrics
                
            return current_metrics

        except Exception as e:
            logging.error(f"Failed to collect GPU metrics: {e}")
            raise MonitoringError(f"GPU metrics collection failed: {e}")

    async def register_component(self, name: str):
        """Register component for monitoring"""
        self.component_status[name] = ComponentStatus(
            name=name,
            healthy=True,
            last_check=time.time()
        )

    async def check_component_health(self):
        """Check health of all registered components"""
        for name, status in self.component_status.items():
            try:
                # Component-specific health checks
                if name == "cache_system":
                    healthy = await self._check_cache_health()
                elif name == "scheduler":
                    healthy = await self._check_scheduler_health()
                elif name == "execution_engine":
                    healthy = await self._check_execution_health()
                else:
                    healthy = True  # Default to healthy
                
                # Update status
                status.healthy = healthy
                status.last_check = time.time()
                status.last_error = None if healthy else "Health check failed"
                
            except Exception as e:
                status.healthy = False
                status.last_error = str(e)
                logging.error(f"Health check failed for {name}: {e}")

    async def check_alerts(self, gpu_metrics: Dict[int, GPUMetrics]):
        """Check metrics against thresholds and trigger alerts"""
        alerts = []
        
        for gpu_id, metrics in gpu_metrics.items():
            # Check GPU utilization
            if metrics.utilization > self.thresholds["gpu_utilization"]:
                alerts.append({
                    "gpu_id": gpu_id,
                    "type": "utilization",
                    "value": metrics.utilization,
                    "threshold": self.thresholds["gpu_utilization"]
                })
            
            # Check memory usage
            memory_usage = (metrics.memory_used / metrics.memory_total) * 100
            if memory_usage > self.thresholds["memory_usage"]:
                alerts.append({
                    "gpu_id": gpu_id,
                    "type": "memory",
                    "value": memory_usage,
                    "threshold": self.thresholds["memory_usage"]
                })
            
            # Check temperature
            if metrics.temperature > self.thresholds["temperature"]:
                alerts.append({
                    "gpu_id": gpu_id,
                    "type": "temperature",
                    "value": metrics.temperature,
                    "threshold": self.thresholds["temperature"]
                })
            
            # Check power usage
            if metrics.power_usage > self.thresholds["power_usage"] * 1000:
                alerts.append({
                    "gpu_id": gpu_id,
                    "type": "power",
                    "value": metrics.power_usage / 1000,
                    "threshold": self.thresholds["power_usage"]
                })

        # Trigger alert callbacks
        for alert in alerts:
            await self._trigger_alert(alert)

    async def register_alert_callback(self, 
                                    alert_type: str, 
                                    callback: Callable):
        """Register callback for alert type"""
        if alert_type not in self.alert_callbacks:
            self.alert_callbacks[alert_type] = []
        self.alert_callbacks[alert_type].append(callback)

    async def _trigger_alert(self, alert: Dict):
        """Trigger alert callbacks"""
        alert_type = alert["type"]
        if alert_type in self.alert_callbacks:
            for callback in self.alert_callbacks[alert_type]:
                try:
                    await callback(alert)
                except Exception as e:
                    logging.error(f"Alert callback failed: {e}")

    async def _record_metrics(self, gpu_metrics: Dict[int, GPUMetrics]):
        """Record collected metrics"""
        try:
            for gpu_id, metrics in gpu_metrics.items():
                labels = {"gpu_id": str(gpu_id)}
                
                # Record GPU metrics
                await self.metrics.record_metric(
                    MetricType.GPU_UTILIZATION,
                    metrics.utilization,
                    labels
                )
                
                await self.metrics.record_metric(
                    MetricType.MEMORY_USAGE,
                    metrics.memory_used / metrics.memory_total * 100,
                    labels
                )
                
                # Record process metrics
                await self.metrics.record_metric(
                    MetricType.PROCESS_COUNT,
                    metrics.compute_processes,
                    labels
                )
                
        except Exception as e:
            logging.error(f"Failed to record metrics: {e}")

    async def _check_cache_health(self) -> bool:
        """Check cache system health"""
        # Implement cache health checks
        return True

    async def _check_scheduler_health(self) -> bool:
        """Check scheduler health"""
        # Implement scheduler health checks
        return True

    async def _check_execution_health(self) -> bool:
        """Check execution engine health"""
        # Implement execution health checks
        return True

    def get_component_status(self, component: str) -> Optional[ComponentStatus]:
        """Get status of specific component"""
        return self.component_status.get(component)

    def get_all_component_status(self) -> Dict[str, ComponentStatus]:
        """Get status of all components"""
        return self.component_status.copy()

    def set_threshold(self, metric: str, value: float):
        """Set alert threshold for metric"""
        if metric not in self.thresholds:
            raise ValueError(f"Invalid metric: {metric}")
        self.thresholds[metric] = value

    async def get_resource_usage(self) -> Dict:
        """Get current resource usage statistics"""
        gpu_metrics = await self.collect_gpu_metrics()
        
        return {
            gpu_id: {
                "utilization": metrics.utilization,
                "memory_used": metrics.memory_used,
                "memory_total": metrics.memory_total,
                "temperature": metrics.temperature,
                "power_usage": metrics.power_usage,
                "processes": metrics.compute_processes
            }
            for gpu_id, metrics in gpu_metrics.items()
        }

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_monitoring()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop_monitoring()