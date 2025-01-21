# src/common/monitoring.py

import pynvml
from typing import Dict, List, Optional
import logging
import asyncio
from dataclasses import dataclass
from .exceptions import MonitoringError

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

class ResourceMonitor:
    def __init__(self, polling_interval: float = 1.0):
        self.polling_interval = polling_interval
        self.metrics_history: Dict[int, List[GPUMetrics]] = {}
        self.alert_thresholds = {
            "utilization": 90.0,  # Alert if GPU utilization > 90%
            "memory": 90.0,      # Alert if memory usage > 90%
            "temperature": 80,    # Alert if temperature > 80C
            "power": 250         # Alert if power usage > 250W
        }
        self.monitoring_task = None
        self._initialize_monitoring()

    def _initialize_monitoring(self):
        """Initialize NVML for monitoring"""
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                self.metrics_history[i] = []
                
            logging.info(f"Initialized monitoring for {device_count} GPUs")
            
        except pynvml.NVMLError as e:
            logging.error(f"Failed to initialize monitoring: {e}")
            raise MonitoringError(f"Monitoring initialization failed: {e}")

    async def start_monitoring(self):
        """Start continuous monitoring"""
        if self.monitoring_task is not None:
            return

        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logging.info("Started GPU monitoring")

    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        if self.monitoring_task is not None:
            self.monitoring_task.cancel()
            self.monitoring_task = None
            logging.info("Stopped GPU monitoring")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while True:
                await self.collect_metrics()
                await self.check_alerts()
                await asyncio.sleep(self.polling_interval)
                
        except asyncio.CancelledError:
            logging.info("Monitoring loop cancelled")
        except Exception as e:
            logging.error(f"Monitoring loop error: {e}")
            raise

    async def collect_metrics(self) -> Dict[int, GPUMetrics]:
        """Collect current metrics from all GPUs"""
        try:
            current_metrics = {}
            timestamp = asyncio.get_event_loop().time()
            
            for gpu_id in self.metrics_history.keys():
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
                self.metrics_history[gpu_id].append(metrics)
                
                # Keep only last hour of metrics
                if len(self.metrics_history[gpu_id]) > 3600 / self.polling_interval:
                    self.metrics_history[gpu_id].pop(0)
                    
            return current_metrics

        except Exception as e:
            logging.error(f"Failed to collect metrics: {e}")
            raise MonitoringError(f"Metrics collection failed: {e}")

    async def check_alerts(self) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []
        try:
            for gpu_id, metrics in self.metrics_history.items():
                if not metrics:
                    continue
                    
                latest = metrics[-1]
                
                # Check utilization
                if latest.utilization > self.alert_thresholds["utilization"]:
                    alerts.append({
                        "gpu_id": gpu_id,
                        "type": "utilization",
                        "value": latest.utilization,
                        "threshold": self.alert_thresholds["utilization"]
                    })
                
                # Check memory
                memory_usage = (latest.memory_used / latest.memory_total) * 100
                if memory_usage > self.alert_thresholds["memory"]:
                    alerts.append({
                        "gpu_id": gpu_id,
                        "type": "memory",
                        "value": memory_usage,
                        "threshold": self.alert_thresholds["memory"]
                    })
                
                # Check temperature
                if latest.temperature > self.alert_thresholds["temperature"]:
                    alerts.append({
                        "gpu_id": gpu_id,
                        "type": "temperature",
                        "value": latest.temperature,
                        "threshold": self.alert_thresholds["temperature"]
                    })
                
                # Check power
                if latest.power_usage > self.alert_thresholds["power"] * 1000:  # convert to mW
                    alerts.append({
                        "gpu_id": gpu_id,
                        "type": "power",
                        "value": latest.power_usage / 1000,  # convert to W
                        "threshold": self.alert_thresholds["power"]
                    })
                    
            return alerts

        except Exception as e:
            logging.error(f"Failed to check alerts: {e}")
            raise MonitoringError(f"Alert check failed: {e}")

    def get_metrics_history(self, gpu_id: int) -> List[GPUMetrics]:
        """Get metrics history for specific GPU"""
        return self.metrics_history.get(gpu_id, [])

    def get_current_metrics(self, gpu_id: Optional[int] = None) -> Dict:
        """Get current metrics for specific or all GPUs"""
        try:
            if gpu_id is not None:
                if gpu_id not in self.metrics_history:
                    return {}
                return {gpu_id: self.metrics_history[gpu_id][-1]}
            
            return {
                gpu_id: metrics[-1] 
                for gpu_id, metrics in self.metrics_history.items() 
                if metrics
            }

        except Exception as e:
            logging.error(f"Failed to get current metrics: {e}")
            raise MonitoringError(f"Metrics retrieval failed: {e}")

    def set_alert_threshold(self, metric: str, value: float):
        """Set alert threshold for specific metric"""
        if metric not in self.alert_thresholds:
            raise MonitoringError(f"Invalid metric: {metric}")
        self.alert_thresholds[metric] = value

    def __del__(self):
        """Cleanup monitoring"""
        try:
            pynvml.nvmlShutdown()
        except:
            pass