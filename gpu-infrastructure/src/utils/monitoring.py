import asyncio
from typing import Dict, List, Optional, Callable
import time
import logging
from dataclasses import dataclass

@dataclass
class MonitoringMetrics:
    timestamp: float
    gpu_id: int
    memory_used: int
    memory_total: int
    utilization: float
    temperature: int
    power_usage: int
    process_count: int

class ResourceMonitor:
    """Monitors GPU resource utilization"""
    
    def __init__(self, polling_interval: float = 1.0):
        self.polling_interval = polling_interval
        self.metrics_history: Dict[int, List[MonitoringMetrics]] = {}
        self.callbacks: List[Callable] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self.alert_thresholds = {
            "memory": 0.9,      # 90% memory usage
            "utilization": 0.9,  # 90% GPU utilization
            "temperature": 80,   # 80°C
            "power": 250        # 250W
        }

    async def start_monitoring(self) -> None:
        """Start monitoring loop"""
        if self.monitoring_task is not None:
            return
            
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logging.info("Started resource monitoring")

    async def stop_monitoring(self) -> None:
        """Stop monitoring loop"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
            logging.info("Stopped resource monitoring")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while True:
            try:
                metrics = await self.collect_metrics()
                await self._process_metrics(metrics)
                await asyncio.sleep(self.polling_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.polling_interval)

    async def collect_metrics(self) -> Dict[int, MonitoringMetrics]:
        """Collect current metrics"""
        raise NotImplementedError("Must be implemented by subclass")

    async def _process_metrics(self, metrics: Dict[int, MonitoringMetrics]) -> None:
        """Process collected metrics"""
        current_time = time.time()
        
        for gpu_id, metric in metrics.items():
            # Update history
            if gpu_id not in self.metrics_history:
                self.metrics_history[gpu_id] = []
            self.metrics_history[gpu_id].append(metric)
            
            # Limit history size
            if len(self.metrics_history[gpu_id]) > 3600:  # 1 hour at 1s interval
                self.metrics_history[gpu_id].pop(0)
                
            # Check thresholds
            await self._check_thresholds(metric)
            
            # Call callbacks
            for callback in self.callbacks:
                try:
                    await callback(metric)
                except Exception as e:
                    logging.error(f"Callback error: {e}")

    async def _check_thresholds(self, metrics: MonitoringMetrics) -> None:
        """Check metrics against thresholds"""
        alerts = []
        
        # Check memory usage
        memory_usage = metrics.memory_used / metrics.memory_total
        if memory_usage > self.alert_thresholds["memory"]:
            alerts.append(f"High memory usage: {memory_usage*100:.1f}%")
            
        # Check GPU utilization
        if metrics.utilization > self.alert_thresholds["utilization"]:
            alerts.append(f"High GPU utilization: {metrics.utilization*100:.1f}%")
            
        # Check temperature
        if metrics.temperature > self.alert_thresholds["temperature"]:
            alerts.append(f"High temperature: {metrics.temperature}°C")
            
        # Check power usage
        if metrics.power_usage > self.alert_thresholds["power"]:
            alerts.append(f"High power usage: {metrics.power_usage}W")
            
        if alerts:
            logging.warning(
                f"GPU {metrics.gpu_id} alerts: {', '.join(alerts)}"
            )

    def register_callback(self, callback: Callable) -> None:
        """Register monitoring callback"""
        self.callbacks.append(callback)

    def get_metrics(self, 
                   gpu_id: int,
                   start_time: Optional[float] = None) -> List[MonitoringMetrics]:
        """Get metrics history"""
        metrics = self.metrics_history.get(gpu_id, [])
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        return metrics

    def get_latest_metrics(self, gpu_id: int) -> Optional[MonitoringMetrics]:
        """Get latest metrics"""
        metrics = self.metrics_history.get(gpu_id, [])
        return metrics[-1] if metrics else None

    def set_threshold(self, metric: str, value: float) -> None:
        """Set alert threshold"""
        if metric not in self.alert_thresholds:
            raise ValueError(f"Invalid metric: {metric}")
        self.alert_thresholds[metric] = value