# common/monitoring.py

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import psutil
import numpy as np
from .exceptions import MonitoringError

@dataclass
class MetricSnapshot:
    timestamp: float
    metric_name: str
    value: float
    labels: Dict[str, str]

class MetricsCollector:
    def __init__(self, max_history: int = 10000):
        self.metrics: Dict[str, List[MetricSnapshot]] = {}
        self.max_history = max_history
        self.callbacks: List[Callable] = []

    async def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        if name not in self.metrics:
            self.metrics[name] = []
            
        snapshot = MetricSnapshot(time.time(), name, value, labels or {})
        self.metrics[name].append(snapshot)
        
        if len(self.metrics[name]) > self.max_history:
            self.metrics[name].pop(0)
            
        await self._notify_callbacks(snapshot)

    async def get_metric(self, name: str, start_time: Optional[float] = None) -> List[MetricSnapshot]:
        metrics = self.metrics.get(name, [])
        if start_time:
            return [m for m in metrics if m.timestamp >= start_time]
        return metrics

class ResourceMonitor:
    def __init__(self, metrics_collector: MetricsCollector):
        self.collector = metrics_collector
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_interval = 1.0

    async def start_monitoring(self):
        if self.monitoring_task:
            return
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None

    async def _monitoring_loop(self):
        while True:
            try:
                await self._collect_system_metrics()
                await self._collect_process_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                await asyncio.sleep(5.0)

    async def _collect_system_metrics(self):
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        await self.collector.record_metric('system.cpu', cpu_percent)
        await self.collector.record_metric('system.memory', memory.percent)
        await self.collector.record_metric('system.disk', disk.percent)

    async def _collect_process_metrics(self):
        process = psutil.Process()
        await self.collector.record_metric('process.cpu', process.cpu_percent())
        await self.collector.record_metric('process.memory', process.memory_percent())

class PerformanceMonitor:
    def __init__(self, metrics_collector: MetricsCollector):
        self.collector = metrics_collector
        self._operation_times: Dict[str, List[float]] = {}

    async def record_operation(self, operation: str, duration: float):
        if operation not in self._operation_times:
            self._operation_times[operation] = []
        
        self._operation_times[operation].append(duration)
        await self.collector.record_metric(f'operation.{operation}', duration)

    def get_statistics(self, operation: str) -> Dict[str, float]:
        times = self._operation_times.get(operation, [])
        if not times:
            return {}
            
        return {
            'count': len(times),
            'mean': np.mean(times),
            'median': np.median(times),
            'p95': np.percentile(times, 95),
            'min': min(times),
            'max': max(times)
        }

class SystemHealthMonitor:
    def __init__(self, 
                 metrics_collector: MetricsCollector,
                 thresholds: Dict[str, float] = None):
        self.collector = metrics_collector
        self.thresholds = thresholds or {
            'cpu': 90.0,
            'memory': 90.0,
            'disk': 90.0
        }

    async def check_health(self) -> Dict[str, bool]:
        return {
            'cpu': await self._check_cpu(),
            'memory': await self._check_memory(),
            'disk': await self._check_disk()
        }

    async def _check_cpu(self) -> bool:
        metrics = await self.collector.get_metric('system.cpu')
        if not metrics:
            return True
        return metrics[-1].value < self.thresholds['cpu']

    async def _check_memory(self) -> bool:
        metrics = await self.collector.get_metric('system.memory')
        if not metrics:
            return True
        return metrics[-1].value < self.thresholds['memory']

    async def _check_disk(self) -> bool:
        metrics = await self.collector.get_metric('system.disk')
        if not metrics:
            return True
        return metrics[-1].value < self.thresholds['disk']