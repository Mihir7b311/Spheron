# src/common/performance_monitor.py

import time
import logging
import asyncio
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    execution_time: float
    cuda_time: float
    batch_size: int
    memory_usage: float
    compute_usage: float
    throughput: float

class PerformanceMonitor:
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.alert_thresholds = {
            "execution_time": 1000,  # ms
            "memory_usage": 0.9,     # 90%
            "compute_usage": 0.9     # 90%
        }
        
    async def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics"""
        self.metrics_history.append(metrics)
        await self._check_alerts(metrics)
        
        # Keep last 1000 records
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
            
    async def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts"""
        alerts = []
        
        if metrics.execution_time > self.alert_thresholds["execution_time"]:
            alerts.append(f"High execution time: {metrics.execution_time}ms")
            
        if metrics.memory_usage > self.alert_thresholds["memory_usage"]:
            alerts.append(f"High memory usage: {metrics.memory_usage*100}%")
            
        if metrics.compute_usage > self.alert_thresholds["compute_usage"]:
            alerts.append(f"High compute usage: {metrics.compute_usage*100}%")
            
        if alerts:
            logging.warning(f"Performance alerts: {', '.join(alerts)}")
            
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        if not self.metrics_history:
            return {}
            
        return {
            "avg_execution_time": sum(m.execution_time for m in self.metrics_history) / len(self.metrics_history),
            "avg_batch_size": sum(m.batch_size for m in self.metrics_history) / len(self.metrics_history),
            "avg_memory_usage": sum(m.memory_usage for m in self.metrics_history) / len(self.metrics_history),
            "avg_throughput": sum(m.throughput for m in self.metrics_history) / len(self.metrics_history)
        }