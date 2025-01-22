# src/common/metrics.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import time
import logging
import numpy as np
from enum import Enum

class MetricType(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    COMPUTE = "compute"
    BATCH = "batch"
    CACHE = "cache"

@dataclass
class ExecutionMetrics:
    """Metrics for single execution"""
    start_time: float
    end_time: float
    cuda_time: float
    batch_size: int
    memory_used: int
    compute_used: float
    cache_hit: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_time": self.end_time - self.start_time,
            "cuda_time": self.cuda_time,
            "batch_size": self.batch_size,
            "memory_used": self.memory_used,
            "compute_used": self.compute_used,
            "cache_hit": self.cache_hit
        }

class MetricsCollector:
    """Collects and manages execution metrics"""
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: List[ExecutionMetrics] = []
        self.aggregate_metrics: Dict[str, List[float]] = {
            MetricType.LATENCY.value: [],
            MetricType.THROUGHPUT.value: [],
            MetricType.MEMORY.value: [],
            MetricType.COMPUTE.value: [],
            MetricType.BATCH.value: [],
            MetricType.CACHE.value: []
        }
        self.alert_thresholds = {
            "latency_ms": 1000,  # 1 second
            "memory_usage": 0.9,  # 90%
            "compute_usage": 0.9  # 90%
        }

    async def record_metrics(self, metrics: ExecutionMetrics):
        """Record new metrics"""
        try:
            # Add to history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.history_size:
                self.metrics_history.pop(0)

            # Update aggregate metrics
            execution_time = metrics.end_time - metrics.start_time
            self.aggregate_metrics[MetricType.LATENCY.value].append(execution_time)
            self.aggregate_metrics[MetricType.MEMORY.value].append(metrics.memory_used)
            self.aggregate_metrics[MetricType.COMPUTE.value].append(metrics.compute_used)
            self.aggregate_metrics[MetricType.BATCH.value].append(metrics.batch_size)
            self.aggregate_metrics[MetricType.CACHE.value].append(1 if metrics.cache_hit else 0)

            # Calculate throughput (requests/second)
            if len(self.metrics_history) >= 2:
                time_window = self.metrics_history[-1].end_time - self.metrics_history[-2].end_time
                throughput = metrics.batch_size / time_window if time_window > 0 else 0
                self.aggregate_metrics[MetricType.THROUGHPUT.value].append(throughput)

            # Check for alerts
            await self._check_alerts(metrics)

        except Exception as e:
            logging.error(f"Failed to record metrics: {e}")

    async def _check_alerts(self, metrics: ExecutionMetrics):
        """Check metrics against alert thresholds"""
        alerts = []
        execution_time = (metrics.end_time - metrics.start_time) * 1000  # convert to ms

        if execution_time > self.alert_thresholds["latency_ms"]:
            alerts.append(f"High latency: {execution_time:.2f}ms")

        if metrics.memory_used > self.alert_thresholds["memory_usage"]:
            alerts.append(f"High memory usage: {metrics.memory_used*100:.1f}%")

        if metrics.compute_used > self.alert_thresholds["compute_usage"]:
            alerts.append(f"High compute usage: {metrics.compute_used*100:.1f}%")

        if alerts:
            logging.warning(f"Performance alerts: {', '.join(alerts)}")

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of metrics"""
        stats = {}
        for metric_type, values in self.aggregate_metrics.items():
            if not values:
                continue

            stats[metric_type] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99))
            }

        return stats

    def get_recent_metrics(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent metrics"""
        return [m.to_dict() for m in self.metrics_history[-n:]]

    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        cache_metrics = self.aggregate_metrics[MetricType.CACHE.value]
        if not cache_metrics:
            return 0.0
        return sum(cache_metrics) / len(cache_metrics)

    def get_average_batch_size(self) -> float:
        """Calculate average batch size"""
        batch_metrics = self.aggregate_metrics[MetricType.BATCH.value]
        if not batch_metrics:
            return 0.0
        return sum(batch_metrics) / len(batch_metrics)

    def get_throughput_stats(self) -> Dict[str, float]:
        """Get throughput statistics"""
        throughput_metrics = self.aggregate_metrics[MetricType.THROUGHPUT.value]
        if not throughput_metrics:
            return {"current": 0.0, "average": 0.0, "peak": 0.0}

        return {
            "current": throughput_metrics[-1],
            "average": sum(throughput_metrics) / len(throughput_metrics),
            "peak": max(throughput_metrics)
        }

    def clear_history(self):
        """Clear metrics history"""
        self.metrics_history.clear()
        for metric_type in self.aggregate_metrics:
            self.aggregate_metrics[metric_type].clear()

    def set_alert_threshold(self, metric: str, value: float):
        """Set alert threshold for metric"""
        if metric not in self.alert_thresholds:
            raise ValueError(f"Invalid metric: {metric}")
        self.alert_thresholds[metric] = value