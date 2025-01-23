# src/common/metrics.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import time
import asyncio
import logging
import numpy as np
from .exceptions import MonitoringError

class MetricType(Enum):
    """Types of metrics tracked by the system"""
    # Cache Metrics
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_EVICTION = "cache_eviction"
    MEMORY_USAGE = "memory_usage"
    CACHE_EFFICIENCY = "cache_efficiency"
    MODEL_LOAD_TIME = "model_load_time"
    
    # Memory Metrics
    MEMORY_FRAGMENTATION = "memory_fragmentation"
    MEMORY_ALLOCATION_TIME = "memory_allocation_time"
    MEMORY_UTILIZATION = "memory_utilization"
    
    # Transfer Metrics
    DATA_TRANSFER_RATE = "data_transfer_rate"
    NETWORK_LATENCY = "network_latency"
    BANDWIDTH_USAGE = "bandwidth_usage"
    
    # Execution Metrics
    EXECUTION_TIME = "execution_time"
    CUDA_TIME = "cuda_time"
    BATCH_SIZE = "batch_size"
    THROUGHPUT = "throughput"
    
    # GPU Metrics
    GPU_UTILIZATION = "gpu_utilization"
    COMPUTE_USAGE = "compute_usage"
    MEMORY_ALLOCATED = "memory_allocated"
    
    # Scheduler Metrics
    QUEUE_LENGTH = "queue_length"
    SCHEDULING_LATENCY = "scheduling_latency"
    TASK_COUNT = "task_count"

@dataclass
class MetricValue:
    """Single metric measurement"""
    type: MetricType
    value: float
    timestamp: float
    labels: Dict[str, str] = None
    
@dataclass
class MetricAggregation:
    """Aggregated metric statistics"""
    count: int
    mean: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float

class MetricsCollector:
    """Unified metrics collection and aggregation"""
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics: Dict[MetricType, List[MetricValue]] = {
            metric_type: [] for metric_type in MetricType
        }
        self.labels: Dict[str, set] = {}
        self._lock = asyncio.Lock()

    async def record_metric(self, 
                          metric_type: MetricType, 
                          value: float,
                          labels: Dict[str, str] = None):
        """Record a single metric value
        Args:
            metric_type: Type of metric
            value: Metric value
            labels: Optional metric labels
        """
        try:
            async with self._lock:
                metric = MetricValue(
                    type=metric_type,
                    value=value,
                    timestamp=time.time(),
                    labels=labels or {}
                )
                
                metrics_list = self.metrics[metric_type]
                metrics_list.append(metric)
                
                # Maintain history limit
                if len(metrics_list) > self.max_history:
                    metrics_list.pop(0)
                    
                # Update label sets
                if labels:
                    for key, value in labels.items():
                        if key not in self.labels:
                            self.labels[key] = set()
                        self.labels[key].add(value)
                        
        except Exception as e:
            logging.error(f"Failed to record metric: {e}")
            raise MonitoringError(f"Metric recording failed: {e}")

    async def get_metrics(self, 
                         metric_type: MetricType,
                         start_time: Optional[float] = None,
                         end_time: Optional[float] = None,
                         labels: Optional[Dict[str, str]] = None) -> List[MetricValue]:
        """Get filtered metrics
        Args:
            metric_type: Type of metric to get
            start_time: Optional start time filter
            end_time: Optional end time filter
            labels: Optional label filters
        Returns:
            List of matching metrics
        """
        try:
            async with self._lock:
                metrics = self.metrics[metric_type]
                
                # Apply time filters
                if start_time is not None:
                    metrics = [m for m in metrics if m.timestamp >= start_time]
                if end_time is not None:
                    metrics = [m for m in metrics if m.timestamp <= end_time]
                    
                # Apply label filters
                if labels:
                    metrics = [
                        m for m in metrics
                        if all(
                            key in m.labels and m.labels[key] == value
                            for key, value in labels.items()
                        )
                    ]
                    
                return metrics
                
        except Exception as e:
            logging.error(f"Failed to get metrics: {e}")
            raise MonitoringError(f"Metric retrieval failed: {e}")

    async def get_aggregation(self,
                            metric_type: MetricType,
                            window: Optional[float] = None) -> MetricAggregation:
        """Get metric aggregation for time window
        Args:
            metric_type: Type of metric to aggregate
            window: Time window in seconds (None for all time)
        Returns:
            Aggregated statistics
        """
        try:
            start_time = time.time() - window if window else None
            metrics = await self.get_metrics(metric_type, start_time=start_time)
            
            if not metrics:
                return MetricAggregation(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                
            values = [m.value for m in metrics]
            
            return MetricAggregation(
                count=len(values),
                mean=float(np.mean(values)),
                min=float(np.min(values)),
                max=float(np.max(values)),
                p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)),
                p99=float(np.percentile(values, 99))
            )
            
        except Exception as e:
            logging.error(f"Failed to aggregate metrics: {e}")
            raise MonitoringError(f"Metric aggregation failed: {e}")

    async def record_task_metrics(self, task_id: str, metrics: Dict[str, float]):
        """Record metrics for a task
        Args:
            task_id: Task identifier
            metrics: Dictionary of metric values
        """
        try:
            labels = {"task_id": task_id}
            
            for metric_name, value in metrics.items():
                try:
                    metric_type = MetricType[metric_name.upper()]
                    await self.record_metric(metric_type, value, labels)
                except KeyError:
                    logging.warning(f"Unknown metric type: {metric_name}")
                    
        except Exception as e:
            logging.error(f"Failed to record task metrics: {e}")
            raise MonitoringError(f"Task metric recording failed: {e}")

    async def get_task_metrics(self, task_id: str) -> Dict[MetricType, List[MetricValue]]:
        """Get all metrics for a task
        Args:
            task_id: Task identifier
        Returns:
            Dictionary of metrics by type
        """
        try:
            labels = {"task_id": task_id}
            results = {}
            
            for metric_type in MetricType:
                metrics = await self.get_metrics(metric_type, labels=labels)
                if metrics:
                    results[metric_type] = metrics
                    
            return results
            
        except Exception as e:
            logging.error(f"Failed to get task metrics: {e}")
            raise MonitoringError(f"Task metric retrieval failed: {e}")

    def get_label_values(self, label: str) -> set:
        """Get all values seen for a label
        Args:
            label: Label key
        Returns:
            Set of label values
        """
        return self.labels.get(label, set())

    async def clear_old_metrics(self, max_age: float):
        """Clear metrics older than max_age seconds
        Args:
            max_age: Maximum age in seconds
        """
        try:
            min_timestamp = time.time() - max_age
            
            async with self._lock:
                for metric_type in self.metrics:
                    self.metrics[metric_type] = [
                        m for m in self.metrics[metric_type]
                        if m.timestamp >= min_timestamp
                    ]
                    
        except Exception as e:
            logging.error(f"Failed to clear old metrics: {e}")
            raise MonitoringError(f"Metric cleanup failed: {e}")

    async def get_recent_metrics(self, 
                               n: int = 100) -> Dict[MetricType, List[MetricValue]]:
        """Get N most recent metrics for each type"""
        try:
            async with self._lock:
                return {
                    metric_type: metrics[-n:]
                    for metric_type, metrics in self.metrics.items()
                    if metrics
                }
        except Exception as e:
            logging.error(f"Failed to get recent metrics: {e}")
            raise MonitoringError(f"Recent metric retrieval failed: {e}")

    async def calculate_rate(self,
                           metric_type: MetricType,
                           window: float = 60.0) -> float:
        """Calculate rate of a metric over time window
        Args:
            metric_type: Type of metric
            window: Time window in seconds
        Returns:
            Rate (events per second)
        """
        try:
            end_time = time.time()
            start_time = end_time - window
            metrics = await self.get_metrics(
                metric_type, 
                start_time=start_time,
                end_time=end_time
            )
            return len(metrics) / window
        except Exception as e:
            logging.error(f"Failed to calculate rate: {e}")
            raise MonitoringError(f"Rate calculation failed: {e}")

    async def get_moving_average(self,
                               metric_type: MetricType,
                               window: float = 60.0,
                               step: float = 1.0) -> List[float]:
        """Calculate moving average of metric
        Args:
            metric_type: Type of metric
            window: Window size in seconds
            step: Step size in seconds  
        Returns:
            List of moving averages
        """
        try:
            current_time = time.time()
            start_time = current_time - window
            
            metrics = await self.get_metrics(
                metric_type,
                start_time=start_time
            )
            if not metrics:
                return []
                
            values = [m.value for m in metrics]
            timestamps = [m.timestamp for m in metrics]
            
            # Calculate moving averages
            averages = []
            current = start_time
            while current <= current_time:
                window_vals = [
                    v for v, t in zip(values, timestamps)
                    if current - window <= t <= current
                ]
                if window_vals:
                    averages.append(sum(window_vals) / len(window_vals))
                current += step
                
            return averages
            
        except Exception as e:
            logging.error(f"Failed to calculate moving average: {e}")
            raise MonitoringError(f"Moving average calculation failed: {e}")

    async def analyze_trend(self,
                          metric_type: MetricType,
                          window: float = 300.0) -> Dict[str, float]:
        """Analyze trend of a metric
        Args:
            metric_type: Type of metric
            window: Analysis window in seconds
        Returns:
            Trend statistics
        """
        try:
            metrics = await self.get_metrics(
                metric_type,
                start_time=time.time() - window
            )
            if not metrics:
                return {
                    "slope": 0.0,
                    "correlation": 0.0,
                    "volatility": 0.0
                }
            
            values = np.array([m.value for m in metrics])
            times = np.array([m.timestamp for m in metrics])
            
            # Calculate trend statistics
            slope, _ = np.polyfit(times, values, 1)
            correlation = np.corrcoef(times, values)[0,1]
            volatility = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            
            return {
                "slope": float(slope),
                "correlation": float(correlation),
                "volatility": float(volatility)
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze trend: {e}")
            raise MonitoringError(f"Trend analysis failed: {e}")

    def set_alert_threshold(self,
                          metric_type: MetricType,
                          threshold: float,
                          window: float = 60.0,
                          callback: Optional[callable] = None):
        """Set alert threshold for metric
        Args:
            metric_type: Type of metric
            threshold: Alert threshold value
            window: Monitoring window in seconds
            callback: Optional callback for alerts
        """
        async def monitor():
            while True:
                try:
                    agg = await self.get_aggregation(metric_type, window)
                    if agg.mean > threshold:
                        if callback:
                            await callback(metric_type, agg.mean, threshold)
                        else:
                            logging.warning(
                                f"Metric {metric_type.value} exceeded threshold: "
                                f"{agg.mean:.2f} > {threshold:.2f}"
                            )
                    await asyncio.sleep(window / 10)  # Check 10 times per window
                except Exception as e:
                    logging.error(f"Alert monitoring failed: {e}")
                    await asyncio.sleep(10)  # Back off on error
                    
        asyncio.create_task(monitor())
        
    async def export_metrics(self, 
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None) -> Dict:
        """Export metrics to dictionary
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
        Returns:
            Dictionary of exported metrics
        """
        try:
            async with self._lock:
                export = {}
                for metric_type in MetricType:
                    metrics = await self.get_metrics(
                        metric_type,
                        start_time=start_time,
                        end_time=end_time
                    )
                    if metrics:
                        export[metric_type.value] = [
                            {
                                "value": m.value,
                                "timestamp": m.timestamp,
                                "labels": m.labels
                            }
                            for m in metrics
                        ]
                return export
                
        except Exception as e:
            logging.error(f"Failed to export metrics: {e}")
            raise MonitoringError(f"Metric export failed: {e}")

    async def import_metrics(self, metrics_data: Dict):
        """Import metrics from dictionary
        Args:
            metrics_data: Dictionary of metrics data
        """
        try:
            async with self._lock:
                for metric_name, metrics in metrics_data.items():
                    try:
                        metric_type = MetricType(metric_name)
                        for m in metrics:
                            await self.record_metric(
                                metric_type=metric_type,
                                value=m["value"],
                                labels=m.get("labels")
                            )
                    except ValueError:
                        logging.warning(f"Unknown metric type: {metric_name}")
                        
        except Exception as e:
            logging.error(f"Failed to import metrics: {e}")
            raise MonitoringError(f"Metric import failed: {e}")