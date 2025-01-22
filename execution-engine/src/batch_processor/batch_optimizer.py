# src/batch_processor/batch_optimizer.py

from typing import Dict, Any
import torch
import logging
import time
from ..common.exceptions import BatchError

class BatchOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.min_batch_size = config.get("min_batch_size", 1)
        self.max_batch_size = config.get("max_batch_size", 32)
        self.wait_timeout = config.get("wait_timeout", 0.1)
        self.performance_history = []
        
    async def optimize_batch_size(self, 
                                current_load: float,
                                memory_usage: float) -> int:
        """Optimize batch size based on current conditions"""
        try:
            # Consider system load
            if current_load > 0.8:  # High load
                return self.min_batch_size
                
            # Consider memory usage    
            if memory_usage > 0.9:  # High memory usage
                return self.min_batch_size
                
            # Analyze performance history
            if self.performance_history:
                avg_latency = sum(p["latency"] for p in self.performance_history) / len(self.performance_history)
                
                if avg_latency > 100:  # High latency (ms)
                    return max(self.min_batch_size, 
                             int(self.max_batch_size * 0.5))
                             
            return self.max_batch_size
            
        except Exception as e:
            raise BatchError(f"Batch optimization failed: {e}")
            
    def record_performance(self, 
                          batch_size: int,
                          latency: float,
                          throughput: float):
        """Record batch processing performance"""
        self.performance_history.append({
            "batch_size": batch_size,
            "latency": latency,
            "throughput": throughput,
            "timestamp": time.time()
        })
        
        # Keep last 100 records
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)