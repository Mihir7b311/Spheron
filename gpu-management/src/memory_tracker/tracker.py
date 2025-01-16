# src/memory_tracker/tracker.py
import pynvml
from typing import Dict, List
import logging
import time

class MemoryTracker:
    def __init__(self, device_id: int):
        """Initialize Memory Tracker for specific GPU"""
        self.device_id = device_id
        self.device_handle = None
        self.total_memory = 0
        self.memory_threshold = 0.9
        self.initialized = False
        self.initialize_nvml()

    def initialize_nvml(self):
        """Initialize NVIDIA Management Library"""
        try:
            # Initialize NVML
            pynvml.nvmlInit()
            
            # Get device count
            device_count = pynvml.nvmlDeviceGetCount()
            if self.device_id >= device_count:
                raise ValueError(f"Invalid GPU device ID. Available devices: {device_count}")
            
            # Get device handle
            self.device_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.device_handle)
            self.total_memory = mem_info.total
            
            self.initialized = True
            logging.info(f"Successfully initialized GPU {self.device_id}")
            
        except pynvml.NVMLError as e:
            logging.error(f"NVML initialization failed: {e}")
            self.initialized = False
            raise
        except Exception as e:
            logging.error(f"Initialization error: {e}")
            self.initialized = False
            raise

    def get_memory_info(self) -> Dict[str, int]:
        """Get current memory usage"""
        if not self.initialized:
            raise RuntimeError("Memory tracker not initialized")
            
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.device_handle)
            return {
                "total": mem_info.total,
                "used": mem_info.used,
                "free": mem_info.free
            }
        except pynvml.NVMLError as e:
            logging.error(f"Failed to get memory info: {e}")
            raise

    def check_memory_availability(self, required_memory: int) -> bool:
        """Check if required memory is available"""
        if not self.initialized:
            raise RuntimeError("Memory tracker not initialized")
            
        try:
            mem_info = self.get_memory_info()
            return mem_info["free"] >= required_memory
        except Exception as e:
            logging.error(f"Failed to check memory availability: {e}")
            return False

    def __del__(self):
        """Cleanup"""
        if self.initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass