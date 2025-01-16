# memory_tracker/exceptions.py

class MemoryTrackerError(Exception):
    """Base class for Memory Tracker exceptions"""
    pass

class MemoryInitializationError(MemoryTrackerError):
    """Raised when NVML initialization fails"""
    pass

class MemoryAllocationError(MemoryTrackerError):
    """Raised when memory allocation fails"""
    pass

class MemoryMonitoringError(MemoryTrackerError):
    """Raised when memory monitoring fails"""
    pass