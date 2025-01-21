# src/common/exceptions.py

class GPUError(Exception):
    """Base class for GPU-related exceptions"""
    def __init__(self, message, cause=None):
        super().__init__(message)
        self.__cause__ = cause  # S

        
class ResourceError(GPUError):
    """Exception raised for resource allocation issues"""
    pass

class MPSError(GPUError):
    """Exception raised for MPS-related issues"""
    pass

class MemoryError(GPUError):
    """Exception raised for memory management issues"""
    pass

class IsolationError(GPUError):
    """Exception raised for resource isolation issues"""
    pass

class SchedulerError(GPUError):
    """Exception raised for scheduling issues"""
    pass

class ContextError(GPUError):
    """Exception raised for context management issues"""
    pass

class MonitoringError(GPUError):
    """Exception raised for monitoring issues"""
    pass