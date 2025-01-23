# src/common/exceptions.py

class GPUFaaSError(Exception):
    """Base exception class for GPU FaaS system"""
    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.__cause__ = cause

# Cache System Exceptions
class CacheError(GPUFaaSError):
    """Base exception for cache-related errors"""
    pass

class CacheFullError(CacheError):
    """Raised when cache is full"""
    pass

class ModelNotFoundError(CacheError):
    """Raised when model is not found in cache"""
    pass

# Execution Engine Exceptions
class ExecutionError(GPUFaaSError):
    """Base exception for execution-related errors"""
    pass

class RuntimeError(ExecutionError):
    """Exception for Python runtime issues"""
    pass

class BatchError(ExecutionError):
    """Exception for batch processing issues"""
    pass

class CUDAError(ExecutionError):
    """Exception for CUDA-related issues"""
    pass

# GPU Sharing Exceptions
class ResourceError(GPUFaaSError):
    """Base exception for resource-related errors"""
    pass

class AllocationError(ResourceError):
    """Exception for resource allocation failures"""
    pass

class IsolationError(ResourceError):
    """Exception for resource isolation issues"""
    pass

# Scheduling Exceptions
class SchedulerError(GPUFaaSError):
    """Base exception for scheduler-related errors"""
    pass

class QueueFullError(SchedulerError):
    """Raised when task queue is full"""
    pass

class TaskTimeoutError(SchedulerError):
    """Raised when task execution times out"""
    pass

# System Management Exceptions
class ConfigurationError(GPUFaaSError):
    """Exception for configuration issues"""
    pass

class MonitoringError(GPUFaaSError):
    """Exception for monitoring issues"""
    pass

class CoordinationError(GPUFaaSError):
    """Exception for component coordination issues"""
    pass

class SystemStateError(GPUFaaSError):
    """Exception for invalid system states"""
    pass

# Resource Management Exceptions
class ResourceManagerError(GPUFaaSError):
    """Base exception for resource management issues"""
    pass

class ResourceExhaustedError(ResourceManagerError):
    """Raised when resource limits are exceeded"""
    pass

class ResourceConflictError(ResourceManagerError):
    """Raised when resource conflicts occur"""
    pass