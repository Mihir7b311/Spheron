class GPUInfrastructureError(Exception):
    """Base exception class for GPU Infrastructure"""
    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.__cause__ = cause

class MPSDaemonError(GPUInfrastructureError):
    """Exception for MPS daemon operations"""
    pass

class GPUSliceError(GPUInfrastructureError):
    """Exception for GPU slice operations"""
    pass

class ResourceError(GPUInfrastructureError):
    """Exception for resource management"""
    pass

class GPUDeviceError(GPUInfrastructureError):
    """Exception for GPU device operations"""
    pass

class ValidationError(GPUInfrastructureError):
    """Exception for validation failures"""
    pass

class ConfigurationError(GPUInfrastructureError):
    """Exception for configuration issues"""
    pass

class MonitoringError(GPUInfrastructureError):
    """Exception for monitoring issues"""
    pass

class StateError(GPUInfrastructureError):
    """Exception for state management issues"""
    pass

class InitializationError(GPUInfrastructureError):
    """Exception for initialization failures"""
    pass