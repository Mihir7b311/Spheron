# tests/common/test_exceptions.py
import pytest
from src.common.exceptions import (
    GPUError,
    ResourceError,
    MPSError,
    MemoryError,
    IsolationError,
    SchedulerError,
    ContextError,
    MonitoringError
)

class TestExceptions:
    def test_gpu_error(self):
        """Test GPU error hierarchy"""
        error = GPUError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_resource_error(self):
        """Test resource error"""
        error = ResourceError("Resource allocation failed")
        assert isinstance(error, GPUError)
        assert str(error) == "Resource allocation failed"

    def test_mps_error(self):
        """Test MPS error"""
        error = MPSError("MPS setup failed")
        assert isinstance(error, GPUError)
        assert str(error) == "MPS setup failed"

    def test_memory_error(self):
        """Test memory error"""
        error = MemoryError("Memory allocation failed")
        assert isinstance(error, GPUError)
        assert str(error) == "Memory allocation failed"

    def test_isolation_error(self):
        """Test isolation error"""
        error = IsolationError("Resource isolation failed")
        assert isinstance(error, GPUError)
        assert str(error) == "Resource isolation failed"

    def test_scheduler_error(self):
        """Test scheduler error"""
        error = SchedulerError("Scheduling failed")
        assert isinstance(error, GPUError)
        assert str(error) == "Scheduling failed"

    def test_context_error(self):
        """Test context error"""
        error = ContextError("Context switch failed")
        assert isinstance(error, GPUError)
        assert str(error) == "Context switch failed"

    def test_monitoring_error(self):
        """Test monitoring error"""
        error = MonitoringError("Monitoring failed")
        assert isinstance(error, GPUError)
        assert str(error) == "Monitoring failed"

    def test_error_chaining(self):
        """Test error chaining"""
        original = Exception("Original error")
        error = GPUError("Wrapped error", original)
        assert error.__cause__ == original

    def test_error_inheritance(self):
        """Test error inheritance hierarchy"""
        assert issubclass(ResourceError, GPUError)
        assert issubclass(MPSError, GPUError)
        assert issubclass(MemoryError, GPUError)
        assert issubclass(IsolationError, GPUError)
        assert issubclass(SchedulerError, GPUError)
        assert issubclass(ContextError, GPUError)
        assert issubclass(MonitoringError, GPUError)