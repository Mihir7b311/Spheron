# tests/conftest.py
import pytest
import asyncio
import pynvml
from unittest.mock import Mock, patch
import os
import sys
# tests/conftest.py
import pytest
import asyncio
import pynvml
from unittest.mock import Mock, patch
import os
import sys

# Add source directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def mock_nvml():
    """Mock NVML for testing"""
    with patch('pynvml.nvmlInit'), \
         patch('pynvml.nvmlDeviceGetCount', return_value=2), \
         patch('pynvml.nvmlDeviceGetHandleByIndex') as mock_handle:
        
        # Create mock handle
        mock_device = Mock()
        
        # Mock memory info
        memory_info = Mock()
        memory_info.total = 8 * 1024 * 1024 * 1024  # 8GB
        memory_info.used = 2 * 1024 * 1024 * 1024   # 2GB
        memory_info.free = 6 * 1024 * 1024 * 1024   # 6GB
        mock_device.nvmlDeviceGetMemoryInfo.return_value = memory_info
        
        # Mock GPU utilization
        utilization = Mock()
        utilization.gpu = 30  # 30% GPU utilization
        mock_device.nvmlDeviceGetUtilizationRates.return_value = utilization
        
        # Mock temperature
        mock_device.nvmlDeviceGetTemperature.return_value = 65
        
        # Mock power usage
        mock_device.nvmlDeviceGetPowerUsage.return_value = 100
        
        # Mock processes
        mock_device.nvmlDeviceGetComputeRunningProcesses.return_value = []
        
        mock_handle.return_value = mock_device
        yield mock_device

@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def virtual_gpu_manager(mock_nvml):
    """Fixture for VirtualGPUManager"""
    from src.virtual_gpu.manager import VirtualGPUManager
    manager = VirtualGPUManager()
    yield manager
    await manager.cleanup_all()

@pytest.fixture
async def mps_manager():
    """Fixture for MPSManager"""
    from src.virtual_gpu.mps_manager import MPSManager
    manager = MPSManager()
    yield manager
    await manager.cleanup_all()

@pytest.fixture
async def memory_manager(mock_nvml):
    """Fixture for MemoryManager"""
    from src.space_sharing.memory_manager import MemoryManager
    manager = MemoryManager(gpu_id=0, total_memory=8*1024)
    yield manager

@pytest.fixture
async def scheduler():
    """Fixture for TimeScheduler"""
    from src.time_sharing.scheduler import TimeScheduler
    scheduler = TimeScheduler()
    yield scheduler

@pytest.fixture
async def resource_monitor(mock_nvml):
    """Fixture for ResourceMonitor"""
    from src.common.monitoring import ResourceMonitor
    monitor = ResourceMonitor()
    yield monitor
    await monitor.stop_monitoring()