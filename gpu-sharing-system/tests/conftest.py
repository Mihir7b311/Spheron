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

@pytest.Fixture
def mock_nvml(mocker):
    # Mock NVML initialization
    mocker.patch("pynvml.nvmlInit", return_value=None)

    # Mock device count
    mocker.patch("pynvml.nvmlDeviceGetCount", return_value=2)

    # Mock GPU handle
    mocker.patch("pynvml.nvmlDeviceGetHandleByIndex", side_effect=lambda x: f"GPU-{x}")

    # Mock utilization rates
    mocker.patch(
        "pynvml.nvmlDeviceGetUtilizationRates",
        side_effect=lambda handle: pynvml.c_nvmlUtilization_t(gpu=50, memory=40),
    )

    # Mock memory info
    mocker.patch(
        "pynvml.nvmlDeviceGetMemoryInfo",
        side_effect=lambda handle: pynvml.c_nvmlMemory_t(total=8_000_000_000, used=4_000_000_000, free=4_000_000_000),
    )

    # Mock temperature
    mocker.patch(
        "pynvml.nvmlDeviceGetTemperature",
        side_effect=lambda handle, _: 70,
    )

    # Mock power usage
    mocker.patch(
        "pynvml.nvmlDeviceGetPowerUsage",
        side_effect=lambda handle: 150_000,
    )


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


