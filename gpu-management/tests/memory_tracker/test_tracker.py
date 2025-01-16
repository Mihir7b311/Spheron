# tests/memory_tracker/test_tracker.py
import pytest
import pynvml
from src.memory_tracker.tracker import MemoryTracker

def check_gpu_available():
    """Check if GPU is available"""
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return device_count > 0
    except:
        return False

# Skip all tests if no GPU is available
pytestmark = pytest.mark.skipif(
    not check_gpu_available(),
    reason="No GPU available"
)

def test_memory_tracker_initialization():
    try:
        tracker = MemoryTracker(0)
        assert tracker.device_id == 0
        assert tracker.total_memory > 0
        assert tracker.initialized is True
    except pynvml.NVMLError as e:
        pytest.skip(f"NVML Error: {e}")

def test_memory_info():
    try:
        tracker = MemoryTracker(0)
        mem_info = tracker.get_memory_info()
        assert "total" in mem_info
        assert "used" in mem_info
        assert "free" in mem_info
        assert mem_info["total"] > 0
    except pynvml.NVMLError as e:
        pytest.skip(f"NVML Error: {e}")

def test_memory_availability():
    try:
        tracker = MemoryTracker(0)
        # Test with 1MB
        assert tracker.check_memory_availability(1024 * 1024)
    except pynvml.NVMLError as e:
        pytest.skip(f"NVML Error: {e}")

def test_invalid_device_id():
    with pytest.raises(ValueError):
        MemoryTracker(9999)  # Invalid device ID