# tests/cache_system/test_memory_tracker.py
import torch
import pytest
import pynvml
from src.cache_system.memory_tracker import MemoryTracker

def check_gpu_available():
    """Check if GPU is available"""
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return device_count > 0
    except:
        return False

@pytest.mark.skipif(not check_gpu_available(), reason="No GPU available")
class TestMemoryTracker:
    @pytest.fixture
    def memory_tracker(self):
        """Create memory tracker instance"""
        tracker = MemoryTracker(device_id=0)
        yield tracker
        tracker.__del__()

    def test_initialization(self, memory_tracker):
        """Test memory tracker initialization"""
        assert memory_tracker.device_id == 0
        assert memory_tracker.initialized is True
        assert memory_tracker.total_memory > 0

    def test_memory_info(self, memory_tracker):
        """Test getting memory information"""
        mem_info = memory_tracker.get_memory_info()
        assert "total" in mem_info
        assert "used" in mem_info
        assert "free" in mem_info
        assert mem_info["total"] > 0

    def test_memory_availability(self, memory_tracker):
        """Test checking memory availability"""
        # Test with 1MB
        available = memory_tracker.check_memory_availability(1024 * 1024)
        assert isinstance(available, bool)

    def test_utilization(self, memory_tracker):
        """Test getting GPU utilization"""
        utilization = memory_tracker.get_utilization()
        assert isinstance(utilization, float)
        assert 0 <= utilization <= 100

    def test_invalid_device_id(self):
        """Test initialization with invalid device ID"""
        with pytest.raises(ValueError):
            MemoryTracker(device_id=9999)


        # Additional tests for test_memory_tracker.py

    def test_memory_leak_detection(self, memory_tracker):
        """Test memory leak detection"""
        initial_usage = memory_tracker.get_memory_info()["used"]
        
        # Allocate some memory
        data = torch.zeros(1000000, device='cuda')  # 4MB tensor
        
        # Check for leaks
        current_usage = memory_tracker.get_memory_info()["used"]
        assert current_usage > initial_usage
        
        # Cleanup
        del data
        torch.cuda.empty_cache()
        
        # Verify memory released
        final_usage = memory_tracker.get_memory_info()["used"]
        assert abs(final_usage - initial_usage) < 1024 * 1024  # Within 1MB

    def test_fragmentation_tracking(self, memory_tracker):
        """Test memory fragmentation detection"""
        # Create fragmented memory pattern
        tensors = []
        for _ in range(10):
            tensors.append(torch.zeros(1000000, device='cuda'))
            
        # Delete every other tensor
        for i in range(0, len(tensors), 2):
            del tensors[i]
        
        # Check fragmentation
        frag_info = memory_tracker.get_fragmentation_info()
        assert "fragmentation_ratio" in frag_info
        assert "largest_free_block" in frag_info
        
    def test_memory_alerts(self, memory_tracker):
        """Test memory usage alerts"""
        alerts = []
        
        def alert_callback(usage_percent):
            alerts.append(usage_percent)
            
        memory_tracker.set_alert_callback(alert_callback)
        memory_tracker.set_alert_threshold(0.5)  # 50% usage alert
        
        # Trigger alert by allocating memory
        data = torch.zeros(int(memory_tracker.total_memory * 0.6), device='cuda')
        
        # Verify alert triggered
        assert len(alerts) > 0
        assert alerts[0] > 50.0
