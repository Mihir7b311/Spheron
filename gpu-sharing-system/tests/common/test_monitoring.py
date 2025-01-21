import pytest
from unittest.mock import patch, Mock
from src.common.monitoring import ResourceMonitor, GPUMetrics
from src.common.exceptions import MonitoringError
import asyncio

@pytest.mark.asyncio
class TestResourceMonitor:
    async def test_initialization(self, mock_nvml):
        """Test monitor initialization"""
        with patch('pynvml.nvmlInit') as mock_init:
            monitor = ResourceMonitor()
            assert monitor.polling_interval == 1.0
            assert len(monitor.metrics_history) == 2
            assert monitor.monitoring_task is None
            mock_init.assert_called_once()

    async def test_metrics_collection(self, mock_nvml):
        """Test collecting GPU metrics"""
        with patch('pynvml.nvmlInit'):
            monitor = ResourceMonitor()
            
            # Mock the NVML functions
            with patch('pynvml.nvmlDeviceGetCount', return_value=2), \
                 patch('pynvml.nvmlDeviceGetHandleByIndex') as mock_handle:
                
                # Setup mock device
                mock_device = Mock()
                mock_device.nvmlDeviceGetMemoryInfo.return_value = Mock(
                    total=8589934592,  # 8GB
                    used=2147483648,   # 2GB
                    free=6442450944    # 6GB
                )
                mock_device.nvmlDeviceGetUtilizationRates.return_value = Mock(
                    gpu=30,
                    memory=20
                )
                mock_device.nvmlDeviceGetTemperature.return_value = 65
                mock_device.nvmlDeviceGetPowerUsage.return_value = 100
                mock_device.nvmlDeviceGetComputeRunningProcesses.return_value = []
                
                mock_handle.return_value = mock_device
                
                metrics = await monitor.collect_metrics()
                
                assert len(metrics) == 2
                for gpu_id, gpu_metrics in metrics.items():
                    assert isinstance(gpu_metrics, GPUMetrics)
                    assert gpu_metrics.utilization == 30
                    assert gpu_metrics.memory_total == 8589934592
                    assert gpu_metrics.temperature == 65

    async def test_alert_generation(self, mock_nvml):
        """Test alert generation"""
        with patch('pynvml.nvmlInit'):
            monitor = ResourceMonitor()
            
            # Mock metrics collection
            async def mock_collect():
                return {
                    0: GPUMetrics(
                        gpu_id=0,
                        timestamp=0.0,
                        utilization=95.0,  # High utilization to trigger alert
                        memory_used=7516192768,  # ~7GB
                        memory_total=8589934592,  # 8GB
                        temperature=85,  # High temperature
                        power_usage=250,
                        compute_processes=1,
                        memory_processes=1
                    )
                }
            
            monitor.collect_metrics = mock_collect
            monitor.alert_thresholds["utilization"] = 90.0
            
            alerts = await monitor.check_alerts()
            assert len(alerts) > 0
            assert any(alert["type"] == "utilization" for alert in alerts)

    async def test_monitoring_lifecycle(self, mock_nvml):
        """Test monitoring start/stop cycle"""
        with patch('pynvml.nvmlInit'):
            monitor = ResourceMonitor()
            
            # Mock metrics collection
            async def mock_collect():
                return {
                    0: GPUMetrics(
                        gpu_id=0,
                        timestamp=0.0,
                        utilization=30.0,
                        memory_used=2147483648,
                        memory_total=8589934592,
                        temperature=65,
                        power_usage=100,
                        compute_processes=1,
                        memory_processes=1
                    )
                }
            
            monitor.collect_metrics = mock_collect
            
            await monitor.start_monitoring()
            assert monitor.monitoring_task is not None
            
            await asyncio.sleep(0.1)  # Short sleep for metrics collection
            
            await monitor.stop_monitoring()
            assert monitor.monitoring_task is None
            
            # Verify metrics were collected
            for gpu_metrics in monitor.metrics_history.values():
                assert len(gpu_metrics) > 0

    async def test_metrics_history(self, mock_nvml):
        """Test metrics history management"""
        with patch('pynvml.nvmlInit'):
            monitor = ResourceMonitor(polling_interval=0.1)
            
            # Mock metrics collection
            async def mock_collect():
                return {
                    0: GPUMetrics(
                        gpu_id=0,
                        timestamp=0.0,
                        utilization=30.0,
                        memory_used=2147483648,
                        memory_total=8589934592,
                        temperature=65,
                        power_usage=100,
                        compute_processes=1,
                        memory_processes=1
                    )
                }
            
            monitor.collect_metrics = mock_collect
            
            await monitor.start_monitoring()
            await asyncio.sleep(0.3)  # Collect several metrics
            await monitor.stop_monitoring()
            
            history = monitor.get_metrics_history(0)
            assert len(history) > 0
            assert all(isinstance(m, GPUMetrics) for m in history)
