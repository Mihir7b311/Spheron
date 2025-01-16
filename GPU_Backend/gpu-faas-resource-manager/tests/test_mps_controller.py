# tests/test_mps_controller.py
import pytest
from mps_controller.controller import MPSController

@pytest.mark.asyncio
async def test_mps_setup(mocker):
    config = {
        "max_processes": 48,
        "compute_percentage": 10,
        "default_memory": "1Gi",
        "max_memory": "16Gi"
    }
    
    controller = MPSController(config)
    
    # Mock command execution
    mock_run = mocker.patch.object(controller, '_run_command')
    mock_run.return_value = ""
    
    # Test MPS setup
    result = await controller.setup_mps("0")
    assert result is True
    
    # Verify commands were called
    mock_run.assert_any_call("nvidia-smi -i 0 -c EXCLUSIVE_PROCESS")
    mock_run.assert_any_call("nvidia-cuda-mps-control -d")