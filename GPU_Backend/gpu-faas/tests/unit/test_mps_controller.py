import pytest
from resource_manager.mps_controller.controller import MPSController

@pytest.mark.asyncio
async def test_mps_setup(test_config, mocker):
    controller = MPSController(test_config["mps"])
    mocker.patch.object(controller, '_run_command', return_value="")
    
    result = await controller.setup_mps("gpu-0")
    assert result is True
    assert "gpu-0" in controller.active_gpus

@pytest.mark.asyncio
async def test_mps_cleanup(test_config, mocker):
    controller = MPSController(test_config["mps"])
    mocker.patch.object(controller, '_run_command', return_value="")
    
    await controller.setup_mps("gpu-0")
    result = await controller.stop_mps("gpu-0")  # Use stop_mps instead of cleanup_mps
    assert result is True
    assert "gpu-0" not in controller.active_gpus
