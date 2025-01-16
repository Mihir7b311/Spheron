# # gpu-faas-scheduler/tests/conftest.py
# import os
# import pytest
# import shutil

# @pytest.fixture(scope="session", autouse=True)
# def setup_test_environment():
#     # Create config directory if it doesn't exist
#     config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
#     os.makedirs(config_dir, exist_ok=True)
    
#     # Copy test config to main config location for tests
#     test_config = os.path.join(os.path.dirname(__file__), "test_config.yaml")
#     main_config = os.path.join(config_dir, "scheduler_config.yaml")
    
#     if os.path.exists(test_config):
#         shutil.copy2(test_config, main_config)

#     yield

#     # Cleanup after tests
#     if os.path.exists(main_config):
#         os.remove(main_config)