# tests/conftest.py

import pytest
import torch
import os
import yaml
from typing import Dict
import torch

def pytest_addoption(parser):
    parser.addoption(
        "--gpu", action="store_true", default=False, help="run gpu tests"
    )

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu") and not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="Need GPU to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
@pytest.fixture
def cuda_available():
    return torch.cuda.is_available()

@pytest.fixture
def test_config() -> Dict:
    config_path = os.path.join(
        os.path.dirname(__file__),
        '../config/execution_config.yaml'
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def mock_gpu_context():
    return {
        "gpu_id": 0,
        "memory_limit": "4GB",
        "compute_limit": 100
    }

@pytest.fixture
def sample_model():
    """Create sample PyTorch model for testing"""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )

@pytest.fixture
def sample_batch_input():
    """Create sample batch input for testing"""
    return torch.randn(32, 10)