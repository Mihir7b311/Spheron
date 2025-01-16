# tests/performance/test_orchestration.py
import pytest
import asyncio
import time
from fastapi.testclient import TestClient

pytestmark = pytest.mark.performance

async def test_deployment_performance(client):
    num_requests = 100
    max_time = 5.0