# tests/cache_system/utils.py

import torch
import asyncio

async def simulate_memory_pressure(device):
    """Simulate memory pressure for testing"""
    tensors = []
    try:
        while True:
            tensors.append(torch.zeros(1000000, device=device))
            await asyncio.sleep(0.01)
    except Exception:
        # Clean up
        for t in tensors:
            del t
        torch.cuda.empty_cache()

def get_memory_stats(tracker):
    """Get formatted memory statistics"""
    info = tracker.get_memory_info()
    return {
        "used_gb": info["used"] / (1024**3),
        "total_gb": info["total"] / (1024**3),
        "free_gb": info["free"] / (1024**3),
        "utilization": info["used"] / info["total"] * 100
    }