# local_queue/gpu_queue.py
from typing import Dict, List, Any
import time

# local_queue/gpu_queue.py
class LocalQueueManager:
    def __init__(self, config: Dict[str, Any]):
        self.queues: Dict[str, List] = {}
        self.per_gpu_size = config["per_gpu_size"]
        self.timeout = config["timeout_seconds"]

    async def add_to_gpu_queue(self, gpu_id: str, request: Dict[str, Any]):
        if gpu_id not in self.queues:
            self.queues[gpu_id] = []
            
        if len(self.queues[gpu_id]) >= self.per_gpu_size:
            raise Exception("Queue is full")  # Changed error message to match test
            
        request["queue_time"] = time.time()
        self.queues[gpu_id].append(request)

    async def get_gpu_queue_length(self, gpu_id: str) -> int:
        return len(self.queues.get(gpu_id, []))
