 
# global_queue/queue_manager.py
from collections import deque
from typing import Dict, Any

class GlobalQueueManager:
    def __init__(self, config: Dict[str, Any]):
        self.queue = deque()
        self.max_size = config["max_size"]
        self.starvation_threshold = config["starvation_threshold"]

    async def add_request(self, request: Dict[str, Any]):
        if len(self.queue) >= self.max_size:
            raise Exception("Queue is full")
        
        request["skip_count"] = 0
        self.queue.append(request)

    async def get_next_request(self):
        if not self.queue:
            return None
        return self.queue.popleft()

    async def peek_requests(self, n: int = 1):
        """Return n requests without removing them"""
        return list(self.queue)[:n]