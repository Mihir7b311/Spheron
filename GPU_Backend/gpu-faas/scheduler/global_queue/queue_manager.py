# scheduler/global_queue/queue_manager.py
from typing import Dict, Any, List
from collections import deque

class GlobalQueueManager:
    def __init__(self, config: Dict[str, Any]):
        self.max_size = config["max_size"]
        self.priority_levels = config["priority_levels"]
        self.queue = deque()

    async def add_request(self, request: Dict):
        if len(self.queue) < self.max_size:
            self.queue.append(request)
            return True
        return False

    async def get_next_request(self):
        if self.queue:
            return self.queue.popleft()
        return None
