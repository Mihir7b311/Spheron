from typing import Optional, List, Dict, Any
import json
import logging
from .client import RedisClient
from ..common.exceptions import QueueError

class TaskQueue:
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
        self.queue_prefix = "queue:"
        self.processing_prefix = "processing:"

    async def enqueue(self, queue_name: str, task: Dict[str, Any]) -> bool:
        try:
            queue_key = f"{self.queue_prefix}{queue_name}"
            return bool(self.redis.client.lpush(queue_key, json.dumps(task)))
        except Exception as e:
            logging.error(f"Failed to enqueue task: {e}")
            raise QueueError(f"Enqueue operation failed: {e}")

    async def dequeue(self, queue_name: str) -> Optional[Dict[str, Any]]:
        try:
            queue_key = f"{self.queue_prefix}{queue_name}"
            processing_key = f"{self.processing_prefix}{queue_name}"
            
            # Atomic move from queue to processing
            task_data = self.redis.client.rpoplpush(queue_key, processing_key)
            
            if task_data:
                return json.loads(task_data)
            return None
            
        except Exception as e:
            logging.error(f"Failed to dequeue task: {e}")
            raise QueueError(f"Dequeue operation failed: {e}")

    async def complete_task(self, queue_name: str, task_id: str) -> bool:
        try:
            processing_key = f"{self.processing_prefix}{queue_name}"
            
            # Remove from processing queue
            self.redis.client.lrem(processing_key, 1, task_id)
            return True
            
        except Exception as e:
            logging.error(f"Failed to complete task {task_id}: {e}")
            raise QueueError(f"Task completion failed: {e}")

    async def get_queue_length(self, queue_name: str) -> int:
        try:
            queue_key = f"{self.queue_prefix}{queue_name}"
            return self.redis.client.llen(queue_key)
        except Exception as e:
            logging.error(f"Failed to get queue length: {e}")
            raise QueueError(f"Queue length check failed: {e}")
        


    async def requeue_failed_tasks(self, queue_name: str) -> int:
        try:
            processing_key = f"{self.processing_prefix}{queue_name}"
            queue_key = f"{self.queue_prefix}{queue_name}"
            
            # Move all tasks back to main queue
            count = 0
            while self.redis.client.llen(processing_key) > 0:
                task = self.redis.client.rpoplpush(processing_key, queue_key)
                if task:
                    count += 1
            return count
        except Exception as e:
            raise QueueError(f"Failed to requeue tasks: {e}")

    async def clear_queue(self, queue_name: str) -> bool:
        try:
            queue_key = f"{self.queue_prefix}{queue_name}"
            processing_key = f"{self.processing_prefix}{queue_name}"
            
            self.redis.client.delete(queue_key, processing_key)
            return True
        except Exception as e:
            raise QueueError(f"Failed to clear queue: {e}")
