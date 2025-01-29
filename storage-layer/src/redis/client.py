import redis
from typing import Optional, Any
import logging
from ..common.exceptions import RedisError

class RedisClient:
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        try:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.client.ping()  # Test connection
            logging.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            raise RedisError(f"Failed to connect to Redis: {e}")

    async def get(self, key: str) -> Optional[str]:
        try:
            return self.client.get(key)
        except redis.RedisError as e:
            logging.error(f"Redis get error: {e}")
            raise RedisError(f"Failed to get key {key}: {e}")

    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        try:
            return self.client.set(key, value, ex=expire)
        except redis.RedisError as e:
            logging.error(f"Redis set error: {e}")
            raise RedisError(f"Failed to set key {key}: {e}")

    async def delete(self, key: str) -> bool:
        try:
            return bool(self.client.delete(key))
        except redis.RedisError as e:
            logging.error(f"Redis delete error: {e}")
            raise RedisError(f"Failed to delete key {key}: {e}")

    async def exists(self, key: str) -> bool:
        try:
            return bool(self.client.exists(key))
        except redis.RedisError as e:
            logging.error(f"Redis exists error: {e}")
            raise RedisError(f"Failed to check key {key}: {e}")


    async def increment(self, key: str) -> int:
        try:
            return self.client.incr(key)
        except redis.RedisError as e:
            raise RedisError(f"Failed to increment key {key}: {e}")

    async def expire(self, key: str, seconds: int) -> bool:
        try:
            return self.client.expire(key, seconds)
        except redis.RedisError as e:
            raise RedisError(f"Failed to set expiry: {e}")

    async def close(self):
        try:
            self.client.close()
        except Exception as e:
            logging.error(f"Error closing Redis connection: {e}")