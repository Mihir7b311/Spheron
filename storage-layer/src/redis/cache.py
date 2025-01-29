from typing import Optional, Dict, Any, List
import json
import logging
from .client import RedisClient
from ..common.exceptions import CacheError

class ModelCache:
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
        self.prefix = "model:"
        self.default_ttl = 3600  # 1 hour

    async def cache_model(self, model_id: str, model_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        try:
            key = f"{self.prefix}{model_id}"
            return await self.redis.set(
                key, 
                json.dumps(model_data),
                expire=ttl or self.default_ttl
            )
        except Exception as e:
            logging.error(f"Failed to cache model {model_id}: {e}")
            raise CacheError(f"Cache operation failed: {e}")

    async def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        try:
            key = f"{self.prefix}{model_id}"
            data = await self.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logging.error(f"Failed to get model {model_id}: {e}")
            raise CacheError(f"Cache retrieval failed: {e}")

    async def invalidate_model(self, model_id: str) -> bool:
        try:
            key = f"{self.prefix}{model_id}"
            return await self.redis.delete(key)
        except Exception as e:
            logging.error(f"Failed to invalidate model {model_id}: {e}")
            raise CacheError(f"Cache invalidation failed: {e}")
        


    async def get_stats(self) -> Dict[str, int]:
        try:
            return {
                "total_models": len(await self.list_models()),
                "cache_hits": int(await self.redis.get("cache:hits") or 0),
                "cache_misses": int(await self.redis.get("cache:misses") or 0)
            }
        except Exception as e:
            raise CacheError(f"Failed to get cache stats: {e}")

    async def list_models(self) -> List[str]:
        try:
            keys = self.redis.client.keys(f"{self.prefix}*")
            return [k.replace(self.prefix, "") for k in keys]
        except Exception as e:
            raise CacheError(f"Failed to list models: {e}")