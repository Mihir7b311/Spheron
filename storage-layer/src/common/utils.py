# common/utils.py

import os
import json
import hashlib
import asyncio
import logging
from typing import Dict, Any, Optional, Union, BinaryIO
from pathlib import Path
from datetime import datetime
import aiofiles
from .exceptions import StorageError
import numpy as np

class FileUtils:
   @staticmethod
   async def read_file(path: Union[str, Path]) -> bytes:
       try:
           async with aiofiles.open(path, 'rb') as f:
               return await f.read()
       except Exception as e:
           raise StorageError(f"Failed to read file {path}: {e}")

   @staticmethod 
   async def write_file(path: Union[str, Path], data: bytes):
       try:
           async with aiofiles.open(path, 'wb') as f:
               await f.write(data)
       except Exception as e:
           raise StorageError(f"Failed to write file {path}: {e}")

   @staticmethod
   async def delete_file(path: Union[str, Path]) -> bool:
       try:
           path = Path(path)
           if path.exists():
               path.unlink()
           return True
       except Exception as e:
           raise StorageError(f"Failed to delete file {path}: {e}")

   @staticmethod
   def get_file_size(path: Union[str, Path]) -> int:
       return Path(path).stat().st_size

   @staticmethod
   async def calculate_checksum(data: bytes) -> str:
       return hashlib.sha256(data).hexdigest()

class JsonUtils:
   @staticmethod
   async def read_json(path: Union[str, Path]) -> Dict:
       try:
           async with aiofiles.open(path, 'r') as f:
               content = await f.read()
               return json.loads(content)
       except Exception as e:
           raise StorageError(f"Failed to read JSON {path}: {e}")

   @staticmethod
   async def write_json(path: Union[str, Path], data: Dict):
       try:
           async with aiofiles.open(path, 'w') as f:
               await f.write(json.dumps(data, indent=2))
       except Exception as e:
           raise StorageError(f"Failed to write JSON {path}: {e}")

class LockManager:
   def __init__(self):
       self._locks: Dict[str, asyncio.Lock] = {}

   def get_lock(self, key: str) -> asyncio.Lock:
       if key not in self._locks:
           self._locks[key] = asyncio.Lock()
       return self._locks[key]

   async def acquire(self, key: str, timeout: Optional[float] = None):
       lock = self.get_lock(key)
       try:
           await asyncio.wait_for(lock.acquire(), timeout)
       except asyncio.TimeoutError:
           raise StorageError(f"Lock acquisition timeout for {key}")

   def release(self, key: str):
       if key in self._locks:
           self._locks[key].release()

class TimeUtils:
   @staticmethod
   def get_timestamp() -> float:
       return datetime.utcnow().timestamp()

   @staticmethod
   def format_duration(seconds: float) -> str:
       return str(datetime.utcfromtimestamp(seconds).strftime('%H:%M:%S'))

class PathUtils:
   @staticmethod
   def ensure_dir(path: Union[str, Path]):
       Path(path).mkdir(parents=True, exist_ok=True)

   @staticmethod
   def is_path_safe(path: str) -> bool:
       return '..' not in Path(path).parts

class ValidationUtils:
   @staticmethod
   def validate_size(size: int, max_size: int) -> bool:
       return 0 < size <= max_size

   @staticmethod 
   def validate_checksum(data: bytes, checksum: str) -> bool:
       return hashlib.sha256(data).hexdigest() == checksum

class EnvUtils:
   @staticmethod
   def get_env(key: str, default: Any = None) -> Any:
       return os.environ.get(key, default)

   @staticmethod
   def get_bool_env(key: str, default: bool = False) -> bool:
       val = os.environ.get(key, str(default)).lower()
       return val in ('true', '1', 'yes')
   

# Add to utils.py

class AsyncUtils:
    @staticmethod
    async def gather_with_concurrency(n: int, *tasks):
        semaphore = asyncio.Semaphore(n)
        async def sem_task(task):
            async with semaphore:
                return await task
        return await asyncio.gather(*(sem_task(task) for task in tasks))

class MetricsUtils:
    @staticmethod
    def calculate_rate(count: int, duration: float) -> float:
        return count / duration if duration > 0 else 0.0

    @staticmethod
    def calculate_percentile(values: list, percentile: float) -> float:
        return float(np.percentile(values, percentile))

class CompressionUtils:
    @staticmethod
    async def compress_data(data: bytes) -> bytes:
        import zlib
        return zlib.compress(data)

    @staticmethod
    async def decompress_data(data: bytes) -> bytes:
        import zlib
        return zlib.decompress(data)

class MemoryUtils:
    @staticmethod
    def get_size(obj: Any) -> int:
        import sys
        return sys.getsizeof(obj)

    @staticmethod
    def format_size(size_bytes: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f}TB"

class RetryUtils:
    @staticmethod
    async def retry_with_backoff(func, max_retries: int = 3, 
                               base_delay: float = 1.0):
        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)



class CacheUtils:
    @staticmethod
    async def memoize(func):
        cache = {}
        async def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key not in cache:
                cache[key] = await func(*args, **kwargs)
            return cache[key]
        return wrapper

class BatchUtils:
    @staticmethod
    def chunk_list(lst: list, chunk_size: int) -> list:
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    @staticmethod
    async def process_in_batches(items: list, batch_size: int, 
                               process_func) -> list:
        results = []
        for batch in BatchUtils.chunk_list(items, batch_size):
            batch_results = await AsyncUtils.gather_with_concurrency(
                len(batch), 
                *(process_func(item) for item in batch)
            )
            results.extend(batch_results)
        return results

class LogUtils:
    @staticmethod
    def setup_logger(name: str, 
                    log_file: str,
                    level=logging.INFO,
                    format_str: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
        logger = logging.getLogger(name)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(handler)
        logger.setLevel(level)
        return logger

    @staticmethod
    def log_exception(logger: logging.Logger, 
                     exc: Exception, 
                     context: str = None):
        msg = f"Exception occurred{f' in {context}' if context else ''}: {exc}"
        logger.exception(msg)