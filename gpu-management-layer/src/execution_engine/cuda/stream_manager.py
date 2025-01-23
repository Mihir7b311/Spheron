# src/execution_engine/cuda/stream_manager.py

import torch
import logging
import asyncio
from typing import Dict, Optional, List
from dataclasses import dataclass
from ..common.exceptions import CUDAError

@dataclass
class StreamInfo:
    """Information about a CUDA stream"""
    stream_id: str
    gpu_id: int
    stream: torch.cuda.Stream
    priority: int
    is_active: bool = False
    last_used: float = 0.0
    total_operations: int = 0
    device: Optional[torch.device] = None  # Make sure device is optional

class CUDAStreamManager:
    def __init__(self, config: Dict = None):
        """Initialize CUDA Stream Manager"""
        self.config = config or {}
        self.streams: Dict[str, StreamInfo] = {}
        self.max_streams_per_gpu = self.config.get('max_streams_per_gpu', 16)
        self.default_priority = self.config.get('default_priority', 0)
        self.sync_timeout = self.config.get('sync_timeout', 1.0)
        self.gpu_stream_count: Dict[int, int] = {}

    async def create_stream(self, gpu_id: int, stream_id: Optional[str] = None, priority: Optional[int] = None) -> StreamInfo:
        """Create a new CUDA stream"""
        try:
            if not torch.cuda.is_available():
                raise CUDAError("CUDA is not available")

            if self.gpu_stream_count.get(gpu_id, 0) >= self.max_streams_per_gpu:
                raise CUDAError(f"Maximum streams ({self.max_streams_per_gpu}) reached for GPU {gpu_id}")

            stream_id = stream_id or f"stream_{len(self.streams)}_{gpu_id}"

            if stream_id in self.streams:
                return self.streams[stream_id]

            priority = priority if priority is not None else self.default_priority

            with torch.cuda.device(gpu_id):
                stream = torch.cuda.Stream(priority=priority)
                device = torch.device(f'cuda:{gpu_id}')  # Ensure device is set here

            # Store the device in StreamInfo
            stream_info = StreamInfo(
                stream_id=stream_id,
                gpu_id=gpu_id,
                stream=stream,
                priority=priority,
                is_active=True,
                last_used=asyncio.get_event_loop().time(),
                device=device  # Include device
            )

            self.streams[stream_id] = stream_info
            self.gpu_stream_count[gpu_id] = self.gpu_stream_count.get(gpu_id, 0) + 1

            logging.info(f"Created CUDA stream {stream_id} on GPU {gpu_id}")
            return stream_info

        except Exception as e:
            logging.error(f"Failed to create CUDA stream: {e}")
            raise CUDAError(f"Stream creation failed: {e}")

    async def get_stream(self, stream_id: str) -> Optional[StreamInfo]:
        """Get existing stream by ID"""
        return self.streams.get(stream_id)

    async def release_stream(self, stream_id: str):
        """Release a CUDA stream"""
        try:
            if stream_id not in self.streams:
                return

            stream_info = self.streams[stream_id]
            
            # Synchronize stream before release
            await self.synchronize_stream(stream_id)
            
            # Update counts and remove stream
            self.gpu_stream_count[stream_info.gpu_id] -= 1
            del self.streams[stream_id]

            logging.info(f"Released CUDA stream {stream_id}")

        except Exception as e:
            logging.error(f"Failed to release stream: {e}")
            raise CUDAError(f"Stream release failed: {e}")

    async def synchronize_stream(self, stream_id: str):
        """Synchronize a CUDA stream"""
        try:
            if stream_id not in self.streams:
                raise CUDAError(f"Stream {stream_id} not found")

            stream_info = self.streams[stream_id]
            stream_info.stream.synchronize()
            
            return True

        except Exception as e:
            logging.error(f"Failed to synchronize stream: {e}")
            raise CUDAError(f"Stream synchronization failed: {e}")

    async def wait_stream(self, stream_id: str, timeout: Optional[float] = None):
        """Wait for stream operations to complete"""
        try:
            timeout = timeout or self.sync_timeout
            start_time = asyncio.get_event_loop().time()

            stream_info = self.streams.get(stream_id)
            if not stream_info:
                raise CUDAError(f"Stream {stream_id} not found")

            while not stream_info.stream.query():
                await asyncio.sleep(0.001)  # Small sleep to prevent CPU spinning
                if asyncio.get_event_loop().time() - start_time > timeout:
                    raise CUDAError(f"Stream wait timeout after {timeout}s")

            stream_info.is_active = False
            stream_info.last_used = asyncio.get_event_loop().time()

        except Exception as e:
            logging.error(f"Failed to wait for stream: {e}")
            raise CUDAError(f"Stream wait failed: {e}")

    def get_active_streams(self, gpu_id: Optional[int] = None) -> List[StreamInfo]:
        """Get list of active streams"""
        if gpu_id is not None:
            return [
                stream for stream in self.streams.values()
                if stream.gpu_id == gpu_id and stream.is_active
            ]
        return [stream for stream in self.streams.values() if stream.is_active]

    def get_stream_stats(self, stream_id: str) -> Dict:
        """Get statistics for a stream"""
        stream_info = self.streams.get(stream_id)
        if not stream_info:
            return {}

        return {
            "stream_id": stream_info.stream_id,
            "gpu_id": stream_info.gpu_id,
            "priority": stream_info.priority,
            "is_active": stream_info.is_active,
            "total_operations": stream_info.total_operations,
            "last_used": stream_info.last_used
        }

    async def cleanup_idle_streams(self, idle_timeout: float = 60.0):
        """Cleanup idle streams"""
        current_time = asyncio.get_event_loop().time()
        streams_to_release = [
            stream_id for stream_id, info in self.streams.items()
            if not info.is_active and (current_time - info.last_used) > idle_timeout
        ]

        for stream_id in streams_to_release:
            await self.release_stream(stream_id)

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        for stream_id in list(self.streams.keys()):
            await self.release_stream(stream_id)