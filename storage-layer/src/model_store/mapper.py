# model_store/mapper.py

import mmap
import os
from typing import Dict, Optional, BinaryIO
import logging
from pathlib import Path
import asyncio
from ..common.exceptions import MapperError
import psutil

class MemoryMapper:
    def __init__(self):
        self.mapped_files: Dict[str, mmap.mmap] = {}
        self.file_handles: Dict[str, BinaryIO] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        self.logger = logging.getLogger(__name__)

    async def map_model(self, model_path: str) -> mmap.mmap:
        try:
            if model_path in self.mapped_files:
                return self.mapped_files[model_path]

            if model_path not in self.locks:
                self.locks[model_path] = asyncio.Lock()

            async with self.locks[model_path]:
                file_handle = open(model_path, 'rb')
                self.file_handles[model_path] = file_handle
                mapped = mmap.mmap(
                    file_handle.fileno(),
                    0,
                    access=mmap.ACCESS_READ
                )
                self.mapped_files[model_path] = mapped
                return mapped

        except Exception as e:
            self.logger.error(f"Failed to memory map {model_path}: {e}")
            raise MapperError(f"Memory mapping failed: {e}")

    async def unmap_model(self, model_path: str) -> bool:
        try:
            async with self.locks.get(model_path, asyncio.Lock()):
                if model_path in self.mapped_files:
                    mapped = self.mapped_files[model_path]
                    mapped.close()
                    del self.mapped_files[model_path]

                if model_path in self.file_handles:
                    fh = self.file_handles[model_path]
                    fh.close()
                    del self.file_handles[model_path]

                if model_path in self.locks:
                    del self.locks[model_path]

                return True

        except Exception as e:
            self.logger.error(f"Failed to unmap {model_path}: {e}")
            raise MapperError(f"Memory unmapping failed: {e}")

    async def get_mapped_region(self, 
                              model_path: str, 
                              offset: int = 0, 
                              size: Optional[int] = None) -> memoryview:
        try:
            mapped = await self.map_model(model_path)
            if size is None:
                size = len(mapped) - offset
            return memoryview(mapped)[offset:offset + size]

        except Exception as e:
            self.logger.error(f"Failed to get mapped region: {e}")
            raise MapperError(f"Region access failed: {e}")

    async def get_mapping_stats(self) -> Dict:
        return {
            "mapped_files": len(self.mapped_files),
            "total_mapped_memory": sum(len(m) for m in self.mapped_files.values()),
            "active_locks": len(self.locks)
        }

    def is_mapped(self, model_path: str) -> bool:
        return model_path in self.mapped_files

    async def remap_model(self, model_path: str) -> bool:
        try:
            await self.unmap_model(model_path)
            await self.map_model(model_path)
            return True
        except Exception as e:
            self.logger.error(f"Failed to remap {model_path}: {e}")
            raise MapperError(f"Remapping failed: {e}")

    async def cleanup(self):
        for path in list(self.mapped_files.keys()):
            await self.unmap_model(path)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

class ModelMapper:
    def __init__(self, base_path: str = "/data/models"):
        self.base_path = Path(base_path)
        self.memory_mapper = MemoryMapper()
        self.logger = logging.getLogger(__name__)

    def _get_model_path(self, model_id: str, version: str) -> str:
        return str(self.base_path / model_id / version / "model.bin")

    async def map_model_version(self, model_id: str, version: str) -> mmap.mmap:
        try:
            model_path = self._get_model_path(model_id, version)
            return await self.memory_mapper.map_model(model_path)
        except Exception as e:
            self.logger.error(f"Failed to map model {model_id}:{version}: {e}")
            raise MapperError(f"Model mapping failed: {e}")

    async def unmap_model_version(self, model_id: str, version: str) -> bool:
        try:
            model_path = self._get_model_path(model_id, version)
            return await self.memory_mapper.unmap_model(model_path)
        except Exception as e:
            self.logger.error(f"Failed to unmap model {model_id}:{version}: {e}")
            raise MapperError(f"Model unmapping failed: {e}")

    async def get_model_region(self,
                             model_id: str,
                             version: str,
                             offset: int = 0,
                             size: Optional[int] = None) -> memoryview:
        try:
            model_path = self._get_model_path(model_id, version)
            return await self.memory_mapper.get_mapped_region(model_path, offset, size)
        except Exception as e:
            self.logger.error(f"Failed to get model region: {e}")
            raise MapperError(f"Region access failed: {e}")

    async def get_stats(self) -> Dict:
        return await self.memory_mapper.get_mapping_stats()

    async def cleanup(self):
        await self.memory_mapper.cleanup()

    def is_model_mapped(self, model_id: str, version: str) -> bool:
        model_path = self._get_model_path(model_id, version)
        return self.memory_mapper.is_mapped(model_path)
    

    async def prefetch_model(self, model_id: str, version: str) -> bool:
        """Prefetch model into memory"""
        try:
            model_path = self._get_model_path(model_id, version)
            if not os.path.exists(model_path):
                raise MapperError("Model file not found")
                
            mapped = await self.map_model_version(model_id, version)
            # Force pages into memory
            mapped.readline()  
            return True
        except Exception as e:
            self.logger.error(f"Prefetch failed for {model_id}:{version}: {e}")
            raise MapperError(f"Prefetch failed: {e}")

    async def get_memory_usage(self, model_id: str, version: str) -> Dict:
        """Get memory usage statistics for mapped model"""
        try:
            if not self.is_model_mapped(model_id, version):
                return {"mapped": False, "size": 0}
                
            model_path = self._get_model_path(model_id, version)
            mapped = await self.memory_mapper.get_mapped_region(model_path)
            
            return {
                "mapped": True,
                "size": len(mapped),
                "path": model_path,
                "resident_size": self._get_resident_size(mapped)
            }
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")
            raise MapperError(f"Memory usage check failed: {e}")

    def _get_resident_size(self, mapped: memoryview) -> int:
        """Get resident set size of mapped memory"""
        try:
            process = psutil.Process()
            return process.memory_maps()[0].rss
        except:
            return 0

    async def copy_mapped_region(self, 
                               model_id: str, 
                               version: str,
                               target_path: str) -> bool:
        """Copy mapped region to target path"""
        try:
            region = await self.get_model_region(model_id, version)
            with open(target_path, 'wb') as f:
                f.write(region)
            return True
        except Exception as e:
            self.logger.error(f"Region copy failed: {e}")
            raise MapperError(f"Copy operation failed: {e}")