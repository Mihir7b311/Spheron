# model_store/store.py

import os
import mmap
import asyncio
import hashlib
from typing import Dict, Optional, List, BinaryIO
from pathlib import Path
import logging
from ..common.exceptions import ModelStoreError

class ModelStore:
    def __init__(self, base_path: str = "/data/models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.locks: Dict[str, asyncio.Lock] = {}
        self.open_files: Dict[str, BinaryIO] = {}

    def _get_model_path(self, model_id: str, version: str) -> Path:
        return self.base_path / model_id / version / "model.bin"

    def _get_metadata_path(self, model_id: str, version: str) -> Path:
        return self.base_path / model_id / version / "metadata.json"

    async def store_model(self, 
                         model_id: str, 
                         version: str, 
                         model_data: bytes,
                         metadata: Optional[Dict] = None) -> Dict:
        try:
            model_path = self._get_model_path(model_id, version)
            model_path.parent.mkdir(parents=True, exist_ok=True)

            # Get or create lock for this model
            if model_id not in self.locks:
                self.locks[model_id] = asyncio.Lock()

            async with self.locks[model_id]:
                # Write model data
                model_path.write_bytes(model_data)

                # Calculate checksum
                checksum = hashlib.sha256(model_data).hexdigest()

                # Store metadata if provided
                if metadata:
                    metadata_path = self._get_metadata_path(model_id, version)
                    metadata.update({
                        'checksum': checksum,
                        'size': len(model_data),
                        'version': version
                    })
                    import json
                    metadata_path.write_text(json.dumps(metadata))

                return {
                    'model_id': model_id,
                    'version': version,
                    'size': len(model_data),
                    'checksum': checksum,
                    'path': str(model_path)
                }

        except Exception as e:
            self.logger.error(f"Failed to store model {model_id}: {e}")
            raise ModelStoreError(f"Model storage failed: {e}")

    async def load_model(self, 
                        model_id: str, 
                        version: str,
                        verify_checksum: bool = True) -> bytes:
        try:
            model_path = self._get_model_path(model_id, version)
            if not model_path.exists():
                raise ModelStoreError(f"Model {model_id}:{version} not found")

            async with self.locks.get(model_id, asyncio.Lock()):
                data = model_path.read_bytes()

                if verify_checksum:
                    metadata = await self.get_metadata(model_id, version)
                    if metadata and metadata.get('checksum'):
                        checksum = hashlib.sha256(data).hexdigest()
                        if checksum != metadata['checksum']:
                            raise ModelStoreError("Model checksum verification failed")

                return data

        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            raise ModelStoreError(f"Model load failed: {e}")

    async def delete_model(self, model_id: str, version: str) -> bool:
        try:
            model_path = self._get_model_path(model_id, version)
            metadata_path = self._get_metadata_path(model_id, version)

            async with self.locks.get(model_id, asyncio.Lock()):
                if model_path.exists():
                    model_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()

                # Remove parent directories if empty
                version_dir = model_path.parent
                model_dir = version_dir.parent
                
                if not any(version_dir.iterdir()):
                    version_dir.rmdir()
                if not any(model_dir.iterdir()):
                    model_dir.rmdir()

                return True

        except Exception as e:
            self.logger.error(f"Failed to delete model {model_id}: {e}")
            raise ModelStoreError(f"Model deletion failed: {e}")

    async def get_metadata(self, model_id: str, version: str) -> Optional[Dict]:
        try:
            metadata_path = self._get_metadata_path(model_id, version)
            if metadata_path.exists():
                import json
                return json.loads(metadata_path.read_text())
            return None
        except Exception as e:
            self.logger.error(f"Failed to get metadata for {model_id}: {e}")
            raise ModelStoreError(f"Metadata retrieval failed: {e}")

    async def list_models(self) -> List[Dict[str, str]]:
        try:
            models = []
            for model_dir in self.base_path.iterdir():
                if model_dir.is_dir():
                    for version_dir in model_dir.iterdir():
                        if version_dir.is_dir():
                            models.append({
                                'model_id': model_dir.name,
                                'version': version_dir.name
                            })
            return models
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            raise ModelStoreError(f"Model listing failed: {e}")

    async def get_model_versions(self, model_id: str) -> List[str]:
        try:
            model_dir = self.base_path / model_id
            if not model_dir.exists():
                return []
            return [d.name for d in model_dir.iterdir() if d.is_dir()]
        except Exception as e:
            self.logger.error(f"Failed to get versions for {model_id}: {e}")
            raise ModelStoreError(f"Version listing failed: {e}")
        

    async def verify_model(self, model_id: str, version: str) -> bool:
        """Verify model integrity"""
        try:
            data = await self.load_model(model_id, version, verify_checksum=False)
            metadata = await self.get_metadata(model_id, version)
            if not metadata or 'checksum' not in metadata:
                return False
            return hashlib.sha256(data).hexdigest() == metadata['checksum']
        except Exception as e:
            self.logger.error(f"Model verification failed: {e}")
            return False

    async def update_metadata(self, model_id: str, version: str, metadata: Dict) -> bool:
        """Update model metadata"""
        try:
            metadata_path = self._get_metadata_path(model_id, version)
            async with self.locks.get(model_id, asyncio.Lock()):
                if not metadata_path.parent.exists():
                    raise ModelStoreError("Model not found")
                current = await self.get_metadata(model_id, version) or {}
                current.update(metadata)
                import json
                metadata_path.write_text(json.dumps(current))
                return True
        except Exception as e:
            self.logger.error(f"Metadata update failed: {e}")
            raise ModelStoreError(f"Metadata update failed: {e}")

    async def cleanup_orphaned(self) -> List[str]:
        """Clean up orphaned model files"""
        cleaned = []
        try:
            for model_dir in self.base_path.iterdir():
                if model_dir.is_dir():
                    for version_dir in model_dir.iterdir():
                        if version_dir.is_dir():
                            model_path = version_dir / "model.bin"
                            metadata_path = version_dir / "metadata.json"
                            if not model_path.exists() or not metadata_path.exists():
                                await self.delete_model(model_dir.name, version_dir.name)
                                cleaned.append(f"{model_dir.name}:{version_dir.name}")
            return cleaned
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            raise ModelStoreError(f"Cleanup failed: {e}")

    async def get_storage_stats(self) -> Dict:
        """Get storage statistics"""
        try:
            total_size = 0
            model_count = 0
            version_count = 0
            for model_dir in self.base_path.iterdir():
                if model_dir.is_dir():
                    model_count += 1
                    for version_dir in model_dir.iterdir():
                        if version_dir.is_dir():
                            version_count += 1
                            model_path = version_dir / "model.bin"
                            if model_path.exists():
                                total_size += model_path.stat().st_size
            return {
                "total_size_bytes": total_size,
                "model_count": model_count,
                "version_count": version_count,
                "average_size": total_size / version_count if version_count > 0 else 0
            }
        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {e}")
            raise ModelStoreError(f"Stats collection failed: {e}")

    

    def __del__(self):
        # Close any open file handles
        for fh in self.open_files.values():
            try:
                fh.close()
            except:
                pass