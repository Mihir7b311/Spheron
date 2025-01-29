# model_store/versioning.py

from typing import Dict, List, Optional, Tuple
import semver
import logging
from datetime import datetime
from pathlib import Path
import json
import shutil
from ..common.exceptions import VersioningError

class ModelVersionManager:
    def __init__(self, base_path: str = "/data/models"):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
        self.version_cache: Dict[str, Dict[str, Dict]] = {}

    async def create_version(self, 
                           model_id: str, 
                           version: str,
                           metadata: Optional[Dict] = None) -> Dict:
        try:
            # Validate version string
            if not self._is_valid_version(version):
                raise VersioningError(f"Invalid version format: {version}")

            version_path = self.base_path / model_id / version
            version_path.mkdir(parents=True, exist_ok=True)

            version_info = {
                'version': version,
                'created_at': datetime.utcnow().isoformat(),
                'metadata': metadata or {},
                'status': 'created'
            }

            # Write version info
            self._write_version_info(model_id, version, version_info)
            self._update_cache(model_id, version, version_info)

            return version_info

        except Exception as e:
            self.logger.error(f"Version creation failed: {e}")
            raise VersioningError(f"Failed to create version: {e}")

    async def get_version(self, model_id: str, version: str) -> Optional[Dict]:
        try:
            # Check cache first
            if model_id in self.version_cache and version in self.version_cache[model_id]:
                return self.version_cache[model_id][version]

            version_info = self._read_version_info(model_id, version)
            if version_info:
                self._update_cache(model_id, version, version_info)
            return version_info

        except Exception as e:
            self.logger.error(f"Version retrieval failed: {e}")
            raise VersioningError(f"Failed to get version: {e}")

    async def list_versions(self, 
                          model_id: str, 
                          include_metadata: bool = False) -> List[Dict]:
        try:
            model_path = self.base_path / model_id
            if not model_path.exists():
                return []

            versions = []
            for version_dir in model_path.iterdir():
                if version_dir.is_dir():
                    version = version_dir.name
                    version_info = await self.get_version(model_id, version)
                    if version_info:
                        if not include_metadata:
                            version_info.pop('metadata', None)
                        versions.append(version_info)

            return sorted(versions, 
                        key=lambda x: semver.VersionInfo.parse(x['version']))

        except Exception as e:
            self.logger.error(f"Version listing failed: {e}")
            raise VersioningError(f"Failed to list versions: {e}")

    async def get_latest_version(self, model_id: str) -> Optional[str]:
        try:
            versions = await self.list_versions(model_id)
            if not versions:
                return None
            return versions[-1]['version']

        except Exception as e:
            self.logger.error(f"Latest version check failed: {e}")
            raise VersioningError(f"Failed to get latest version: {e}")

    async def delete_version(self, model_id: str, version: str) -> bool:
        try:
            version_path = self.base_path / model_id / version
            if version_path.exists():
                shutil.rmtree(version_path)
                if model_id in self.version_cache:
                    self.version_cache[model_id].pop(version, None)
                return True
            return False

        except Exception as e:
            self.logger.error(f"Version deletion failed: {e}")
            raise VersioningError(f"Failed to delete version: {e}")

    async def update_version_metadata(self, 
                                   model_id: str, 
                                   version: str,
                                   metadata: Dict) -> Dict:
        try:
            version_info = await self.get_version(model_id, version)
            if not version_info:
                raise VersioningError("Version not found")

            version_info['metadata'].update(metadata)
            version_info['updated_at'] = datetime.utcnow().isoformat()

            self._write_version_info(model_id, version, version_info)
            self._update_cache(model_id, version, version_info)

            return version_info

        except Exception as e:
            self.logger.error(f"Metadata update failed: {e}")
            raise VersioningError(f"Failed to update metadata: {e}")

    async def compare_versions(self, 
                             model_id: str,
                             version1: str,
                             version2: str) -> Dict:
        try:
            v1_info = await self.get_version(model_id, version1)
            v2_info = await self.get_version(model_id, version2)

            if not v1_info or not v2_info:
                raise VersioningError("One or both versions not found")

            v1 = semver.VersionInfo.parse(version1)
            v2 = semver.VersionInfo.parse(version2)

            return {
                'newer': str(max(v1, v2)),
                'older': str(min(v1, v2)),
                'difference': {
                    'major': abs(v1.major - v2.major),
                    'minor': abs(v1.minor - v2.minor),
                    'patch': abs(v1.patch - v2.patch)
                },
                'metadata_diff': self._diff_metadata(
                    v1_info.get('metadata', {}),
                    v2_info.get('metadata', {})
                )
            }

        except Exception as e:
            self.logger.error(f"Version comparison failed: {e}")
            raise VersioningError(f"Failed to compare versions: {e}")

    def _is_valid_version(self, version: str) -> bool:
        try:
            semver.VersionInfo.parse(version)
            return True
        except ValueError:
            return False

    def _get_version_file(self, model_id: str, version: str) -> Path:
        return self.base_path / model_id / version / 'version.json'

    def _write_version_info(self, model_id: str, version: str, info: Dict):
        version_file = self._get_version_file(model_id, version)
        with open(version_file, 'w') as f:
            json.dump(info, f, indent=2)

    def _read_version_info(self, model_id: str, version: str) -> Optional[Dict]:
        version_file = self._get_version_file(model_id, version)
        if version_file.exists():
            with open(version_file) as f:
                return json.load(f)
        return None

    def _update_cache(self, model_id: str, version: str, info: Dict):
        if model_id not in self.version_cache:
            self.version_cache[model_id] = {}
        self.version_cache[model_id][version] = info

    def _diff_metadata(self, metadata1: Dict, metadata2: Dict) -> Dict:
        added = {k: metadata2[k] for k in metadata2.keys() - metadata1.keys()}
        removed = {k: metadata1[k] for k in metadata1.keys() - metadata2.keys()}
        modified = {
            k: (metadata1[k], metadata2[k])
            for k in metadata1.keys() & metadata2.keys()
            if metadata1[k] != metadata2[k]
        }
        
        return {
            'added': added,
            'removed': removed,
            'modified': modified
        }

    async def get_version_history(self, model_id: str) -> List[Dict]:
        try:
            versions = await self.list_versions(model_id, include_metadata=True)
            for version in versions:
                version['created_at'] = datetime.fromisoformat(
                    version['created_at']
                )
                if 'updated_at' in version:
                    version['updated_at'] = datetime.fromisoformat(
                        version['updated_at']
                    )
            return sorted(versions, key=lambda x: x['created_at'])

        except Exception as e:
            self.logger.error(f"Version history retrieval failed: {e}")
            raise VersioningError(f"Failed to get version history: {e}")

    async def clear_cache(self):
        self.version_cache.clear()



    async def tag_version(self, model_id: str, version: str, tag: str) -> bool:
        """Tag a specific version"""
        try:
            if not await self.get_version(model_id, version):
                raise VersioningError("Version not found")
            
            tags_file = self.base_path / model_id / 'tags.json'
            tags = {}
            
            if tags_file.exists():
                with open(tags_file) as f:
                    tags = json.load(f)
                    
            tags[tag] = version
            
            with open(tags_file, 'w') as f:
                json.dump(tags, f, indent=2)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to tag version: {e}")
            raise VersioningError(f"Version tagging failed: {e}")

    async def get_version_by_tag(self, model_id: str, tag: str) -> Optional[str]:
        """Get version associated with tag"""
        try:
            tags_file = self.base_path / model_id / 'tags.json'
            if not tags_file.exists():
                return None
                
            with open(tags_file) as f:
                tags = json.load(f)
                return tags.get(tag)
        except Exception as e:
            self.logger.error(f"Failed to get version by tag: {e}")
            raise VersioningError(f"Tag lookup failed: {e}")

    async def create_release(self, 
                           model_id: str, 
                           version: str,
                           release_notes: str,
                           artifacts: Optional[Dict] = None) -> Dict:
        """Create a release for a version"""
        try:
            version_info = await self.get_version(model_id, version)
            if not version_info:
                raise VersioningError("Version not found")
                
            release_info = {
                'version': version,
                'release_date': datetime.utcnow().isoformat(),
                'release_notes': release_notes,
                'artifacts': artifacts or {},
                'status': 'released'
            }
            
            releases_dir = self.base_path / model_id / 'releases'
            releases_dir.mkdir(exist_ok=True)
            
            release_file = releases_dir / f"{version}.json"
            with open(release_file, 'w') as f:
                json.dump(release_info, f, indent=2)
                
            # Update version status
            version_info['status'] = 'released'
            self._write_version_info(model_id, version, version_info)
            self._update_cache(model_id, version, version_info)
            
            return release_info
        except Exception as e:
            self.logger.error(f"Failed to create release: {e}")
            raise VersioningError(f"Release creation failed: {e}")

    async def get_release_info(self, model_id: str, version: str) -> Optional[Dict]:
        """Get release information for a version"""
        try:
            release_file = self.base_path / model_id / 'releases' / f"{version}.json"
            if not release_file.exists():
                return None
                
            with open(release_file) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to get release info: {e}")
            raise VersioningError(f"Release info retrieval failed: {e}")

    async def rollback_version(self, model_id: str, to_version: str) -> bool:
        """Rollback to a previous version"""
        try:
            if not await self.get_version(model_id, to_version):
                raise VersioningError("Target version not found")
                
            latest = await self.get_latest_version(model_id)
            if not latest:
                raise VersioningError("No version to rollback from")
                
            rollback_info = {
                'from_version': latest,
                'to_version': to_version,
                'timestamp': datetime.utcnow().isoformat(),
                'reason': 'rollback'
            }
            
            # Record rollback
            rollbacks_dir = self.base_path / model_id / 'rollbacks'
            rollbacks_dir.mkdir(exist_ok=True)
            
            rollback_file = rollbacks_dir / f"{datetime.utcnow().isoformat()}.json"
            with open(rollback_file, 'w') as f:
                json.dump(rollback_info, f, indent=2)
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to rollback version: {e}")
            raise VersioningError(f"Version rollback failed: {e}")

    async def get_version_lineage(self, model_id: str) -> List[Dict]:
        """Get version history with relationships"""
        try:
            versions = await self.list_versions(model_id, include_metadata=True)
            releases = []
            rollbacks = []
            
            # Get releases
            releases_dir = self.base_path / model_id / 'releases'
            if releases_dir.exists():
                for release_file in releases_dir.iterdir():
                    with open(release_file) as f:
                        releases.append(json.load(f))
                        
            # Get rollbacks
            rollbacks_dir = self.base_path / model_id / 'rollbacks'
            if rollbacks_dir.exists():
                for rollback_file in rollbacks_dir.iterdir():
                    with open(rollback_file) as f:
                        rollbacks.append(json.load(f))
                        
            return {
                'versions': versions,
                'releases': releases,
                'rollbacks': rollbacks
            }
        except Exception as e:
            self.logger.error(f"Failed to get version lineage: {e}")
            raise VersioningError(f"Lineage retrieval failed: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.version_cache.clear()