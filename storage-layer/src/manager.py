# manager.py

from typing import Dict, Optional, List
import asyncio
import logging
from .redis import RedisClient, ModelCache, TaskQueue
from .postgresql import PostgresClient, ModelRepository, MetricsRepository  
from .model_store import ModelStore, MemoryMapper, ModelVersionManager
from .common.exceptions import StorageError
from .common.monitoring import MetricsCollector, ResourceMonitor

class StorageManager:
   def __init__(self, config: Dict):
       self.logger = logging.getLogger(__name__)
       self.metrics = MetricsCollector()
       self.monitor = ResourceMonitor(self.metrics)
       
       # Initialize components
       self.redis = RedisClient(
           host=config['redis']['host'],
           port=config['redis']['port']
       )
       
       self.postgres = PostgresClient(
           host=config['postgres']['host'],
           port=config['postgres']['port'],
           database=config['postgres']['database']
       )
       
       self.model_store = ModelStore(config['store']['base_path'])
       self.model_cache = ModelCache(self.redis)
       self.task_queue = TaskQueue(self.redis)
       self.memory_mapper = MemoryMapper()
       self.version_manager = ModelVersionManager(config['store']['base_path'])
       
       # Initialize repositories
       self.model_repo = ModelRepository(self.postgres)
       self.metrics_repo = MetricsRepository(self.postgres)

   async def initialize(self):
       """Initialize storage manager"""
       try:
           await self.monitor.start_monitoring()
           await self.postgres.create_tables()
           self.logger.info("Storage manager initialized")
       except Exception as e:
           raise StorageError(f"Initialization failed: {e}")

   async def store_model(self, 
                        model_id: str,
                        version: str, 
                        model_data: bytes,
                        metadata: Optional[Dict] = None) -> Dict:
       """Store model with version"""
       try:
           # Store model binary
           store_info = await self.model_store.store_model(
               model_id, version, model_data, metadata
           )
           
           # Create version
           version_info = await self.version_manager.create_version(
               model_id, version, metadata
           )
           
           # Cache model info
           await self.model_cache.cache_model(model_id, {
               **store_info,
               **version_info
           })
           
           # Record in database
           await self.model_repo.create({
               'model_id': model_id,
               'version': version,
               'size_bytes': len(model_data),
               'metadata': metadata
           })
           
           return store_info
           
       except Exception as e:
           self.logger.error(f"Failed to store model: {e}")
           raise StorageError(f"Model storage failed: {e}")

   async def load_model(self,
                       model_id: str,
                       version: Optional[str] = None) -> bytes:
       """Load model by ID and optional version"""
       try:
           # Try cache first
           cached = await self.model_cache.get_model(model_id)
           if cached:
               return cached
           
           # Get version
           if not version:
               version = await self.version_manager.get_latest_version(model_id)
               
           # Load from store
           model_data = await self.model_store.load_model(model_id, version)
           
           # Update cache
           await self.model_cache.cache_model(model_id, {
               'model_id': model_id,
               'version': version,
               'data': model_data
           })
           
           return model_data
           
       except Exception as e:
           self.logger.error(f"Failed to load model: {e}")
           raise StorageError(f"Model load failed: {e}")

   async def delete_model(self, model_id: str, version: Optional[str] = None) -> bool:
       """Delete model and optional version"""
       try:
           if version:
               # Delete specific version
               await self.model_store.delete_model(model_id, version)
               await self.version_manager.delete_version(model_id, version)
           else:
               # Delete all versions
               versions = await self.version_manager.list_versions(model_id)
               for v in versions:
                   await self.delete_model(model_id, v['version'])
                   
           # Clear cache
           await self.model_cache.invalidate_model(model_id)
           
           # Delete from database 
           await self.model_repo.delete(model_id)
           
           return True
           
       except Exception as e:
           self.logger.error(f"Failed to delete model: {e}")
           raise StorageError(f"Model deletion failed: {e}")

   async def get_model_info(self, model_id: str) -> Dict:
       """Get model information"""
       try:
           model = await self.model_repo.get_by_id(model_id)
           versions = await self.version_manager.list_versions(model_id)
           metrics = await self.metrics_repo.get_metrics(model.id)
           
           return {
               'model': model,
               'versions': versions,
               'metrics': metrics
           }
       except Exception as e:
           self.logger.error(f"Failed to get model info: {e}")
           raise StorageError(f"Info retrieval failed: {e}")
           
   async def get_storage_stats(self) -> Dict:
       """Get storage statistics"""
       return {
           'model_store': await self.model_store.get_storage_stats(),
           'cache': await self.model_cache.get_stats(),
           'metrics': self.metrics.get_stats()
       }

   async def cleanup(self):
       """Cleanup manager resources"""
       try:
           await self.monitor.stop_monitoring()
           await self.memory_mapper.cleanup()
           await self.model_cache.clear()
           self.logger.info("Storage manager cleaned up")
       except Exception as e:
           self.logger.error(f"Cleanup failed: {e}")
           raise StorageError(f"Cleanup failed: {e}")
       

   async def list_models(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """List all models with pagination"""
        return await self.model_repo.list_models(limit, offset)

   async def update_model_metadata(self, 
                                  model_id: str, 
                                  version: str,
                                  metadata: Dict) -> Dict:
        """Update model metadata"""
        await self.version_manager.update_version_metadata(model_id, version, metadata)
        await self.model_repo.update(model_id, {'metadata': metadata})
        await self.model_cache.invalidate_model(model_id)
        return await self.get_model_info(model_id)

   async def compare_versions(self,
                             model_id: str,
                             version1: str,
                             version2: str) -> Dict:
        """Compare model versions"""
        return await self.version_manager.compare_versions(model_id, version1, version2)

   async def rollback_version(self, model_id: str, to_version: str) -> bool:
        """Rollback model to specific version"""
        if await self.rollback_version(model_id, to_version):
            await self.model_cache.invalidate_model(model_id)
            return True
        return False
   
   async def handle_low_resources(self):
        """Handle low resource conditions"""
        stats = await self.get_storage_stats()
        if stats['cache']['memory_utilization'] > 0.9:
            await self.model_cache.clear()
        if stats['model_store']['disk_utilization'] > 0.9:
            await self.cleanup_old_versions()

   async def cleanup_old_versions(self, retention_days: int = 30):
        """Cleanup old model versions"""
        models = await self.list_models()
        for model in models:
            versions = await self.version_manager.list_versions(model['model_id'])
            for version in versions[:-3]:  # Keep last 3 versions
                await self.delete_model(model['model_id'], version['version'])


   async def verify_model_integrity(self, model_id: str, version: str) -> bool:
        """Verify model data integrity"""
        try:
            data = await self.load_model(model_id, version)
            metadata = await self.version_manager.get_version(model_id, version)
            return await self.model_store.verify_model(model_id, version)
        except Exception as e:
            self.logger.error(f"Integrity check failed: {e}")
            return False

   async def optimize_storage(self):
        """Optimize storage usage"""
        try:
            # Defragment model store
            await self.model_store.cleanup_orphaned()
            
            # Optimize cache
            await self.model_cache.optimize()
            
            # Clean up old metrics
            await self.metrics_repo.delete_old_metrics()
            
            self.logger.info("Storage optimization completed")
        except Exception as e:
            self.logger.error(f"Storage optimization failed: {e}")
            raise StorageError(f"Optimization failed: {e}")

   async def get_resource_usage(self) -> Dict:
        """Get detailed resource usage"""
        return {
            'memory': await self.memory_mapper.get_mapping_stats(),
            'cache': await self.model_cache.get_stats(),
            'database': await self.postgres.get_stats(),
            'queue': await self.task_queue.get_stats()
        }

   async def healthcheck(self) -> Dict[str, bool]:
        """Comprehensive health check"""
        return {
            'redis': await self.redis.healthcheck(),
            'postgres': await self.postgres.healthcheck(),
            'model_store': await self.model_store.healthcheck(),
            'cache': await self.model_cache.healthcheck()
        }
 

   async def __aenter__(self):
       await self.initialize()
       return self

   async def __aexit__(self, exc_type, exc_val, exc_tb):
       await self.cleanup()