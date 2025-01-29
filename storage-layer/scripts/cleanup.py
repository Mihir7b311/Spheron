import asyncio
import argparse
import logging
from datetime import datetime, timedelta
import yaml
from pathlib import Path
from typing import Dict, Optional, List
import aiofiles
import aiofiles.os

from src.postgresql.client import PostgresClient
from src.redis.client import RedisClient
from src.model_store.store import ModelStore
from src.common.exceptions import StorageError
from src.common.monitoring import MetricsCollector

class StorageCleanup:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.postgres = PostgresClient(self.config)
        self.redis = RedisClient(self.config)
        self.model_store = ModelStore(self.config)
        self.metrics = MetricsCollector()
        self.locks = {}

    async def acquire_lock(self, resource: str, timeout: int = 30):
        """Acquire cleanup lock"""
        lock_key = f"cleanup_lock:{resource}"
        acquired = await self.redis.set_nx(lock_key, "1", timeout)
        if not acquired:
            raise StorageError(f"Could not acquire lock for {resource}")
        self.locks[resource] = lock_key

    async def release_lock(self, resource: str):
        """Release cleanup lock"""
        if resource in self.locks:
            await self.redis.delete(self.locks[resource])
            del self.locks[resource]

    async def create_backup(self, resource: str):
        """Create backup before cleanup"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_dir = Path("backups") / resource
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        if resource == "postgres":
            backup_path = backup_dir / f"db_backup_{timestamp}.sql"
            # Create database backup
            import subprocess
            subprocess.run([
                'pg_dump',
                f'-h{self.config["postgres"]["host"]}',
                f'-p{self.config["postgres"]["port"]}',
                f'-U{self.config["postgres"]["user"]}',
                '-Fc',
                self.config["postgres"]["database"],
                f'-f{str(backup_path)}'
            ], env={'PGPASSWORD': self.config["postgres"]["password"]})
            
        elif resource == "models":
            backup_path = backup_dir / f"models_{timestamp}.tar.gz"
            # Create model store backup
            import tarfile
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(self.config["store"]["base_path"])
                
        return backup_path

    async def verify_cleanup(self, before_metrics: Dict, after_metrics: Dict):
        """Verify cleanup results"""
        # Check metrics were actually reduced
        if after_metrics['total_size'] >= before_metrics['total_size']:
            raise StorageError("Cleanup did not reduce storage size")
            
        # Check no critical data was lost
        for metric in ['active_models', 'recent_metrics']:
            if after_metrics[metric] < before_metrics[metric] * 0.9:
                raise StorageError(f"Too much {metric} data was removed")

    async def cleanup_models(self, days: int, dry_run: bool = True):
        """Cleanup old model files"""
        try:
            await self.acquire_lock("models")
            
            if not dry_run:
                backup_path = await self.create_backup("models")
                logging.info(f"Created model backup at {backup_path}")



            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Get metrics before cleanup
            before_metrics = await self.get_storage_metrics()
            
            # Find old models
            old_models = await self.model_store.list_models(before_date=cutoff_date)
            old_model_ids = [m['model_id'] for m in old_models]
            
            if not dry_run:
                # Delete model files
                for model_id in old_model_ids:
                    await self.model_store.delete_model(model_id)
                    
                # Clean related data in transaction
                async with self.postgres.transaction() as tx:
                    await tx.execute("""
                        DELETE FROM model_storage.models 
                        WHERE model_id = ANY($1)
                    """, old_model_ids)
                    
                    await tx.execute("""
                        DELETE FROM model_storage.versions
                        WHERE model_id = ANY($1)
                    """, old_model_ids)
                    
                # Verify cleanup
                after_metrics = await self.get_storage_metrics()
                await self.verify_cleanup(before_metrics, after_metrics)

        except Exception as e:
            logging.error(f"Model cleanup failed: {e}")
            raise
        finally:
            await self.release_lock("models")

    async def cleanup_metrics(self, days: int, dry_run: bool = True):
        """Cleanup old metrics data"""
        try:
            await self.acquire_lock("metrics")
            if not dry_run:
                backup_path = await self.create_backup("postgres")
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            async with self.postgres.transaction() as tx:
                # Delete old metrics in batches
                batch_size = 10000
                while True:
                    deleted = await tx.execute("""
                        DELETE FROM metrics.model_metrics 
                        WHERE timestamp < $1
                        LIMIT $2
                        RETURNING metric_id
                    """, cutoff_date, batch_size)
                    
                    if deleted.rowcount < batch_size:
                        break
                        
                    await asyncio.sleep(0.1)  # Prevent overload

                # Delete GPU metrics
                await tx.execute("""
                    DELETE FROM metrics.gpu_metrics
                    WHERE timestamp < $1
                """, cutoff_date)

        finally:
            await self.release_lock("metrics")

    async def cleanup_cache(self, dry_run: bool = True):
        """Cleanup Redis cache"""
        try:
            await self.acquire_lock("cache")
            
            # Find unused cache entries
            all_keys = await self.redis.scan_iter("model:*")
            active_models = await self.postgres.fetch_all(
                "SELECT model_id FROM model_storage.models"
            )
            active_model_ids = {m['model_id'] for m in active_models}
            
            to_delete = []
            for key in all_keys:
                model_id = key.split(":")[-1]
                if model_id not in active_model_ids:
                    to_delete.append(key)
                    
            if not dry_run:
                for key in to_delete:
                    await self.redis.delete(key)
                    
        finally:
            await self.release_lock("cache")

    async def get_storage_metrics(self) -> Dict:
        """Get current storage metrics"""
        return {
            'total_size': await self.model_store.get_total_size(),
            'active_models': await self.postgres.fetch_val(
                "SELECT COUNT(*) FROM model_storage.models"
            ),
            'recent_metrics': await self.postgres.fetch_val("""
                SELECT COUNT(*) FROM metrics.model_metrics
                WHERE timestamp > NOW() - INTERVAL '24 hours'
            """)
        }

    async def schedule_background_cleanup(self, interval_days: int = 7):
        """Schedule periodic cleanup"""
        while True:
            try:
                await self.cleanup_metrics(interval_days, dry_run=False)
                await self.cleanup_cache(dry_run=False)
                await asyncio.sleep(interval_days * 86400)
            except Exception as e:
                logging.error(f"Background cleanup failed: {e}")
                await asyncio.sleep(3600)  # Retry in an hour

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/store_config.yaml")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()

    cleanup = StorageCleanup(args.config)
    
    if args.background:
        await cleanup.schedule_background_cleanup()
    else:
        await cleanup.cleanup_models(args.days, args.dry_run)
        await cleanup.cleanup_metrics(args.days, args.dry_run)
        await cleanup.cleanup_cache(args.dry_run)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())