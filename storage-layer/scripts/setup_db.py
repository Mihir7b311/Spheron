import asyncio
import argparse
import yaml
import psycopg2
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateSchema
import subprocess
from datetime import datetime

from src.postgresql.models import Base, User, Role
from src.common.config import Configuration
from src.common.exceptions import DatabaseError

class DatabaseSetup:
    def __init__(self, config_path: str):
        self.config = Configuration(config_path)
        self.db_config = self.config.get_postgres_config()
        self.engine = None
        self.session_factory = None

    async def initialize_engine(self):
        self.engine = create_async_engine(
            f"postgresql+asyncpg://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}",
            pool_size=20,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            echo=True
        )
        self.session_factory = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def create_schemas(self):
        async with self.engine.begin() as conn:
            schemas = ['model_storage', 'metrics', 'gpu_resources', 'user_management']
            for schema in schemas:
                await conn.execute(CreateSchema(schema, if_not_exists=True))

    async def create_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def create_indices(self):
        async with self.engine.begin() as conn:
            # Model indices
            await conn.execute("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_models_created ON model_storage.models (created_at);
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_models_version ON model_storage.models (version);
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_models_status ON model_storage.models (status);
                
                -- Metrics indices
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metrics_timestamp ON metrics.model_metrics (timestamp);
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metrics_type ON metrics.model_metrics (metric_type);
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metrics_model ON metrics.model_metrics (model_id);
                
                -- GPU resource indices
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gpu_alloc ON gpu_resources.allocations (gpu_id, status);
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gpu_usage ON gpu_resources.usage_metrics (timestamp);
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gpu_memory ON gpu_resources.memory_usage (gpu_id, timestamp);
            """)

    async def setup_roles(self):
        async with self.session_factory() as session:
            roles = [
                Role(name='admin', permissions=['read', 'write', 'delete', 'manage']),
                Role(name='user', permissions=['read', 'write']),
                Role(name='readonly', permissions=['read'])
            ]
            session.add_all(roles)
            await session.commit()

    async def create_admin_user(self):
        async with self.session_factory() as session:
            admin_role = await session.get(Role, name='admin')
            admin_user = User(
                username='admin',
                password_hash='change_me',  # Should be properly hashed
                role_id=admin_role.id
            )
            session.add(admin_user)
            await session.commit()

    async def optimize_database(self):
        async with self.engine.begin() as conn:
            await conn.execute("VACUUM ANALYZE")
            await conn.execute("REINDEX DATABASE CONCURRENTLY")

    async def create_backup(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"backups/db_backup_{timestamp}.sql"
        Path("backups").mkdir(exist_ok=True)
        
        subprocess.run([
            'pg_dump',
            f'-h{self.db_config["host"]}',
            f'-p{self.db_config["port"]}',
            f'-U{self.db_config["user"]}',
            '-Fc',
            self.db_config["database"],
            f'-f{backup_path}'
        ], env={'PGPASSWORD': self.db_config["password"]})
        
        return backup_path

    async def check_health(self):
        try:
            async with self.engine.begin() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    async def setup_database(self):
        try:
            # Initialize
            await self.initialize_engine()
            
            # Create backup if database exists
            if await self.check_health():
                backup_path = await self.create_backup()
                print(f"Created backup at {backup_path}")

            # Setup database
            await self.create_schemas()
            await self.create_tables()
            await self.create_indices()
            await self.setup_roles()
            await self.create_admin_user()
            await self.optimize_database()

            print("Database setup completed successfully")
            
        except Exception as e:
            raise DatabaseError(f"Database setup failed: {e}")
        finally:
            if self.engine:
                await self.engine.dispose()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/postgres_config.yaml")
    args = parser.parse_args()
    
    setup = DatabaseSetup(args.config)
    await setup.setup_database()

if __name__ == "__main__":
    asyncio.run(main())