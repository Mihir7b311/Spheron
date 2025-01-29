from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from typing import Optional
import logging
from ..common.exceptions import DatabaseError

class PostgresClient:
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 5432,
                 user: str = 'postgres',
                 password: str = 'postgres',
                 database: str = 'gpu_faas'):
        
        self.url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
        self.engine = create_async_engine(
            self.url,
            echo=False,
            pool_size=20,
            max_overflow=10
        )
        self.session_factory = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        self._current_session: Optional[AsyncSession] = None

    async def get_session(self) -> AsyncSession:
        if not self._current_session:
            self._current_session = self.session_factory()
        return self._current_session

    async def begin_transaction(self) -> AsyncSession:
        session = await self.get_session()
        await session.begin()
        return session

    async def commit(self):
        if self._current_session:
            try:
                await self._current_session.commit()
            except Exception as e:
                await self._current_session.rollback()
                raise DatabaseError(f"Failed to commit transaction: {e}")

    async def rollback(self):
        if self._current_session:
            await self._current_session.rollback()

    async def close(self):
        if self._current_session:
            await self._current_session.close()
            self._current_session = None


    async def create_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
    async def drop_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def healthcheck(self) -> bool:
        try:
            async with self.get_session() as session:
                await session.execute("SELECT 1")
            return True
        except Exception:
            return False
        

    async def __aenter__(self):
        return await self.get_session()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()