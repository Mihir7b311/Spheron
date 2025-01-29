from typing import List, Optional, Dict, Any
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from ..models import Model
from datetime import datetime
import logging
from ...common.exceptions import RepositoryError

class ModelRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.logger = logging.getLogger(__name__)

    async def create(self, model_data: Dict[str, Any]) -> Model:
        try:
            model = Model(
                model_id=model_data['model_id'],
                name=model_data['name'],
                version=model_data['version'],
                size_bytes=model_data['size_bytes'],
                framework=model_data['framework'],
                metadata=model_data.get('metadata', {}),
                created_at=datetime.utcnow()
            )
            self.session.add(model)
            await self.session.flush()
            await self.session.refresh(model)
            return model
        except Exception as e:
            self.logger.error(f"Failed to create model: {e}")
            await self.session.rollback()
            raise RepositoryError(f"Model creation failed: {e}")

    async def get_by_id(self, model_id: str) -> Optional[Model]:
        try:
            stmt = select(Model).where(Model.model_id == model_id)
            result = await self.session.execute(stmt)
            return result.scalars().first()
        except Exception as e:
            self.logger.error(f"Failed to get model {model_id}: {e}")
            raise RepositoryError(f"Model retrieval failed: {e}")

    async def get_by_version(self, model_id: str, version: str) -> Optional[Model]:
        try:
            stmt = select(Model).where(
                Model.model_id == model_id,
                Model.version == version
            )
            result = await self.session.execute(stmt)
            return result.scalars().first()
        except Exception as e:
            self.logger.error(f"Failed to get model version: {e}")
            raise RepositoryError(f"Model version retrieval failed: {e}")

    async def list_models(self, 
                         limit: int = 100, 
                         offset: int = 0,
                         filters: Optional[Dict[str, Any]] = None) -> List[Model]:
        try:
            stmt = select(Model)
            
            if filters:
                for key, value in filters.items():
                    if hasattr(Model, key):
                        stmt = stmt.where(getattr(Model, key) == value)
                        
            stmt = stmt.limit(limit).offset(offset)
            result = await self.session.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            raise RepositoryError(f"Model listing failed: {e}")

    async def update(self, 
                    model_id: str, 
                    update_data: Dict[str, Any]) -> Optional[Model]:
        try:
            update_data['updated_at'] = datetime.utcnow()
            stmt = update(Model).where(
                Model.model_id == model_id
            ).values(**update_data)
            
            await self.session.execute(stmt)
            await self.session.flush()
            
            return await self.get_by_id(model_id)
        except Exception as e:
            self.logger.error(f"Failed to update model {model_id}: {e}")
            await self.session.rollback()
            raise RepositoryError(f"Model update failed: {e}")

    async def delete(self, model_id: str) -> bool:
        try:
            stmt = delete(Model).where(Model.model_id == model_id)
            result = await self.session.execute(stmt)
            await self.session.flush()
            return result.rowcount > 0
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_id}: {e}")
            await self.session.rollback()
            raise RepositoryError(f"Model deletion failed: {e}")

    async def get_model_count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        try:
            stmt = select(Model)
            if filters:
                for key, value in filters.items():
                    if hasattr(Model, key):
                        stmt = stmt.where(getattr(Model, key) == value)
            result = await self.session.execute(stmt)
            return len(result.scalars().all())
        except Exception as e:
            self.logger.error(f"Failed to get model count: {e}")
            raise RepositoryError(f"Model count failed: {e}")

    async def get_models_by_framework(self, 
                                    framework: str,
                                    limit: int = 100) -> List[Model]:
        try:
            stmt = select(Model).where(
                Model.framework == framework
            ).limit(limit)
            result = await self.session.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Failed to get models by framework: {e}")
            raise RepositoryError(f"Framework query failed: {e}")

    async def get_model_versions(self, model_id: str) -> List[str]:
        try:
            stmt = select(Model.version).where(Model.model_id == model_id)
            result = await self.session.execute(stmt)
            return [r[0] for r in result.all()]
        except Exception as e:
            self.logger.error(f"Failed to get model versions: {e}")
            raise RepositoryError(f"Version query failed: {e}")

    async def bulk_create(self, models: List[Dict[str, Any]]) -> List[Model]:
        try:
            model_objects = [
                Model(
                    model_id=m['model_id'],
                    name=m['name'],
                    version=m['version'],
                    size_bytes=m['size_bytes'],
                    framework=m['framework'],
                    metadata=m.get('metadata', {}),
                    created_at=datetime.utcnow()
                ) for m in models
            ]
            self.session.add_all(model_objects)
            await self.session.flush()
            for model in model_objects:
                await self.session.refresh(model)
            return model_objects
        except Exception as e:
            self.logger.error(f"Failed bulk model creation: {e}")
            await self.session.rollback()
            raise RepositoryError(f"Bulk creation failed: {e}")