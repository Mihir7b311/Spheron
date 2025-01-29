# postgresql/repositories/metrics_repo.py

from typing import List, Optional, Dict, Any
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from ..models import Metric, Model
from datetime import datetime, timedelta
import logging
from ...common.exceptions import RepositoryError

class MetricsRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.logger = logging.getLogger(__name__)

    async def create_metric(self, metric_data: Dict[str, Any]) -> Metric:
        try:
            metric = Metric(
                model_id=metric_data['model_id'],
                metric_type=metric_data['metric_type'],
                value=metric_data['value'],
                timestamp=metric_data.get('timestamp', datetime.utcnow()),
                metadata=metric_data.get('metadata', {})
            )
            self.session.add(metric)
            await self.session.flush()
            await self.session.refresh(metric)
            return metric
        except Exception as e:
            self.logger.error(f"Failed to create metric: {e}")
            await self.session.rollback()
            raise RepositoryError(f"Metric creation failed: {e}")

    async def bulk_create_metrics(self, metrics: List[Dict[str, Any]]) -> List[Metric]:
        try:
            metric_objects = [
                Metric(
                    model_id=m['model_id'],
                    metric_type=m['metric_type'],
                    value=m['value'],
                    timestamp=m.get('timestamp', datetime.utcnow()),
                    metadata=m.get('metadata', {})
                ) for m in metrics
            ]
            self.session.add_all(metric_objects)
            await self.session.flush()
            for metric in metric_objects:
                await self.session.refresh(metric)
            return metric_objects
        except Exception as e:
            self.logger.error(f"Failed bulk metric creation: {e}")
            await self.session.rollback()
            raise RepositoryError(f"Bulk metrics creation failed: {e}")

    async def get_metrics(self,
                         model_id: int,
                         metric_type: Optional[str] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         limit: int = 1000) -> List[Metric]:
        try:
            query = select(Metric).where(Metric.model_id == model_id)
            
            if metric_type:
                query = query.where(Metric.metric_type == metric_type)
            if start_time:
                query = query.where(Metric.timestamp >= start_time)
            if end_time:
                query = query.where(Metric.timestamp <= end_time)
                
            query = query.order_by(Metric.timestamp.desc()).limit(limit)
            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            raise RepositoryError(f"Metrics retrieval failed: {e}")

    async def get_aggregated_metrics(self,
                                   model_id: int,
                                   metric_type: str,
                                   interval: str = '1 hour',
                                   start_time: Optional[datetime] = None,
                                   end_time: Optional[datetime] = None) -> List[Dict]:
        try:
            query = select(
                func.date_trunc(interval, Metric.timestamp).label('time_bucket'),
                func.avg(Metric.value).label('avg_value'),
                func.min(Metric.value).label('min_value'),
                func.max(Metric.value).label('max_value'),
                func.count(Metric.id).label('count')
            ).where(
                and_(
                    Metric.model_id == model_id,
                    Metric.metric_type == metric_type
                )
            )

            if start_time:
                query = query.where(Metric.timestamp >= start_time)
            if end_time:
                query = query.where(Metric.timestamp <= end_time)

            query = query.group_by('time_bucket').order_by('time_bucket')
            result = await self.session.execute(query)
            
            return [
                {
                    'timestamp': row.time_bucket,
                    'avg_value': float(row.avg_value),
                    'min_value': float(row.min_value),
                    'max_value': float(row.max_value),
                    'count': row.count
                }
                for row in result
            ]
        except Exception as e:
            self.logger.error(f"Failed to get aggregated metrics: {e}")
            raise RepositoryError(f"Metrics aggregation failed: {e}")

    async def get_latest_metric(self,
                              model_id: int,
                              metric_type: str) -> Optional[Metric]:
        try:
            query = select(Metric).where(
                and_(
                    Metric.model_id == model_id,
                    Metric.metric_type == metric_type
                )
            ).order_by(Metric.timestamp.desc()).limit(1)
            
            result = await self.session.execute(query)
            return result.scalars().first()
        except Exception as e:
            self.logger.error(f"Failed to get latest metric: {e}")
            raise RepositoryError(f"Latest metric retrieval failed: {e}")

    async def delete_old_metrics(self, 
                               retention_days: int = 30) -> int:
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            stmt = Metric.__table__.delete().where(Metric.timestamp < cutoff_date)
            result = await self.session.execute(stmt)
            await self.session.flush()
            return result.rowcount
        except Exception as e:
            self.logger.error(f"Failed to delete old metrics: {e}")
            await self.session.rollback()
            raise RepositoryError(f"Metrics cleanup failed: {e}")

    async def get_metric_summary(self, 
                               model_id: int,
                               metric_type: str,
                               window: str = '24 hours') -> Dict[str, float]:
        try:
            start_time = datetime.utcnow() - timedelta(hours=24)
            query = select(
                func.avg(Metric.value).label('avg'),
                func.min(Metric.value).label('min'),
                func.max(Metric.value).label('max'),
                func.stddev(Metric.value).label('stddev')
            ).where(
                and_(
                    Metric.model_id == model_id,
                    Metric.metric_type == metric_type,
                    Metric.timestamp >= start_time
                )
            )
            
            result = await self.session.execute(query)
            row = result.first()
            
            return {
                'average': float(row.avg) if row.avg else 0.0,
                'minimum': float(row.min) if row.min else 0.0,
                'maximum': float(row.max) if row.max else 0.0,
                'std_dev': float(row.stddev) if row.stddev else 0.0
            }
        except Exception as e:
            self.logger.error(f"Failed to get metric summary: {e}")
            raise RepositoryError(f"Metric summary failed: {e}")



    async def compare_metrics(self,
                            model_ids: List[int],
                            metric_type: str,
                            window: str = '24 hours') -> Dict[int, Dict[str, float]]:
        try:
            start_time = datetime.utcnow() - timedelta(hours=24)
            query = select(
                Metric.model_id,
                func.avg(Metric.value).label('avg'),
                func.min(Metric.value).label('min'),
                func.max(Metric.value).label('max')
            ).where(
                and_(
                    Metric.model_id.in_(model_ids),
                    Metric.metric_type == metric_type,
                    Metric.timestamp >= start_time
                )
            ).group_by(Metric.model_id)
            
            result = await self.session.execute(query)
            return {
                row.model_id: {
                    'average': float(row.avg),
                    'minimum': float(row.min),
                    'maximum': float(row.max)
                }
                for row in result
            }
        except Exception as e:
            self.logger.error(f"Failed to compare metrics: {e}")
            raise RepositoryError(f"Metrics comparison failed: {e}")

    async def get_metric_trends(self,
                              model_id: int,
                              metric_type: str,
                              periods: int = 5) -> Dict[str, float]:
        try:
            # Calculate trend over periods
            query = select(
                func.avg(Metric.value),
                func.regr_slope(
                    Metric.value,
                    func.extract('epoch', Metric.timestamp)
                )
            ).where(
                and_(
                    Metric.model_id == model_id,
                    Metric.metric_type == metric_type
                )
            )
            
            result = await self.session.execute(query)
            avg, slope = result.first()
            
            return {
                'average': float(avg) if avg else 0.0,
                'trend': float(slope) if slope else 0.0
            }
        except Exception as e:
            self.logger.error(f"Failed to get metric trends: {e}")
            raise RepositoryError(f"Trend analysis failed: {e}")
