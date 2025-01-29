from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import Index, UniqueConstraint
from datetime import datetime

Base = declarative_base()

class Model(Base):
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True)
    model_id = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    size_bytes = Column(Integer, nullable=False)
    framework = Column(String, nullable=False)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    metrics = relationship("Metric", back_populates="model")

    __table_args__ = (
        Index('idx_model_id', 'model_id'),
        Index('idx_created_at', 'created_at'),
        UniqueConstraint('model_id', 'version', name='uq_model_version')
    )

class Metric(Base):
    __tablename__ = 'metrics'

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('models.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    metric_type = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    metadata = Column(JSON, nullable=True)

    model = relationship("Model", back_populates="metrics")
    
    __table_args__ = (
        Index('idx_model_metric', 'model_id', 'metric_type'),
        Index('idx_timestamp', 'timestamp')
    )