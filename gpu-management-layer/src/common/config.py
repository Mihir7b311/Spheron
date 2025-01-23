# src/common/config.py

import os
import yaml
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from .exceptions import ConfigurationError

@dataclass
class CacheConfig:
    """Cache system configuration"""
    capacity_gb: float
    min_partition_size: int
    eviction_policy: str
    cache_threshold: float
    enable_prefetch: bool

@dataclass
class ExecutionConfig:
    """Execution engine configuration"""
    max_batch_size: int
    min_batch_size: int
    batch_timeout: float
    max_queue_size: int
    enable_dynamic_batching: bool
    cuda_memory_fraction: float

@dataclass
class SharingConfig:
    """GPU sharing configuration"""
    max_vgpus_per_gpu: int
    min_memory_mb: int
    compute_granularity: int
    enable_mps: bool
    oversubscription_ratio: float
    isolation_policy: str

@dataclass
class MonitoringConfig:
    """Monitoring system configuration"""
    polling_interval: float
    metrics_history_size: int
    enable_alerts: bool
    alert_thresholds: Dict[str, float]
    log_level: str

class ConfigurationManager:
    """Unified configuration management"""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize Configuration Manager
        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Any] = {}
        self.component_configs: Dict[str, Any] = {}
        
        # Load all configurations
        self._load_configurations()
        
        # Parse into component configs
        self._parse_configurations()
        
        logging.info("Configuration manager initialized")

    def _load_configurations(self):
        """Load all configuration files"""
        try:
            # Load integrated config first
            integrated_path = self.config_dir / "integrated_config.yaml"
            if integrated_path.exists():
                with open(integrated_path) as f:
                    self.configs["integrated"] = yaml.safe_load(f)
            
            # Load component configs
            config_files = [
                "cache_config.yaml",
                "execution_config.yaml",
                "sharing_config.yaml"
            ]
            
            for config_file in config_files:
                path = self.config_dir / config_file
                if path.exists():
                    with open(path) as f:
                        name = config_file.replace("_config.yaml", "")
                        self.configs[name] = yaml.safe_load(f)
                        
            logging.info(f"Loaded configurations from {self.config_dir}")
            
        except Exception as e:
            logging.error(f"Failed to load configurations: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")

    def _parse_configurations(self):
        """Parse configurations into component configs"""
        try:
            # Parse cache config
            cache_config = self.configs.get("cache", {})
            integrated_cache = self.configs.get("integrated", {}).get("cache", {})
            self.component_configs["cache"] = CacheConfig(
                capacity_gb=integrated_cache.get("capacity_gb", cache_config.get("capacity_gb", 8.0)),
                min_partition_size=integrated_cache.get("min_partition_size", cache_config.get("min_partition_size", 256)),
                eviction_policy=integrated_cache.get("eviction_policy", cache_config.get("eviction_policy", "lru")),
                cache_threshold=integrated_cache.get("cache_threshold", cache_config.get("cache_threshold", 0.2)),
                enable_prefetch=integrated_cache.get("enable_prefetch", cache_config.get("enable_prefetch", False))
            )
            
            # Parse execution config
            exec_config = self.configs.get("execution", {})
            integrated_exec = self.configs.get("integrated", {}).get("execution", {})
            self.component_configs["execution"] = ExecutionConfig(
                max_batch_size=integrated_exec.get("max_batch_size", exec_config.get("max_batch_size", 32)),
                min_batch_size=integrated_exec.get("min_batch_size", exec_config.get("min_batch_size", 1)),
                batch_timeout=integrated_exec.get("batch_timeout", exec_config.get("batch_timeout", 0.1)),
                max_queue_size=integrated_exec.get("max_queue_size", exec_config.get("max_queue_size", 1000)),
                enable_dynamic_batching=integrated_exec.get("enable_dynamic_batching", exec_config.get("enable_dynamic_batching", True)),
                cuda_memory_fraction=integrated_exec.get("cuda_memory_fraction", exec_config.get("cuda_memory_fraction", 0.8))
            )
            
            # Parse sharing config
            sharing_config = self.configs.get("sharing", {})
            integrated_sharing = self.configs.get("integrated", {}).get("sharing", {})
            self.component_configs["sharing"] = SharingConfig(
                max_vgpus_per_gpu=integrated_sharing.get("max_vgpus_per_gpu", sharing_config.get("max_vgpus_per_gpu", 8)),
                min_memory_mb=integrated_sharing.get("min_memory_mb", sharing_config.get("min_memory_mb", 256)),
                compute_granularity=integrated_sharing.get("compute_granularity", sharing_config.get("compute_granularity", 10)),
                enable_mps=integrated_sharing.get("enable_mps", sharing_config.get("enable_mps", True)),
                oversubscription_ratio=integrated_sharing.get("oversubscription_ratio", sharing_config.get("oversubscription_ratio", 1.0)),
                isolation_policy=integrated_sharing.get("isolation_policy", sharing_config.get("isolation_policy", "strict"))
            )
            
            # Parse monitoring config
            monitoring_config = self.configs.get("integrated", {}).get("monitoring", {})
            self.component_configs["monitoring"] = MonitoringConfig(
                polling_interval=monitoring_config.get("polling_interval", 1.0),
                metrics_history_size=monitoring_config.get("metrics_history_size", 10000),
                enable_alerts=monitoring_config.get("enable_alerts", True),
                alert_thresholds=monitoring_config.get("alert_thresholds", {
                    "gpu_utilization": 90.0,
                    "memory_usage": 90.0,
                    "temperature": 80,
                    "power_usage": 250
                }),
                log_level=monitoring_config.get("log_level", "INFO")
            )
            
        except Exception as e:
            logging.error(f"Failed to parse configurations: {e}")
            raise ConfigurationError(f"Configuration parsing failed: {e}")

    def get_component_config(self, component: str) -> Any:
        """Get configuration for specific component"""
        if component not in self.component_configs:
            raise ConfigurationError(f"Unknown component: {component}")
        return self.component_configs[component]

    def get_raw_config(self, config_name: str) -> Dict:
        """Get raw configuration dictionary"""
        return self.configs.get(config_name, {})

    def update_config(self, component: str, updates: Dict):
        """Update component configuration
        Args:
            component: Component to update
            updates: Configuration updates
        """
        try:
            if component not in self.component_configs:
                raise ConfigurationError(f"Unknown component: {component}")
                
            config = self.component_configs[component]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logging.warning(f"Unknown config field: {key} for {component}")
                    
            logging.info(f"Updated configuration for {component}")
            
        except Exception as e:
            logging.error(f"Failed to update configuration: {e}")
            raise ConfigurationError(f"Configuration update failed: {e}")

    def validate_config(self, component: str) -> bool:
        """Validate component configuration
        Args:
            component: Component to validate
        Returns:
            Validation status
        """
        try:
            config = self.component_configs[component]
            
            if component == "cache":
                return self._validate_cache_config(config)
            elif component == "execution":
                return self._validate_execution_config(config)
            elif component == "sharing":
                return self._validate_sharing_config(config)
            elif component == "monitoring":
                return self._validate_monitoring_config(config)
            else:
                raise ConfigurationError(f"Unknown component: {component}")
                
        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            return False

    def _validate_cache_config(self, config: CacheConfig) -> bool:
        """Validate cache configuration"""
        if config.capacity_gb <= 0:
            return False
        if config.min_partition_size <= 0:
            return False
        if config.eviction_policy not in ["lru", "fifo", "lfu"]:
            return False
        if not 0 <= config.cache_threshold <= 1:
            return False
        return True

    def _validate_execution_config(self, config: ExecutionConfig) -> bool:
        """Validate execution configuration"""
        if config.max_batch_size < config.min_batch_size:
            return False
        if config.batch_timeout <= 0:
            return False
        if config.max_queue_size <= 0:
            return False
        if not 0 < config.cuda_memory_fraction <= 1:
            return False
        return True

    def _validate_sharing_config(self, config: SharingConfig) -> bool:
        """Validate sharing configuration"""
        if config.max_vgpus_per_gpu <= 0:
            return False
        if config.min_memory_mb <= 0:
            return False
        if not 0 < config.compute_granularity <= 100:
            return False
        if config.oversubscription_ratio <= 0:
            return False
        return True

    def _validate_monitoring_config(self, config: MonitoringConfig) -> bool:
        """Validate monitoring configuration"""
        if config.polling_interval <= 0:
            return False
        if config.metrics_history_size <= 0:
            return False
        if not isinstance(config.alert_thresholds, dict):
            return False
        return True

    def save_config(self, config_file: str):
        """Save current configuration to file"""
        try:
            path = self.config_dir / config_file
            
            # Convert component configs to dict
            config_dict = {
                component: vars(config)
                for component, config in self.component_configs.items()
            }
            
            with open(path, 'w') as f:
                yaml.safe_dump(config_dict, f)
                
            logging.info(f"Saved configuration to {path}")
            
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
            raise ConfigurationError(f"Configuration save failed: {e}")

    def get_all_configs(self) -> Dict[str, Any]:
        """Get all component configurations"""
        return self.component_configs.copy()