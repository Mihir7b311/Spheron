# common/config.py

import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from .exceptions import ConfigurationError

class Configuration:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        self._load_configs()

    def _load_configs(self):
        try:
            config_files = [
                "redis_config.yaml",
                "postgres_config.yaml",
                "store_config.yaml"
            ]

            for config_file in config_files:
                path = self.config_dir / config_file
                if path.exists():
                    with open(path) as f:
                        name = config_file.replace("_config.yaml", "")
                        self.config[name] = yaml.safe_load(f)

        except Exception as e:
            self.logger.error(f"Failed to load configs: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")

    def get_redis_config(self) -> Dict[str, Any]:
        return self.config.get("redis", {})

    def get_postgres_config(self) -> Dict[str, Any]:
        return self.config.get("postgres", {})

    def get_store_config(self) -> Dict[str, Any]:
        return self.config.get("store", {})

    def get_value(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        current = self.config
        
        for k in keys:
            if isinstance(current, dict):
                current = current.get(k)
            else:
                return default
                
        return current if current is not None else default

    def set_value(self, key: str, value: Any):
        keys = key.split(".")
        current = self.config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
            
        current[keys[-1]] = value

    def save_config(self):
        try:
            for name, config in self.config.items():
                path = self.config_dir / f"{name}_config.yaml"
                with open(path, 'w') as f:
                    yaml.safe_dump(config, f)
        except Exception as e:
            raise ConfigurationError(f"Failed to save config: {e}")

    def validate_config(self) -> bool:
        required_fields = {
            "redis": ["host", "port"],
            "postgres": ["host", "port", "database"],
            "store": ["base_path"]
        }

        for section, fields in required_fields.items():
            config = self.config.get(section, {})
            for field in fields:
                if field not in config:
                    return False
        return True

    @staticmethod
    def create_default_config(config_dir: str) -> 'Configuration':
        default_config = {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            },
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "model_store",
                "user": "postgres",
                "password": "postgres"
            },
            "store": {
                "base_path": "/data/models",
                "max_file_size": "10GB",
                "compression": True
            }
        }

        config_path = Path(config_dir)
        config_path.mkdir(parents=True, exist_ok=True)

        for name, config in default_config.items():
            path = config_path / f"{name}_config.yaml"
            with open(path, 'w') as f:
                yaml.safe_dump(config, f)

        return Configuration(config_dir)
    

    async def load_environment_overrides(self):
        """Override config values from environment variables"""
        import os
        
        env_mappings = {
            'REDIS_HOST': 'redis.host',
            'REDIS_PORT': 'redis.port',
            'POSTGRES_HOST': 'postgres.host',
            'POSTGRES_PORT': 'postgres.port',
            'POSTGRES_DB': 'postgres.database',
            'STORE_PATH': 'store.base_path'
        }

        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                self.set_value(config_key, os.environ[env_var])

    def get_section_config(self, section: str) -> Dict[str, Any]:
        """Get complete configuration for a section"""
        if section not in self.config:
            raise ConfigurationError(f"Unknown section: {section}")
        return self.config[section].copy()

    def merge_config(self, other_config: Dict[str, Any]):
        """Merge another configuration into current"""
        for section, values in other_config.items():
            if section not in self.config:
                self.config[section] = {}
            self.config[section].update(values)

    def validate_section(self, section: str) -> bool:
        """Validate specific configuration section"""
        validators = {
            'redis': self._validate_redis,
            'postgres': self._validate_postgres,
            'store': self._validate_store
        }
        return validators.get(section, lambda: False)()

    def _validate_redis(self) -> bool:
        config = self.get_redis_config()
        return all(k in config for k in ['host', 'port', 'db'])

    def _validate_postgres(self) -> bool:
        config = self.get_postgres_config()
        return all(k in config for k in ['host', 'port', 'database', 'user'])

    def _validate_store(self) -> bool:
        config = self.get_store_config()
        return all(k in config for k in ['base_path', 'max_file_size'])
