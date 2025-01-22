# src/runtime/runtime_config.py

from typing import Dict, Any
import yaml
import logging

class RuntimeConfig:
    def __init__(self, config_path: str = "config/execution_config.yaml"):
        self.config = self._load_config(config_path)
        self.runtime_settings = self.config.get("runtime", {})
        self.batch_settings = self.config.get("batch", {})
        self.cuda_settings = self.config.get("cuda", {})
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from yaml file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            return {}
            
    def get_runtime_setting(self, key: str, default: Any = None) -> Any:
        """Get runtime setting"""
        return self.runtime_settings.get(key, default)
        
    def get_batch_setting(self, key: str, default: Any = None) -> Any:
        """Get batch processing setting"""
        return self.batch_settings.get(key, default)
        
    def get_cuda_setting(self, key: str, default: Any = None) -> Any:
        """Get CUDA setting"""
        return self.cuda_settings.get(key, default)