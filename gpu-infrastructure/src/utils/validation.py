from typing import Dict, Any, Optional
import logging
from ..base.exceptions import ValidationError

class ResourceValidator:
    """Validates GPU resource requests and allocations"""
    
    @staticmethod
    def validate_memory_request(memory_mb: int, 
                              available_memory: int,
                              min_memory: int = 256) -> None:
        """Validate memory request"""
        if memory_mb <= 0:
            raise ValidationError("Memory request must be positive")
            
        if memory_mb < min_memory:
            raise ValidationError(f"Memory request below minimum ({min_memory}MB)")
            
        if memory_mb > available_memory:
            raise ValidationError(
                f"Insufficient memory: requested {memory_mb}MB, "
                f"available {available_memory}MB"
            )

    @staticmethod
    def validate_compute_request(compute_percentage: int,
                               available_compute: int,
                               min_compute: int = 5) -> None:
        """Validate compute request"""
        if compute_percentage <= 0:
            raise ValidationError("Compute percentage must be positive")
            
        if compute_percentage < min_compute:
            raise ValidationError(
                f"Compute percentage below minimum ({min_compute}%)"
            )
            
        if compute_percentage > 100:
            raise ValidationError("Compute percentage cannot exceed 100%")
            
        if compute_percentage > available_compute:
            raise ValidationError(
                f"Insufficient compute: requested {compute_percentage}%, "
                f"available {available_compute}%"
            )

    @staticmethod
    def validate_gpu_config(config: Dict[str, Any]) -> None:
        """Validate GPU configuration"""
        required_fields = [
            "gpu_id",
            "total_memory",
            "min_memory",
            "min_compute"
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required field: {field}")
                
        if config["total_memory"] <= 0:
            raise ValidationError("Total memory must be positive")
            
        if config["min_memory"] <= 0:
            raise ValidationError("Minimum memory must be positive")
            
        if config["min_compute"] <= 0:
            raise ValidationError("Minimum compute must be positive")
            
        if config["min_compute"] > 100:
            raise ValidationError("Minimum compute cannot exceed 100%")

    @staticmethod
    def validate_process_priority(priority: int,
                                max_priority: int = 100) -> None:
        """Validate process priority"""
        if priority < 0:
            raise ValidationError("Priority cannot be negative")
            
        if priority > max_priority:
            raise ValidationError(f"Priority cannot exceed {max_priority}")

    @staticmethod
    def validate_allocation_request(request: Dict[str, Any]) -> None:
        """Validate resource allocation request"""
        required_fields = [
            "memory_mb",
            "compute_percentage",
            "priority",
            "owner_id"
        ]
        
        for field in required_fields:
            if field not in request:
                raise ValidationError(f"Missing required field: {field}")
                
        ResourceValidator.validate_memory_request(
            request["memory_mb"],
            float('inf')  # Available memory checked later
        )
        
        ResourceValidator.validate_compute_request(
            request["compute_percentage"],
            100  # Available compute checked later
        )
        
        ResourceValidator.validate_process_priority(
            request["priority"]
        )
        
        if not request["owner_id"]:
            raise ValidationError("Owner ID cannot be empty")

class ConfigValidator:
    """Validates configuration settings"""
    
    @staticmethod
    def validate_monitoring_config(config: Dict[str, Any]) -> None:
        """Validate monitoring configuration"""
        required_fields = [
            "polling_interval",
            "metrics_history_size",
            "alert_thresholds"
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required field: {field}")
                
        if config["polling_interval"] <= 0:
            raise ValidationError("Polling interval must be positive")
            
        if config["metrics_history_size"] <= 0:
            raise ValidationError("Metrics history size must be positive")
            
        thresholds = config["alert_thresholds"]
        required_thresholds = ["memory", "utilization", "temperature", "power"]
        
        for threshold in required_thresholds:
            if threshold not in thresholds:
                raise ValidationError(f"Missing threshold: {threshold}")
                
            if not 0 <= thresholds[threshold] <= 1:
                raise ValidationError(
                    f"Invalid threshold value for {threshold}"
                )

    @staticmethod
    def validate_mps_config(config: Dict[str, Any]) -> None:
        """Validate MPS configuration"""
        required_fields = [
            "max_processes",
            "min_compute_percentage"
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required field: {field}")
                
        if config["max_processes"] <= 0:
            raise ValidationError("Maximum processes must be positive")
            
        if not 0 < config["min_compute_percentage"] <= 100:
            raise ValidationError(
                "Minimum compute percentage must be between 0 and 100"
            )