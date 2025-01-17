# src/cache_system/exceptions.py
class CacheException(Exception):
    """Base exception for cache system"""
    pass

class CacheFullException(CacheException):
    """Raised when cache is full"""
    pass

class ModelTooLargeException(CacheException):
    """Raised when model is too large for cache"""
    pass