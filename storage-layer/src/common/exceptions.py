# common/exceptions.py

class StorageError(Exception):
   """Base exception for storage layer"""
   def __init__(self, message: str, cause: Exception = None):
       super().__init__(message)
       self.__cause__ = cause

class RedisError(StorageError):
   """Redis operations errors"""
   pass

class PostgresError(StorageError):
   """PostgreSQL operations errors"""
   pass

class ModelStoreError(StorageError):
   """Model store operations errors"""
   pass

class ConfigurationError(StorageError):
   """Configuration errors"""
   pass

class DatabaseError(StorageError):
   """Database operations errors"""
   pass

class CacheError(StorageError):
   """Cache operations errors"""
   pass

class QueueError(StorageError):
   """Queue operations errors"""
   pass

class RepositoryError(StorageError):
   """Repository operations errors"""
   pass

class MapperError(StorageError):
   """Memory mapping errors"""
   pass

class VersioningError(StorageError):
   """Version management errors"""
   pass

class ValidationError(StorageError):
   """Data validation errors"""
   def __init__(self, message: str, errors: dict = None):
       super().__init__(message)
       self.errors = errors or {}

class ResourceError(StorageError):
   """Resource allocation errors"""
   def __init__(self, message: str, resource_type: str = None):
       super().__init__(message)
       self.resource_type = resource_type

class ConnectionError(StorageError):
   """Connection errors"""
   def __init__(self, message: str, service: str = None):
       super().__init__(message)
       self.service = service

class OperationError(StorageError):
   """Operation errors"""
   def __init__(self, message: str, operation: str = None):
       super().__init__(message) 
       self.operation = operation

class TransactionError(StorageError):
   """Transaction errors"""
   def __init__(self, message: str, transaction_id: str = None):
       super().__init__(message)
       self.transaction_id = transaction_id

class TimeoutError(StorageError):
   """Timeout errors"""
   def __init__(self, message: str, timeout: float = None):
       super().__init__(message)
       self.timeout = timeout

class ConcurrencyError(StorageError):
   """Concurrency errors"""
   def __init__(self, message: str, resource_id: str = None):
       super().__init__(message)
       self.resource_id = resource_id

def handle_exceptions(func):
   """Exception handling decorator"""
   async def wrapper(*args, **kwargs):
       try:
           return await func(*args, **kwargs)
       except StorageError as e:
           raise e
       except Exception as e:
           raise StorageError(f"Unhandled error in {func.__name__}: {str(e)}", e)
   return wrapper



class RetryableError(StorageError):
    """Errors that can be retried"""
    def __init__(self, message: str, retry_after: float = None):
        super().__init__(message)
        self.retry_after = retry_after

class DataIntegrityError(StorageError):
    """Data integrity violations"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}

class StateError(StorageError):
    """Invalid state errors"""
    def __init__(self, message: str, current_state: str = None, expected_state: str = None):
        super().__init__(message)
        self.current_state = current_state
        self.expected_state = expected_state

def create_error_context(error_info: dict) -> dict:
    """Create standardized error context"""
    return {
        'error_type': error_info.get('type'),
        'timestamp': error_info.get('timestamp'),
        'trace_id': error_info.get('trace_id'),
        'details': error_info.get('details', {})
    }

class ErrorHandler:
    """Centralized error handling"""
    
    @staticmethod
    async def handle_error(error: Exception) -> dict:
        """Handle and format error response"""
        if isinstance(error, StorageError):
            return {
                'error': str(error),
                'type': error.__class__.__name__,
                'details': getattr(error, 'details', {}),
                'retryable': isinstance(error, RetryableError)
            }
        return {
            'error': str(error),
            'type': 'UnhandledError'
        }

    @staticmethod
    def is_retryable(error: Exception) -> bool:
        """Check if error is retryable"""
        return isinstance(error, RetryableError) or (
            isinstance(error, StorageError) and 
            not isinstance(error, (ValidationError, DataIntegrityError))
        )