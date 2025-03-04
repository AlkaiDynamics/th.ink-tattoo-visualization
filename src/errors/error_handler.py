"""
Error handling system for the Th.ink AR application.

This module provides standardized error handling, categorization, and logging
for both expected application errors and unexpected exceptions.
"""

from enum import Enum
from typing import Dict, Optional, Any, List, Type, Callable
import traceback
import logging
import json
from functools import wraps
from datetime import datetime

# Configure logger
logger = logging.getLogger("think.errors")


class ErrorLevel(Enum):
    """Error severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories for application errors."""
    
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    RESOURCE = "resource"
    EXTERNAL_SERVICE = "external_service"
    DATABASE = "database"
    CAMERA = "camera"
    MODEL = "model"
    RENDERING = "rendering"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    UNKNOWN = "unknown"


class ARError(Exception):
    """
    Base exception class for application-specific errors.
    
    Attributes:
        message: Human-readable error message
        level: Error severity level
        category: Error category for classification
        code: Error code for client-side handling
        details: Additional error details
    """
    
    def __init__(
        self, 
        message: str, 
        level: ErrorLevel = ErrorLevel.ERROR,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        code: str = "UNKNOWN_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.level = level
        self.category = category
        self.code = code
        self.details = details or {}
        self.timestamp = datetime.utcnow().isoformat()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "message": self.message,
            "level": self.level.value,
            "category": self.category.value,
            "code": self.code,
            "details": self.details,
            "timestamp": self.timestamp
        }
    
    def __str__(self) -> str:
        """String representation of error."""
        return f"{self.code} [{self.level.value}]: {self.message}"


# Common application errors
class AuthenticationError(ARError):
    """Raised when authentication fails."""
    
    def __init__(
        self, 
        message: str = "Authentication failed", 
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message, 
            level=ErrorLevel.ERROR,
            category=ErrorCategory.AUTHENTICATION,
            code="AUTHENTICATION_FAILED",
            details=details
        )


class AuthorizationError(ARError):
    """Raised when a user lacks permission for an operation."""
    
    def __init__(
        self, 
        message: str = "Permission denied", 
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message, 
            level=ErrorLevel.ERROR,
            category=ErrorCategory.AUTHORIZATION,
            code="PERMISSION_DENIED",
            details=details
        )


class ValidationError(ARError):
    """Raised when input validation fails."""
    
    def __init__(
        self, 
        message: str = "Validation failed", 
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message, 
            level=ErrorLevel.WARNING,
            category=ErrorCategory.VALIDATION,
            code="VALIDATION_FAILED",
            details=details
        )


class ResourceNotFoundError(ARError):
    """Raised when a requested resource is not found."""
    
    def __init__(
        self, 
        message: str = "Resource not found", 
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message, 
            level=ErrorLevel.WARNING,
            category=ErrorCategory.RESOURCE,
            code="RESOURCE_NOT_FOUND",
            details=details
        )


class ExternalServiceError(ARError):
    """Raised when an external service call fails."""
    
    def __init__(
        self, 
        message: str = "External service error", 
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message, 
            level=ErrorLevel.ERROR,
            category=ErrorCategory.EXTERNAL_SERVICE,
            code="EXTERNAL_SERVICE_ERROR",
            details=details
        )


class DatabaseError(ARError):
    """Raised when a database operation fails."""
    
    def __init__(
        self, 
        message: str = "Database operation failed", 
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message, 
            level=ErrorLevel.ERROR,
            category=ErrorCategory.DATABASE,
            code="DATABASE_ERROR",
            details=details
        )


class CameraError(ARError):
    """Raised when camera operations fail."""
    
    def __init__(
        self, 
        message: str = "Camera error", 
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message, 
            level=ErrorLevel.ERROR,
            category=ErrorCategory.CAMERA,
            code="CAMERA_ERROR",
            details=details
        )


class ModelError(ARError):
    """Raised when AI model operations fail."""
    
    def __init__(
        self, 
        message: str = "Model error", 
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message, 
            level=ErrorLevel.ERROR,
            category=ErrorCategory.MODEL,
            code="MODEL_ERROR",
            details=details
        )


class RenderingError(ARError):
    """Raised when rendering operations fail."""
    
    def __init__(
        self, 
        message: str = "Rendering error", 
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message, 
            level=ErrorLevel.ERROR,
            category=ErrorCategory.RENDERING,
            code="RENDERING_ERROR",
            details=details
        )


class ErrorHandler:
    """
    Central error handler for the application.
    
    This class provides methods to handle, log, and format errors consistently
    throughout the application.
    """
    
    def __init__(self):
        """Initialize the error handler."""
        self.errors: List[Dict[str, Any]] = []
        self.error_handlers: Dict[Type[Exception], Callable] = {}
        
        # Register default handlers
        self.register_handler(ARError, self._handle_ar_error)
    
    def register_handler(self, 
                         exception_type: Type[Exception], 
                         handler: Callable[[Exception, Dict[str, Any]], Dict[str, Any]]) -> None:
        """
        Register a custom handler for a specific exception type.
        
        Args:
            exception_type: The type of exception to handle
            handler: Function that processes the exception
        """
        self.error_handlers[exception_type] = handler
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle and log an error based on its type.
        
        Args:
            error: The exception to handle
            context: Additional context information
            
        Returns:
            Dictionary with error information
        """
        context = context or {}
        
        # Find the appropriate handler
        for exception_type, handler in self.error_handlers.items():
            if isinstance(error, exception_type):
                return handler(error, context)
        
        # Use generic handler for unregistered exception types
        return self._handle_generic_error(error, context)
    
    def _handle_ar_error(self, error: ARError, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle application-specific errors.
        
        Args:
            error: The ARError exception
            context: Additional context information
            
        Returns:
            Dictionary with error information
        """
        error_data = error.to_dict()
        error_data.update({"context": context})
        
        self.errors.append(error_data)
        self._log_error(error, error_data)
        
        return error_data
    
    def _handle_generic_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle unexpected errors.
        
        Args:
            error: The exception
            context: Additional context information
            
        Returns:
            Dictionary with error information
        """
        error_data = {
            "message": str(error),
            "type": type(error).__name__,
            "traceback": traceback.format_exc(),
            "context": context,
            "level": ErrorLevel.ERROR.value,
            "category": ErrorCategory.UNKNOWN.value,
            "code": "UNHANDLED_ERROR",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.errors.append(error_data)
        self._log_error(error, error_data)
        
        return error_data
    
    def _log_error(self, error: Exception, error_data: Dict[str, Any]) -> None:
        """
        Log error details.
        
        Args:
            error: The exception
            error_data: Dictionary with error details
        """
        level = error_data.get("level", ErrorLevel.ERROR.value)
        log_level = {
            ErrorLevel.INFO.value: logging.INFO,
            ErrorLevel.WARNING.value: logging.WARNING,
            ErrorLevel.ERROR.value: logging.ERROR,
            ErrorLevel.CRITICAL.value: logging.CRITICAL
        }.get(level, logging.ERROR)
        
        logger.log(
            log_level,
            f"{error_data.get('code', 'ERROR')}: {error_data.get('message', str(error))}",
            exc_info=error,
            extra={"error_data": error_data}
        )
    
    def clear_errors(self) -> None:
        """Clear the error history."""
        self.errors = []


# Create a default instance for easy import
default_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """
    Get the default error handler instance.
    
    Returns:
        The default error handler
    """
    return default_error_handler


def handle_errors(error_handler: Optional[ErrorHandler] = None):
    """
    Decorator to automatically handle errors in functions.
    
    Args:
        error_handler: Error handler to use, or None for default
        
    Returns:
        Decorated function
    """
    handler = error_handler or default_error_handler
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler.handle_error(e, {
                    "function": f"{func.__module__}.{func.__qualname__}",
                    "args": args,
                    "kwargs": kwargs
                })
                raise
                
        return wrapper
    
    return decorator