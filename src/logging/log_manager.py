"""
Comprehensive logging management for the Th.ink AR application.

This module provides a centralized logging system with multiple specialized loggers
for different aspects of the application (system, performance, user activity, etc.).
It supports structured logging, rotation, and different log levels.
"""

import os
import json
import logging
import traceback
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, cast
from datetime import datetime
import functools
import uuid


class JsonFormatter(logging.Formatter):
    """Custom formatter to output logs in JSON format for better parsing."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in getattr(record, "extra_fields", {}).items():
            if key not in log_data:
                log_data[key] = value
        
        return json.dumps(log_data)


class LogManager:
    """
    Centralized logging manager for the Th.ink AR application.
    
    This class provides specialized loggers for different aspects of the application
    and ensures consistent logging format, rotation, and storage.
    """
    
    def __init__(self, app_name: str = "think", log_dir: Optional[str] = None):
        """
        Initialize the log manager.
        
        Args:
            app_name: Name of the application (used as prefix for loggers)
            log_dir: Directory to store log files (defaults to ./logs)
        """
        self.app_name = app_name
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        self.configure_root_logger()
        
        # Setup specialized loggers
        self.system_logger = self.create_logger("system", "system.log")
        self.user_logger = self.create_logger("user", "user_activity.log")
        self.perf_logger = self.create_logger("performance", "performance.log")
        self.security_logger = self.create_logger("security", "security.log")
        self.api_logger = self.create_logger("api", "api.log")
        self.error_logger = self.create_logger("error", "error.log", level=logging.ERROR)
        
        # Trace ID for request tracking
        self.current_trace_id: Optional[str] = None
    
    def configure_root_logger(self) -> None:
        """Configure the root logger with console handler."""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove existing handlers if any
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        root_logger.addHandler(console_handler)
    
    def create_logger(self, 
                     name: str, 
                     filename: str, 
                     level: int = logging.INFO, 
                     use_json: bool = True,
                     max_bytes: int = 10 * 1024 * 1024,  # 10MB
                     backup_count: int = 5) -> logging.Logger:
        """
        Create a specialized logger with file handler.
        
        Args:
            name: Logger name
            filename: Log file name
            level: Logging level
            use_json: Whether to use JSON formatter
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(f"{self.app_name}.{name}")
        logger.setLevel(level)
        
        # Remove existing handlers if any
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create file handler with rotation
        file_path = self.log_dir / filename
        handler = RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        
        # Set formatter
        if use_json:
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        logger.addHandler(handler)
        return logger
    
    def set_trace_id(self, trace_id: Optional[str] = None) -> str:
        """
        Set the current trace ID for request tracking.
        
        Args:
            trace_id: Trace ID to use, or None to generate a new one
            
        Returns:
            The current trace ID
        """
        self.current_trace_id = trace_id or str(uuid.uuid4())
        return self.current_trace_id
    
    def log_system(self, level: str, message: str, **kwargs: Any) -> None:
        """
        Log system-level events.
        
        Args:
            level: Log level (debug, info, warning, error, critical)
            message: Log message
            **kwargs: Additional fields to include in the log
        """
        extra = {"extra_fields": {**kwargs}}
        if self.current_trace_id:
            extra["extra_fields"]["trace_id"] = self.current_trace_id
            
        getattr(self.system_logger, level.lower())(message, extra=extra)
    
    def log_performance(self, metrics: Dict[str, Any]) -> None:
        """
        Log performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        extra = {"extra_fields": metrics}
        if self.current_trace_id:
            extra["extra_fields"]["trace_id"] = self.current_trace_id
            
        self.perf_logger.info("Performance metrics", extra=extra)
    
    def log_user_activity(self, user_id: str, activity: str, **kwargs: Any) -> None:
        """
        Log user activities.
        
        Args:
            user_id: User identifier
            activity: Activity description
            **kwargs: Additional fields to include in the log
        """
        extra = {
            "extra_fields": {
                "user_id": user_id,
                "activity": activity,
                **kwargs
            }
        }
        if self.current_trace_id:
            extra["extra_fields"]["trace_id"] = self.current_trace_id
            
        self.user_logger.info(f"User {user_id}: {activity}", extra=extra)
    
    def log_security(self, event_type: str, severity: str, details: Dict[str, Any]) -> None:
        """
        Log security-related events.
        
        Args:
            event_type: Type of security event (auth_attempt, access_denied, etc.)
            severity: Severity level (low, medium, high, critical)
            details: Details about the event
        """
        extra = {
            "extra_fields": {
                "event_type": event_type,
                "severity": severity,
                **details
            }
        }
        if self.current_trace_id:
            extra["extra_fields"]["trace_id"] = self.current_trace_id
            
        self.security_logger.warning(
            f"Security event: {event_type} ({severity})", 
            extra=extra
        )
    
    def log_api(self, method: str, endpoint: str, status_code: int, 
                response_time: float, **kwargs: Any) -> None:
        """
        Log API requests.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            status_code: HTTP status code
            response_time: Response time in seconds
            **kwargs: Additional fields to include in the log
        """
        extra = {
            "extra_fields": {
                "method": method,
                "endpoint": endpoint,
                "status_code": status_code,
                "response_time": response_time,
                **kwargs
            }
        }
        if self.current_trace_id:
            extra["extra_fields"]["trace_id"] = self.current_trace_id
            
        level = logging.INFO if status_code < 400 else logging.WARNING
        self.api_logger.log(
            level,
            f"{method} {endpoint} {status_code} ({response_time:.3f}s)",
            extra=extra
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """
        Log application errors.
        
        Args:
            error: Exception that occurred
            context: Additional context about the error
        """
        error_context = context or {}
        extra = {
            "extra_fields": {
                "error_type": type(error).__name__,
                "context": error_context
            }
        }
        if self.current_trace_id:
            extra["extra_fields"]["trace_id"] = self.current_trace_id
            
        self.error_logger.error(
            f"Error: {str(error)}", 
            exc_info=error,
            extra=extra
        )


# Create a default instance for easy import
default_log_manager = LogManager()


def get_logger() -> LogManager:
    """
    Get the default log manager instance.
    
    Returns:
        Default log manager instance
    """
    return default_log_manager


# Convenience decorators
def log_function_call(logger: Optional[LogManager] = None) -> callable:
    """
    Decorator to log function calls.
    
    Args:
        logger: Logger to use, or None to use default
        
    Returns:
        Decorated function
    """
    log_mgr = logger or default_log_manager
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            function_name = f"{func.__module__}.{func.__qualname__}"
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                log_mgr.log_system(
                    "debug",
                    f"Function {function_name} completed in {duration:.3f}s",
                    function=function_name,
                    duration=duration
                )
                
                return result
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                log_mgr.log_error(
                    e,
                    {
                        "function": function_name,
                        "duration": duration,
                        "args": str(args),
                        "kwargs": str(kwargs)
                    }
                )
                
                raise
                
        return wrapper
    
    return decorator