from enum import Enum
from typing import Dict, Optional
import traceback
import logging

class ErrorLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ARError(Exception):
    def __init__(self, message: str, level: ErrorLevel, details: Optional[Dict] = None):
        self.message = message
        self.level = level
        self.details = details or {}
        super().__init__(self.message)

class ErrorHandler:
    def __init__(self):
        self.errors = []
        logging.basicConfig(
            filename='c:/devdrive/thInk/logs/ar_errors.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def handle_error(self, error: Exception, context: Dict = None) -> Dict:
        """Handle and log error"""
        if isinstance(error, ARError):
            return self._handle_ar_error(error)
        return self._handle_generic_error(error, context)
    
    def _handle_ar_error(self, error: ARError) -> Dict:
        """Handle application-specific errors"""
        error_data = {
            'message': error.message,
            'level': error.level.value,
            'details': error.details
        }
        
        self.errors.append(error_data)
        logging.error(f"AR Error: {error.message}", extra=error_data)
        return error_data
    
    def _handle_generic_error(self, error: Exception, context: Dict = None) -> Dict:
        """Handle unexpected errors"""
        error_data = {
            'message': str(error),
            'type': type(error).__name__,
            'traceback': traceback.format_exc(),
            'context': context or {},
            'level': ErrorLevel.ERROR.value
        }
        
        self.errors.append(error_data)
        logging.error(f"Unexpected error: {str(error)}", extra=error_data)
        return error_data