import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from typing import Dict, Any

class LogManager:
    def __init__(self, app_name: str = "think"):
        self.app_name = app_name
        self.log_dir = "c:/devdrive/thInk/logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup different loggers
        self.setup_system_logger()
        self.setup_performance_logger()
        self.setup_user_logger()
        
    def setup_system_logger(self):
        """Setup system-level logging"""
        self.system_logger = logging.getLogger(f"{self.app_name}.system")
        self._configure_logger(
            self.system_logger,
            "system.log",
            '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
        )
        
    def setup_performance_logger(self):
        """Setup performance metrics logging"""
        self.perf_logger = logging.getLogger(f"{self.app_name}.performance")
        self._configure_logger(
            self.perf_logger,
            "performance.log",
            '%(asctime)s - %(message)s'
        )
        
    def setup_user_logger(self):
        """Setup user activity logging"""
        self.user_logger = logging.getLogger(f"{self.app_name}.user")
        self._configure_logger(
            self.user_logger,
            "user_activity.log",
            '%(asctime)s - %(user_id)s - %(message)s'
        )
        
    def _configure_logger(self, logger: logging.Logger, filename: str, format_str: str):
        """Configure individual logger settings"""
        handler = RotatingFileHandler(
            os.path.join(self.log_dir, filename),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
    def log_system(self, level: str, message: str, **kwargs):
        """Log system-level events"""
        getattr(self.system_logger, level)(message, extra=kwargs)
        
    def log_performance(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        self.perf_logger.info(
            f"FPS: {metrics.get('fps', 0):.2f}, "
            f"Battery: {metrics.get('battery_level', 0)}%, "
            f"Temperature: {metrics.get('temperature', 0)}°C"
        )
        
    def log_user_activity(self, user_id: str, activity: str, **kwargs):
        """Log user activities"""
        self.user_logger.info(
            activity,
            extra={'user_id': user_id, **kwargs}
        )