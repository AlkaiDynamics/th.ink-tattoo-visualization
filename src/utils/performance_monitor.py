"""
Performance monitoring utilities for the Th.ink AR application.
"""

import time
import logging
import psutil
import functools
from typing import Dict, Optional, Callable, Any, TypeVar, cast
from datetime import datetime

# Type variable for generic function decorator
F = TypeVar('F', bound=Callable[..., Any])

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitor system performance metrics including CPU, RAM, battery, and execution time.
    
    This class provides utilities to track performance metrics for the AR application,
    helping to identify bottlenecks and optimize resource usage.
    """
    
    def __init__(self, 
                 log_metrics: bool = True, 
                 metrics_interval: float = 5.0, 
                 thermal_threshold: float = 80.0):
        """
        Initialize the performance monitor.
        
        Args:
            log_metrics: Whether to automatically log metrics periodically
            metrics_interval: Interval in seconds between metrics logging
            thermal_threshold: Temperature threshold in Celsius for warnings
        """
        self.start_time = time.time()
        self.frame_count = 0
        self.last_frame_time = self.start_time
        self.last_metrics_time = self.start_time
        self.current_fps = 0.0
        self.log_metrics = log_metrics
        self.metrics_interval = metrics_interval
        self.thermal_threshold = thermal_threshold
        
        # Initialize counters
        self.processing_times: Dict[str, float] = {}
        self.call_counts: Dict[str, int] = {}
        
        # Get initial battery state
        self.initial_battery = self.get_battery_level()
        
        # Track memory usage
        self.peak_memory_usage = 0.0
        
        logger.info("Performance monitor initialized")
    
    def update(self) -> Dict[str, float]:
        """
        Update performance metrics with the latest frame processing.
        
        Returns:
            Dict containing current performance metrics
        """
        self.frame_count += 1
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Calculate FPS based on exponential moving average
        alpha = 0.2  # Smoothing factor
        if self.current_fps == 0:
            self.current_fps = 1.0 / max(frame_time, 0.001)
        else:
            self.current_fps = alpha * (1.0 / max(frame_time, 0.001)) + (1 - alpha) * self.current_fps
        
        # Update memory usage
        memory_usage = self.get_memory_usage()
        self.peak_memory_usage = max(self.peak_memory_usage, memory_usage)
        
        metrics = {
            'fps': self.current_fps,
            'frame_time_ms': frame_time * 1000,
            'battery_level': self.get_battery_level(),
            'battery_drain': self.initial_battery - self.get_battery_level(),
            'temperature': self.get_device_temperature(),
            'memory_usage_mb': memory_usage,
            'peak_memory_mb': self.peak_memory_usage,
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'uptime_seconds': current_time - self.start_time
        }
        
        # Check if it's time to log metrics
        if self.log_metrics and (current_time - self.last_metrics_time) >= self.metrics_interval:
            self._log_current_metrics(metrics)
            self.last_metrics_time = current_time
        
        # Check for performance warnings
        self._check_performance_warnings(metrics)
        
        return metrics
    
    def get_battery_level(self) -> float:
        """
        Get current battery level percentage.
        
        Returns:
            Battery level as a percentage (0-100), or 100 if not available
        """
        try:
            battery = psutil.sensors_battery()
            return battery.percent if battery else 100.0
        except Exception as e:
            logger.warning(f"Failed to get battery level: {e}")
            return 100.0
    
    def get_device_temperature(self) -> float:
        """
        Get device temperature in Celsius.
        
        Returns:
            Device temperature in Celsius, or 0.0 if not available
        """
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Find the highest temperature from available sensors
                max_temp = 0.0
                for name, entries in temps.items():
                    if entries:
                        current = entries[0].current
                        max_temp = max(max_temp, current)
                return max_temp
        except Exception as e:
            logger.warning(f"Failed to get device temperature: {e}")
        return 0.0
    
    def get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def _log_current_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log current performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        logger.info(
            f"Performance Metrics: "
            f"FPS: {metrics['fps']:.1f}, "
            f"Frame time: {metrics['frame_time_ms']:.1f}ms, "
            f"Memory: {metrics['memory_usage_mb']:.1f}MB, "
            f"CPU: {metrics['cpu_percent']:.1f}%, "
            f"Battery: {metrics['battery_level']:.1f}%, "
            f"Temp: {metrics['temperature']:.1f}°C"
        )
    
    def _check_performance_warnings(self, metrics: Dict[str, float]) -> None:
        """
        Check for performance warnings based on metrics.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        # Check for low FPS
        if metrics['fps'] < 30 and metrics['fps'] > 0:
            logger.warning(f"Low FPS detected: {metrics['fps']:.1f}")
        
        # Check for high temperature
        if metrics['temperature'] > self.thermal_threshold:
            logger.warning(f"High temperature detected: {metrics['temperature']:.1f}°C")
        
        # Check for high memory usage
        if metrics['memory_usage_mb'] > 1000:  # 1GB
            logger.warning(f"High memory usage: {metrics['memory_usage_mb']:.1f}MB")
        
        # Check for low battery
        if metrics['battery_level'] < 20:
            logger.warning(f"Low battery: {metrics['battery_level']:.1f}%")
    
    def reset_counters(self) -> None:
        """Reset all performance counters."""
        self.frame_count = 0
        self.current_fps = 0.0
        self.last_frame_time = time.time()
        self.processing_times = {}
        self.call_counts = {}
        self.peak_memory_usage = self.get_memory_usage()
        logger.info("Performance counters reset")
    
    def record_function_metrics(self, name: str) -> Callable[[F], F]:
        """
        Decorator to record performance metrics for a function.
        
        Args:
            name: Name identifier for the function being monitored
            
        Returns:
            Decorated function with performance monitoring
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    execution_time = time.time() - start_time
                    
                    if name not in self.processing_times:
                        self.processing_times[name] = 0.0
                        self.call_counts[name] = 0
                    
                    self.processing_times[name] += execution_time
                    self.call_counts[name] += 1
                    
                    # Log if execution time is unusually long
                    avg_time = self.processing_times[name] / self.call_counts[name]
                    if execution_time > avg_time * 2 and self.call_counts[name] > 10:
                        logger.warning(
                            f"Slow execution detected for {name}: "
                            f"{execution_time*1000:.1f}ms (avg: {avg_time*1000:.1f}ms)"
                        )
            
            return cast(F, wrapper)
        
        return decorator


# Singleton instance for easy import
performance_monitor = PerformanceMonitor()


def measure_time(label: str) -> Callable[[F], F]:
    """
    Decorator to measure and log execution time of functions.
    
    Args:
        label: Label to identify the function in logs
        
    Returns:
        Decorated function
    """
    return performance_monitor.record_function_metrics(label)