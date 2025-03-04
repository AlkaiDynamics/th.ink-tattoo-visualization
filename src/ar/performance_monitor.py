"""
Performance monitoring module for the Th.ink AR application.

This module provides real-time monitoring of AR performance metrics such as frame rate,
processing times, memory usage, and battery consumption, with support for adaptive
quality adjustments based on device conditions.
"""

import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Tuple, Deque
from collections import deque
import numpy as np
import threading
from dataclasses import dataclass, field

# Configure logger
logger = logging.getLogger("think.ar.performance")


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    fps: float = 0.0
    frame_time: float = 0.0
    battery_level: float = 100.0
    temperature: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: Optional[float] = None
    tracking_time: float = 0.0
    rendering_time: float = 0.0
    skin_detection_time: float = 0.0
    total_processing_time: float = 0.0
    dropped_frames: int = 0
    total_frames: int = 0
    timestamp: float = time.time()


@dataclass
class PerformanceThresholds:
    """Performance thresholds for quality adjustment."""
    
    min_fps: float = 25.0
    max_frame_time: float = 0.04  # 40ms
    min_battery: float = 20.0
    max_temperature: float = 40.0
    max_memory_usage: float = 1024.0  # MB
    max_cpu_usage: float = 80.0
    max_gpu_usage: float = 90.0


class PerformanceMonitor:
    """
    Performance monitoring system for the AR application.
    
    This class provides methods to track performance metrics in real-time,
    with support for performance analysis and quality adjustments.
    """
    
    def __init__(self, 
                window_size: int = 60, 
                thresholds: Optional[PerformanceThresholds] = None,
                auto_adjust_quality: bool = True):
        """
        Initialize the performance monitor.
        
        Args:
            window_size: Number of frames to consider for moving averages
            thresholds: Performance thresholds for quality adjustment
            auto_adjust_quality: Whether to automatically adjust quality based on performance
        """
        self.start_time = time.time()
        self.frame_count = 0
        self.last_frame_time = self.start_time
        self.fps_update_interval = 0.5  # Update FPS every 0.5 seconds
        self.last_fps_update_time = self.start_time
        self.current_fps = 0.0
        
        # Moving window of frame times for stable FPS calculation
        self.window_size = window_size
        self.frame_times: Deque[float] = deque(maxlen=window_size)
        
        # Performance thresholds
        self.thresholds = thresholds if thresholds is not None else PerformanceThresholds()
        self.auto_adjust_quality = auto_adjust_quality
        
        # Get initial battery state
        self.initial_battery = self.get_battery_level()
        self.battery_history: Deque[Tuple[float, float]] = deque(maxlen=window_size)
        
        # Track component times
        self.component_times: Dict[str, Deque[float]] = {
            "tracking": deque(maxlen=window_size),
            "skin_detection": deque(maxlen=window_size),
            "rendering": deque(maxlen=window_size),
            "total": deque(maxlen=window_size)
        }
        
        # Performance history for trend analysis
        self.metrics_history: Deque[PerformanceMetrics] = deque(maxlen=window_size * 10)
        
        # Performance alerts
        self.alerts: List[Dict[str, Any]] = []
        self.max_alerts = 100
        
        # Quality levels
        self.quality_level = 2  # 0=Low, 1=Medium, 2=High, 3=Ultra
        self.quality_level_names = ["Low", "Medium", "High", "Ultra"]
        self.quality_change_timestamp = self.start_time
        self.min_quality_change_interval = 5.0  # Minimum seconds between quality changes
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("Performance monitor initialized")
    
    def update(self) -> Dict[str, float]:
        """
        Update performance metrics with the latest frame processing.
        
        Returns:
            Dictionary containing current performance metrics
        """
        with self.lock:
            self.frame_count += 1
            current_time = time.time()
            frame_time = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            # Add to frame times history
            self.frame_times.append(frame_time)
            
            # Calculate current FPS using moving average
            if current_time - self.last_fps_update_time >= self.fps_update_interval:
                if self.frame_times:
                    avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                    self.current_fps = 1.0 / max(avg_frame_time, 0.001)
                else:
                    self.current_fps = 0.0
                
                self.last_fps_update_time = current_time
            
            # Get system metrics
            battery_level = self.get_battery_level()
            temperature = self.get_device_temperature()
            memory_usage = self.get_memory_usage()
            cpu_usage = psutil.cpu_percent(interval=0.1)
            gpu_usage = self.get_gpu_usage()
            
            # Track battery usage over time
            self.battery_history.append((current_time, battery_level))
            
            # Create metrics object
            metrics = PerformanceMetrics(
                fps=self.current_fps,
                frame_time=frame_time * 1000,  # Convert to milliseconds
                battery_level=battery_level,
                temperature=temperature,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                gpu_usage=gpu_usage,
                total_frames=self.frame_count,
                timestamp=current_time
            )
            
            # Add to history
            self.metrics_history.append(metrics)
            
            # Check for performance issues
            self._check_performance_warnings(metrics)
            
            # Adaptive quality adjustment if enabled
            if self.auto_adjust_quality:
                self._adjust_quality_if_needed(metrics)
            
            # Return current metrics as dictionary
            return {
                'fps': metrics.fps,
                'frame_time_ms': metrics.frame_time,
                'battery_level': metrics.battery_level,
                'battery_drain': self.initial_battery - metrics.battery_level,
                'temperature': metrics.temperature,
                'memory_usage_mb': metrics.memory_usage,
                'cpu_percent': metrics.cpu_usage,
                'gpu_percent': metrics.gpu_usage if metrics.gpu_usage is not None else 0.0,
                'tracking_time_ms': metrics.tracking_time,
                'rendering_time_ms': metrics.rendering_time,
                'skin_detection_time_ms': metrics.skin_detection_time,
                'total_processing_time_ms': metrics.total_processing_time,
                'quality_level': self.quality_level_names[self.quality_level],
                'dropped_frames': metrics.dropped_frames,
                'uptime_seconds': current_time - self.start_time
            }
    
    def record_component_time(self, component: str, execution_time: float) -> None:
        """
        Record execution time for a specific component.
        
        Args:
            component: Component name ("tracking", "skin_detection", "rendering", "total")
            execution_time: Execution time in seconds
        """
        with self.lock:
            if component in self.component_times:
                self.component_times[component].append(execution_time)
                
                # Update the latest metrics object
                if self.metrics_history:
                    metrics = self.metrics_history[-1]
                    if component == "tracking":
                        metrics.tracking_time = execution_time * 1000  # ms
                    elif component == "skin_detection":
                        metrics.skin_detection_time = execution_time * 1000  # ms
                    elif component == "rendering":
                        metrics.rendering_time = execution_time * 1000  # ms
                    elif component == "total":
                        metrics.total_processing_time = execution_time * 1000  # ms
    
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
    
    def get_gpu_usage(self) -> Optional[float]:
        """
        Get GPU usage percentage if available.
        
        Returns:
            GPU usage as a percentage, or None if not available
        """
        try:
            # Try to import nvidia-ml-py package if available
            import pynvml
            
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            pynvml.nvmlShutdown()
            
            return util.gpu
        except (ImportError, Exception):
            try:
                # Try to get GPU info using AMD ROCm if available
                import rocm_smi_lib as rsmi
                
                rsmi.rsmi_init()
                gpu_count = rsmi.rsmi_num_monitor_devices()
                
                if gpu_count > 0:
                    usage = 0
                    for i in range(gpu_count):
                        usage_rate = rsmi.rsmi_dev_gpu_busy_percent_get(i)[1]
                        usage += usage_rate
                    
                    rsmi.rsmi_shut_down()
                    return usage / gpu_count
            except (ImportError, Exception):
                # GPU monitoring not available
                pass
        
        return None
    
    def get_battery_drain_rate(self) -> float:
        """
        Calculate battery drain rate in percentage per hour.
        
        Returns:
            Battery drain rate in percentage per hour
        """
        if len(self.battery_history) < 2:
            return 0.0
        
        # Get oldest and newest battery measurements
        oldest = self.battery_history[0]
        newest = self.battery_history[-1]
        
        time_diff = newest[0] - oldest[0]  # seconds
        if time_diff < 60:  # Require at least 1 minute of data
            return 0.0
        
        battery_diff = oldest[1] - newest[1]  # percentage points
        
        # Calculate hourly rate
        hourly_rate = battery_diff * (3600 / time_diff)
        
        return max(0.0, hourly_rate)  # Ensure non-negative
    
    def get_component_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for each component.
        
        Returns:
            Dictionary with component performance metrics
        """
        metrics = {}
        
        for component, times in self.component_times.items():
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                min_time = min(times)
                
                metrics[component] = {
                    "avg_time_ms": avg_time * 1000,
                    "max_time_ms": max_time * 1000,
                    "min_time_ms": min_time * 1000,
                    "percentage": (avg_time * 1000) / (1000 / max(self.current_fps, 0.001)) * 100 if self.current_fps > 0 else 0
                }
        
        return metrics
    
    def _check_performance_warnings(self, metrics: PerformanceMetrics) -> None:
        """
        Check for performance warnings based on metrics.
        
        Args:
            metrics: Current performance metrics
        """
        warnings = []
        
        # Check for low FPS
        if metrics.fps < self.thresholds.min_fps and metrics.fps > 0:
            warnings.append({
                "type": "low_fps",
                "value": metrics.fps,
                "threshold": self.thresholds.min_fps,
                "message": f"Low FPS detected: {metrics.fps:.1f}",
                "timestamp": metrics.timestamp,
                "severity": "warning"
            })
        
        # Check for high frame time
        if metrics.frame_time > self.thresholds.max_frame_time * 1000:
            warnings.append({
                "type": "high_frame_time",
                "value": metrics.frame_time,
                "threshold": self.thresholds.max_frame_time * 1000,
                "message": f"High frame time: {metrics.frame_time:.1f}ms",
                "timestamp": metrics.timestamp,
                "severity": "warning"
            })
        
        # Check for high temperature
        if metrics.temperature > self.thresholds.max_temperature:
            warnings.append({
                "type": "high_temperature",
                "value": metrics.temperature,
                "threshold": self.thresholds.max_temperature,
                "message": f"High temperature detected: {metrics.temperature:.1f}°C",
                "timestamp": metrics.timestamp,
                "severity": "warning"
            })
        
        # Check for low battery
        if metrics.battery_level < self.thresholds.min_battery:
            warnings.append({
                "type": "low_battery",
                "value": metrics.battery_level,
                "threshold": self.thresholds.min_battery,
                "message": f"Low battery: {metrics.battery_level:.1f}%",
                "timestamp": metrics.timestamp,
                "severity": "warning"
            })
        
        # Check for high memory usage
        if metrics.memory_usage > self.thresholds.max_memory_usage:
            warnings.append({
                "type": "high_memory_usage",
                "value": metrics.memory_usage,
                "threshold": self.thresholds.max_memory_usage,
                "message": f"High memory usage: {metrics.memory_usage:.1f}MB",
                "timestamp": metrics.timestamp,
                "severity": "warning"
            })
        
        # Check for high CPU usage
        if metrics.cpu_usage > self.thresholds.max_cpu_usage:
            warnings.append({
                "type": "high_cpu_usage",
                "value": metrics.cpu_usage,
                "threshold": self.thresholds.max_cpu_usage,
                "message": f"High CPU usage: {metrics.cpu_usage:.1f}%",
                "timestamp": metrics.timestamp,
                "severity": "warning"
            })
        
        # Check for high GPU usage if available
        if metrics.gpu_usage is not None and metrics.gpu_usage > self.thresholds.max_gpu_usage:
            warnings.append({
                "type": "high_gpu_usage",
                "value": metrics.gpu_usage,
                "threshold": self.thresholds.max_gpu_usage,
                "message": f"High GPU usage: {metrics.gpu_usage:.1f}%",
                "timestamp": metrics.timestamp,
                "severity": "warning"
            })
        
        # Add warnings to alert list
        for warning in warnings:
            self.alerts.append(warning)
            # Log warning
            logger.warning(warning["message"])
        
        # Limit alert list size
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
    
    def _adjust_quality_if_needed(self, metrics: PerformanceMetrics) -> None:
        """
        Adjust quality settings based on performance metrics.
        
        Args:
            metrics: Current performance metrics
        """
        current_time = time.time()
        
        # Don't change quality too frequently
        if current_time - self.quality_change_timestamp < self.min_quality_change_interval:
            return
        
        # Check if we need to reduce quality
        reduce_quality = False
        increase_quality = False
        
        # Conditions for reducing quality
        if metrics.fps < self.thresholds.min_fps and metrics.fps > 0:
            reduce_quality = True
        elif metrics.temperature > self.thresholds.max_temperature:
            reduce_quality = True
        elif metrics.battery_level < self.thresholds.min_battery:
            reduce_quality = True
        elif metrics.frame_time > self.thresholds.max_frame_time * 1000:
            reduce_quality = True
        
        # Conditions for increasing quality
        if (not reduce_quality and 
                metrics.fps > self.thresholds.min_fps * 1.5 and
                metrics.temperature < self.thresholds.max_temperature * 0.8 and
                metrics.battery_level > self.thresholds.min_battery * 2 and
                metrics.frame_time < self.thresholds.max_frame_time * 1000 * 0.5):
            increase_quality = True
        
        # Apply quality changes
        if reduce_quality and self.quality_level > 0:
            self.quality_level -= 1
            self.quality_change_timestamp = current_time
            logger.info(f"Reduced quality to {self.quality_level_names[self.quality_level]} due to performance issues")
        elif increase_quality and self.quality_level < 3:
            self.quality_level += 1
            self.quality_change_timestamp = current_time
            logger.info(f"Increased quality to {self.quality_level_names[self.quality_level]} due to good performance")
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Get performance alerts.
        
        Returns:
            List of performance alerts
        """
        return self.alerts
    
    def reset_counters(self) -> None:
        """Reset all performance counters."""
        with self.lock:
            self.frame_count = 0
            self.current_fps = 0.0
            self.frame_times.clear()
            self.component_times = {
                "tracking": deque(maxlen=self.window_size),
                "skin_detection": deque(maxlen=self.window_size),
                "rendering": deque(maxlen=self.window_size),
                "total": deque(maxlen=self.window_size)
            }
            self.battery_history.clear()
            self.metrics_history.clear()
            self.alerts.clear()
            self.start_time = time.time()
            self.last_frame_time = self.start_time
            self.last_fps_update_time = self.start_time
            self.initial_battery = self.get_battery_level()
            
            logger.info("Performance counters reset")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dictionary with performance report data
        """
        with self.lock:
            # Current metrics
            if self.metrics_history:
                current_metrics = self.metrics_history[-1]
            else:
                current_metrics = PerformanceMetrics()
            
            # Calculate averages from history
            if self.metrics_history:
                avg_fps = sum(m.fps for m in self.metrics_history) / len(self.metrics_history)
                avg_frame_time = sum(m.frame_time for m in self.metrics_history) / len(self.metrics_history)
                avg_cpu = sum(m.cpu_usage for m in self.metrics_history) / len(self.metrics_history)
                avg_memory = sum(m.memory_usage for m in self.metrics_history) / len(self.metrics_history)
                
                # GPU usage might be None for some entries
                gpu_values = [m.gpu_usage for m in self.metrics_history if m.gpu_usage is not None]
                avg_gpu = sum(gpu_values) / len(gpu_values) if gpu_values else None
            else:
                avg_fps = 0.0
                avg_frame_time = 0.0
                avg_cpu = 0.0
                avg_memory = 0.0
                avg_gpu = None
            
            # Component times
            component_metrics = self.get_component_metrics()
            
            # Battery drain
            battery_drain_rate = self.get_battery_drain_rate()
            estimated_runtime = current_metrics.battery_level / battery_drain_rate if battery_drain_rate > 0 else 0.0
            
            # Generate report
            report = {
                "current": {
                    "fps": current_metrics.fps,
                    "frame_time_ms": current_metrics.frame_time,
                    "battery_level": current_metrics.battery_level,
                    "temperature": current_metrics.temperature,
                    "memory_usage_mb": current_metrics.memory_usage,
                    "cpu_usage": current_metrics.cpu_usage,
                    "gpu_usage": current_metrics.gpu_usage
                },
                "average": {
                    "fps": avg_fps,
                    "frame_time_ms": avg_frame_time,
                    "cpu_usage": avg_cpu,
                    "memory_usage_mb": avg_memory,
                    "gpu_usage": avg_gpu
                },
                "components": component_metrics,
                "battery": {
                    "drain_rate_per_hour": battery_drain_rate,
                    "estimated_runtime_hours": estimated_runtime,
                    "initial_level": self.initial_battery,
                    "current_level": current_metrics.battery_level
                },
                "quality": {
                    "level": self.quality_level,
                    "name": self.quality_level_names[self.quality_level],
                    "auto_adjust": self.auto_adjust_quality
                },
                "session": {
                    "total_frames": self.frame_count,
                    "uptime_seconds": time.time() - self.start_time,
                    "alert_count": len(self.alerts)
                }
            }
            
            return report


# Factory function for creating performance monitor with predefined settings
def create_performance_monitor(
    device_preset: str = "auto", 
    window_size: int = 60,
    auto_adjust: bool = True
) -> PerformanceMonitor:
    """
    Create a performance monitor with settings based on device type.
    
    Args:
        device_preset: Device preset ("high_end", "mid_range", "low_end", "auto")
        window_size: Window size for moving averages
        auto_adjust: Whether to automatically adjust quality based on performance
        
    Returns:
        Configured performance monitor
    """
    # Auto-detect device capabilities if preset is "auto"
    if device_preset == "auto":
        # Check if GPU is available
        gpu_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            gpu_available = gpu_count > 0
        except ImportError:
            try:
                import rocm_smi_lib as rsmi
                rsmi.rsmi_init()
                gpu_count = rsmi.rsmi_num_monitor_devices()
                rsmi.rsmi_shut_down()
                gpu_available = gpu_count > 0
            except ImportError:
                pass
        
        # Check CPU
        cpu_count = psutil.cpu_count(logical=False)
        if cpu_count is None:
            cpu_count = psutil.cpu_count()
        
        # Determine device class
        if gpu_available and cpu_count >= 4:
            device_preset = "high_end"
        elif cpu_count >= 2:
            device_preset = "mid_range"
        else:
            device_preset = "low_end"
    
    # Create thresholds based on device preset
    if device_preset == "high_end":
        thresholds = PerformanceThresholds(
            min_fps=45.0,
            max_frame_time=0.022,  # 22ms
            min_battery=15.0,
            max_temperature=75.0,
            max_memory_usage=2048.0,  # 2GB
            max_cpu_usage=85.0,
            max_gpu_usage=85.0
        )
    elif device_preset == "mid_range":
        thresholds = PerformanceThresholds(
            min_fps=30.0,
            max_frame_time=0.033,  # 33ms
            min_battery=20.0,
            max_temperature=65.0,
            max_memory_usage=1024.0,  # 1GB
            max_cpu_usage=80.0,
            max_gpu_usage=80.0
        )
    else:  # low_end
        thresholds = PerformanceThresholds(
            min_fps=20.0,
            max_frame_time=0.05,  # 50ms
            min_battery=25.0,
            max_temperature=60.0,
            max_memory_usage=512.0,  # 512MB
            max_cpu_usage=75.0,
            max_gpu_usage=75.0
        )
    
    # Create and return monitor
    monitor = PerformanceMonitor(
        window_size=window_size,
        thresholds=thresholds,
        auto_adjust_quality=auto_adjust
    )
    
    return monitor