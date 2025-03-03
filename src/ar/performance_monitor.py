import psutil
import time
from typing import Dict, Optional

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.last_fps_check = self.start_time
        self.current_fps = 0.0
        
    def update(self) -> Dict[str, float]:
        """Update and return current performance metrics"""
        self.frame_count += 1
        current_time = time.time()
        
        # Calculate FPS every second
        if current_time - self.last_fps_check >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_check)
            self.frame_count = 0
            self.last_fps_check = current_time
        
        return {
            'fps': self.current_fps,
            'battery_level': self.get_battery_level(),
            'temperature': self.get_device_temperature(),
            'memory_usage': self.get_memory_usage()
        }
    
    def get_battery_level(self) -> float:
        """Get current battery level percentage"""
        battery = psutil.sensors_battery()
        return battery.percent if battery else 100.0
    
    def get_device_temperature(self) -> float:
        """Get device temperature in Celsius"""
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get CPU temperature as approximation
                cpu_temp = next(iter(temps.values()))[0].current
                return cpu_temp
        except:
            pass
        return 0.0
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024