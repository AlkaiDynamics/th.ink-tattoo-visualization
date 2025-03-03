import time
import psutil
from typing import Dict

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.initial_battery = self.get_battery_percentage()
        
    def get_battery_percentage(self) -> float:
        battery = psutil.sensors_battery()
        return battery.percent if battery else 100.0
    
    def get_device_temperature(self) -> float:
        # Implementation will vary by device
        return 0.0
    
    def get_metrics(self) -> Dict:
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        current_battery = self.get_battery_percentage()
        
        return {
            'elapsed_time': elapsed_time,
            'battery_drain': self.initial_battery - current_battery,
            'temperature': self.get_device_temperature(),
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024  # MB
        }