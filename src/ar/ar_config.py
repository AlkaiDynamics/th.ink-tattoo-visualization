from dataclasses import dataclass
from typing import Tuple, Dict
from enum import Enum

class SubscriptionTier(Enum):
    FREE = "free"
    PREMIUM = "premium"
    PRO = "pro"

@dataclass
class ARConfig:
    CAMERA_RESOLUTION: Tuple[int, int] = (1920, 1080)
    MIN_FPS: int = 45
    MAX_FPS: int = 60
    SHADOW_QUALITY: str = "high"
    TRACKING_PRECISION: float = 0.95
    
    # AR-specific configurations
    SKIN_DETECTION_THRESHOLD: float = 0.85
    TATTOO_OVERLAY_OPACITY: float = 0.9
    SURFACE_MAPPING_QUALITY: str = "high"
    DEPTH_SENSING_ENABLED: bool = True
    
    # Subscription-based limits
    PREVIEW_LIMITS: Dict[SubscriptionTier, int] = {
        SubscriptionTier.FREE: 5,
        SubscriptionTier.PREMIUM: -1,  # Unlimited
        SubscriptionTier.PRO: -1       # Unlimited
    }
    
    # Device-specific AR settings
    DEVICE_SPECIFIC_SETTINGS: Dict[str, Dict] = {
        "lidar_enabled": {
            "depth_precision": "high",
            "surface_mapping": "detailed",
            "scan_angles_required": 4
        },
        "standard_camera": {
            "depth_precision": "medium",
            "surface_mapping": "standard",
            "scan_angles_required": 6
        }
    }
    
    # Performance thresholds
    THERMAL_THRESHOLD: float = 40.0  # Celsius
    BATTERY_THRESHOLD: float = 20.0  # Percentage
    TARGET_BATTERY_DRAIN: float = 0.1  # 10% per 15 minutes
    
    def adjust_quality(self, battery_level: float, temperature: float):
        """Dynamically adjust AR quality based on device conditions"""
        if temperature > self.THERMAL_THRESHOLD or battery_level < self.BATTERY_THRESHOLD:
            self.SHADOW_QUALITY = "low"
            self.MAX_FPS = 45
            self.SURFACE_MAPPING_QUALITY = "medium"
            return True
        return False
    
    def get_scan_requirements(self, has_lidar: bool) -> Dict:
        """Return scanning requirements based on device capabilities"""
        return self.DEVICE_SPECIFIC_SETTINGS["lidar_enabled" if has_lidar else "standard_camera"]