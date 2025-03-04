"""
AR configuration module for the Th.ink application.

This module provides configuration classes for augmented reality features,
including AR visualization settings, device capabilities, and performance parameters.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Union, Any
from enum import Enum
import logging

# Configure logger
logger = logging.getLogger("think.ar")


class SubscriptionTier(Enum):
    """Subscription tier levels."""
    
    FREE = "free"
    PREMIUM = "premium"
    PRO = "pro"


class RenderingQuality(Enum):
    """Rendering quality levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class SurfaceMappingQuality(Enum):
    """Surface mapping quality levels."""
    
    BASIC = "basic"
    STANDARD = "standard"
    DETAILED = "detailed"
    PRECISE = "precise"


class DeviceCapability(Enum):
    """Device capability types."""
    
    STANDARD_CAMERA = "standard_camera"
    DEPTH_SENSOR = "depth_sensor"
    LIDAR = "lidar"
    MOTION_SENSORS = "motion_sensors"
    HIGH_PERFORMANCE_GPU = "high_performance_gpu"


@dataclass
class ARPerformanceSettings:
    """Performance settings for AR visualization."""
    
    target_fps: int = 60
    min_acceptable_fps: int = 30
    dynamic_quality_adjustment: bool = True
    reduce_quality_on_low_battery: bool = True
    thermal_throttling_enabled: bool = True
    low_battery_threshold: float = 20.0  # percentage
    thermal_threshold: float = 40.0  # Celsius
    memory_limit_mb: int = 1024  # Megabytes
    background_processing: bool = True


@dataclass
class ARCameraSettings:
    """Camera settings for AR visualization."""
    
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 60
    auto_focus: bool = True
    exposure_compensation: int = 0
    white_balance: str = "auto"
    stabilization: bool = True
    use_front_camera: bool = False
    preview_quality: str = "high"


@dataclass
class ARVisualizationSettings:
    """Visualization settings for AR effects."""
    
    shadow_quality: RenderingQuality = RenderingQuality.HIGH
    lighting_model: str = "physically_based"
    reflection_quality: RenderingQuality = RenderingQuality.MEDIUM
    tattoo_overlay_opacity: float = 0.9
    tattoo_blend_mode: str = "multiply"
    skin_tone_adaptation: bool = True
    antialiasing: bool = True
    texture_filtering: str = "trilinear"
    ambient_occlusion: bool = True
    hdr_rendering: bool = True
    outline_thickness: float = 1.0
    outline_color: Tuple[int, int, int] = (0, 0, 0)  # RGB


@dataclass
class ARTrackingSettings:
    """Body tracking settings for AR visualization."""
    
    tracking_mode: str = "full_body"  # full_body, upper_body, arms_only
    tracking_precision: float = 0.95  # 0.0 to 1.0
    smoothing_factor: float = 0.5  # 0.0 to 1.0
    prediction_frames: int = 5
    keypoint_confidence_threshold: float = 0.7
    occlusion_handling: bool = True
    reacquisition_delay_ms: int = 500
    stabilization_enabled: bool = True
    maximum_tracking_range_cm: int = 300


@dataclass
class ARSkinDetectionSettings:
    """Skin detection settings for AR visualization."""
    
    detection_threshold: float = 0.85
    color_model: str = "hsv"  # hsv, ycrcb, rgb
    adapt_to_lighting: bool = True
    smoothing_kernel_size: int = 3
    refinement_iterations: int = 2
    use_face_landmarks: bool = True
    minimum_skin_area_percentage: float = 5.0
    maximum_skin_area_percentage: float = 95.0
    dynamic_threshold_adjustment: bool = True


@dataclass
class ARNeRFSettings:
    """NeRF Metahuman Avatar settings."""
    
    enabled: bool = True
    resolution: int = 512
    samples_per_ray: int = 64
    chunk_size: int = 32768
    network_depth: int = 8
    network_width: int = 256
    use_view_directions: bool = True
    white_background: bool = False
    render_factor: int = 1  # Downsampling factor, 1 means full resolution
    num_encoding_functions: int = 10
    learning_rate: float = 5e-4
    batch_size: int = 4096
    precision: str = "half"  # half, single, double
    use_hierarchical_sampling: bool = True
    perturb_samples: bool = True
    lindisp: bool = False
    use_fine_network: bool = True
    view_dependent_effects: bool = True


@dataclass
class DeviceSpecificSettings:
    """Device-specific AR settings."""
    
    depth_precision: str = "medium"  # low, medium, high
    surface_mapping: SurfaceMappingQuality = SurfaceMappingQuality.STANDARD
    scan_angles_required: int = 5
    use_accelerometer: bool = True
    use_gyroscope: bool = True
    gpu_optimization_level: int = 2  # 0-3, with 3 being most aggressive
    cpu_threads: int = 4
    memory_buffer_mb: int = 200


@dataclass
class ARConfig:
    """
    Main configuration class for AR visualization.
    
    This class combines all AR-specific settings and provides methods
    for adjusting configuration based on device capabilities and performance.
    """
    
    # Core settings
    camera: ARCameraSettings = field(default_factory=ARCameraSettings)
    visualization: ARVisualizationSettings = field(default_factory=ARVisualizationSettings)
    tracking: ARTrackingSettings = field(default_factory=ARTrackingSettings)
    skin_detection: ARSkinDetectionSettings = field(default_factory=ARSkinDetectionSettings)
    performance: ARPerformanceSettings = field(default_factory=ARPerformanceSettings)
    nerf: ARNeRFSettings = field(default_factory=ARNeRFSettings)
    
    # Subscription-based limits
    preview_limits: Dict[SubscriptionTier, int] = field(default_factory=lambda: {
        SubscriptionTier.FREE: 5,
        SubscriptionTier.PREMIUM: -1,  # Unlimited
        SubscriptionTier.PRO: -1       # Unlimited
    })
    
    # Device-specific settings
    device_specific_settings: Dict[DeviceCapability, DeviceSpecificSettings] = field(default_factory=lambda: {
        DeviceCapability.STANDARD_CAMERA: DeviceSpecificSettings(
            depth_precision="low",
            surface_mapping=SurfaceMappingQuality.BASIC,
            scan_angles_required=6
        ),
        DeviceCapability.DEPTH_SENSOR: DeviceSpecificSettings(
            depth_precision="medium",
            surface_mapping=SurfaceMappingQuality.STANDARD,
            scan_angles_required=4
        ),
        DeviceCapability.LIDAR: DeviceSpecificSettings(
            depth_precision="high",
            surface_mapping=SurfaceMappingQuality.PRECISE,
            scan_angles_required=3
        )
    })
    
    def __post_init__(self):
        """Initialize configuration."""
        self._current_quality_level = RenderingQuality.HIGH
    
    def adjust_quality_for_performance(self, 
                                      battery_level: float, 
                                      temperature: float, 
                                      current_fps: float) -> bool:
        """
        Dynamically adjust AR quality based on device conditions.
        
        Args:
            battery_level: Current battery level (percentage)
            temperature: Current device temperature (Celsius)
            current_fps: Current frames per second
            
        Returns:
            True if quality settings were adjusted, False otherwise
        """
        if not self.performance.dynamic_quality_adjustment:
            return False
        
        # Check if we need to reduce quality
        reduce_quality = False
        
        # Check battery level
        if (self.performance.reduce_quality_on_low_battery and 
                battery_level < self.performance.low_battery_threshold):
            logger.info(f"Reducing quality due to low battery: {battery_level:.1f}%")
            reduce_quality = True
        
        # Check temperature
        if (self.performance.thermal_throttling_enabled and 
                temperature > self.performance.thermal_threshold):
            logger.info(f"Reducing quality due to high temperature: {temperature:.1f}°C")
            reduce_quality = True
        
        # Check FPS
        if current_fps < self.performance.min_acceptable_fps:
            logger.info(f"Reducing quality due to low FPS: {current_fps:.1f}")
            reduce_quality = True
        
        # Apply quality changes if needed
        if reduce_quality:
            self._apply_lower_quality_settings()
            return True
        elif (battery_level > self.performance.low_battery_threshold + 10 and
              temperature < self.performance.thermal_threshold - 5 and
              current_fps > self.performance.target_fps * 0.9):
            # Conditions are good, try to restore quality
            self._restore_quality_settings()
            return True
        
        return False
    
    def _apply_lower_quality_settings(self) -> None:
        """Apply lower quality settings to conserve resources."""
        current_quality = self._current_quality_level
        
        # Don't go below LOW quality
        if current_quality == RenderingQuality.LOW:
            return
        
        # Determine new quality level
        if current_quality == RenderingQuality.ULTRA:
            new_quality = RenderingQuality.HIGH
        elif current_quality == RenderingQuality.HIGH:
            new_quality = RenderingQuality.MEDIUM
        else:
            new_quality = RenderingQuality.LOW
        
        # Apply quality settings
        self._set_quality_level(new_quality)
        logger.info(f"Quality reduced from {current_quality.value} to {new_quality.value}")
    
    def _restore_quality_settings(self) -> None:
        """Try to restore higher quality settings if conditions allow."""
        current_quality = self._current_quality_level
        
        # Don't go above ULTRA quality
        if current_quality == RenderingQuality.ULTRA:
            return
        
        # Determine new quality level
        if current_quality == RenderingQuality.LOW:
            new_quality = RenderingQuality.MEDIUM
        elif current_quality == RenderingQuality.MEDIUM:
            new_quality = RenderingQuality.HIGH
        else:
            new_quality = RenderingQuality.ULTRA
        
        # Apply quality settings
        self._set_quality_level(new_quality)
        logger.info(f"Quality increased from {current_quality.value} to {new_quality.value}")
    
    def _set_quality_level(self, quality: RenderingQuality) -> None:
        """
        Set specific quality level for AR visualization.
        
        Args:
            quality: Target quality level
        """
        self._current_quality_level = quality
        
        # Adjust visualization settings
        self.visualization.shadow_quality = quality
        self.visualization.reflection_quality = quality
        
        # Adjust camera settings
        if quality == RenderingQuality.LOW:
            self.camera.resolution = (1280, 720)
            self.camera.fps = 30
        elif quality == RenderingQuality.MEDIUM:
            self.camera.resolution = (1920, 1080)
            self.camera.fps = 45
        elif quality == RenderingQuality.HIGH:
            self.camera.resolution = (1920, 1080)
            self.camera.fps = 60
        else:  # ULTRA
            self.camera.resolution = (2560, 1440)
            self.camera.fps = 60
        
        # Adjust tracking settings
        if quality == RenderingQuality.LOW:
            self.tracking.prediction_frames = 2
            self.tracking.smoothing_factor = 0.7  # More smoothing
        else:
            self.tracking.prediction_frames = 5
            self.tracking.smoothing_factor = 0.5
        
        # Adjust NeRF settings
        if quality == RenderingQuality.LOW:
            self.nerf.resolution = 256
            self.nerf.samples_per_ray = 32
            self.nerf.render_factor = 2  # Half resolution
        elif quality == RenderingQuality.MEDIUM:
            self.nerf.resolution = 384
            self.nerf.samples_per_ray = 48
            self.nerf.render_factor = 1
        elif quality == RenderingQuality.HIGH:
            self.nerf.resolution = 512
            self.nerf.samples_per_ray = 64
            self.nerf.render_factor = 1
        else:  # ULTRA
            self.nerf.resolution = 768
            self.nerf.samples_per_ray = 96
            self.nerf.render_factor = 1
        
        # Adjust skin detection settings
        if quality == RenderingQuality.LOW:
            self.skin_detection.refinement_iterations = 1
        else:
            self.skin_detection.refinement_iterations = 2
    
    def get_scan_requirements(self, capabilities: List[DeviceCapability]) -> DeviceSpecificSettings:
        """
        Get scanning requirements based on device capabilities.
        
        Args:
            capabilities: List of device capabilities
            
        Returns:
            Device-specific settings for given capabilities
        """
        # Check for best available capability in order of preference
        for capability in [DeviceCapability.LIDAR, 
                           DeviceCapability.DEPTH_SENSOR, 
                           DeviceCapability.STANDARD_CAMERA]:
            if capability in capabilities and capability in self.device_specific_settings:
                return self.device_specific_settings[capability]
        
        # Default to standard camera if no match
        return self.device_specific_settings.get(
            DeviceCapability.STANDARD_CAMERA,
            DeviceSpecificSettings()
        )
    
    def get_preview_limit(self, tier: SubscriptionTier) -> int:
        """
        Get preview limit for subscription tier.
        
        Args:
            tier: Subscription tier
            
        Returns:
            Number of previews allowed (-1 for unlimited)
        """
        return self.preview_limits.get(tier, 5)  # Default to FREE tier limit
    
    def is_nerf_avatar_enabled(self, tier: SubscriptionTier) -> bool:
        """
        Check if NeRF Metahuman Avatar is enabled for subscription tier.
        
        Args:
            tier: Subscription tier
            
        Returns:
            True if NeRF avatar is enabled for this tier
        """
        if tier == SubscriptionTier.FREE:
            return False
        return self.nerf.enabled