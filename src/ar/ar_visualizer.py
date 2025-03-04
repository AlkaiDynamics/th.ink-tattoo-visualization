"""
AR visualization core for the Th.ink AR Tattoo Visualizer.

This module provides the main AR functionality for projecting tattoo designs
onto the user's body in real-time, with features for tracking, skin detection,
and dynamic tattoo rendering with lighting and perspective adjustments.
"""

import os
import logging
import asyncio
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import uuid

from .ar_config import ARConfig, SubscriptionTier, DeviceCapability
from .motion_tracker import MotionTracker, TrackingModel
from .skin_detector import SkinDetector, SkinColorModel
from .tattoo_renderer import TattooRenderer, BlendMode
from .performance_monitor import PerformanceMonitor

from ..ai.model_handler import get_model_handler
from ..ai.tattoo_generator import get_tattoo_generator

from ..subscription.subscription_manager import get_subscription_manager
from ..privacy.privacy_manager import get_privacy_manager
from ..auth.auth_manager import get_auth_manager
from ..data.data_manager import DataManager
from ..session.session_manager import get_session_manager
from ..errors.error_handler import handle_errors, ARError, ErrorLevel
from ..logging.log_manager import get_logger

# Get logger
logger = get_logger()


@dataclass
class ARState:
    """State data for the AR visualization system."""
    
    is_active: bool = False
    is_tracking: bool = False
    frame_count: int = 0
    fps: float = 0.0
    current_overlay: Optional[np.ndarray] = None
    overlay_id: Optional[str] = None
    current_design_id: Optional[int] = None
    overlay_position: Optional[Dict[str, float]] = None
    overlay_scale: float = 1.0
    overlay_rotation: float = 0.0
    detected_body_parts: List[str] = None
    skin_confidence: float = 0.0
    preview_mode: bool = True
    ar_session_start: Optional[datetime] = None
    last_activity: datetime = None
    target_body_part: Optional[str] = None
    device_capabilities: List[DeviceCapability] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.detected_body_parts is None:
            self.detected_body_parts = []
        
        if self.device_capabilities is None:
            self.device_capabilities = []
            
        if self.last_activity is None:
            self.last_activity = datetime.now()


class ARVisualizer:
    """
    Core AR visualization system for tattoo placement and preview.
    
    This class integrates motion tracking, skin detection, and tattoo rendering
    to provide a real-time AR experience for visualizing tattoos on the body.
    """
    
    def __init__(self, config: Optional[ARConfig] = None):
        """
        Initialize the AR visualizer.
        
        Args:
            config: AR configuration or None to use default
        """
        # Load configuration
        self.config = config or ARConfig()
        
        # Initialize state
        self.state = ARState()
        
        # Camera setup
        self.camera = None
        self.camera_id = 0
        
        # Initialize components
        self.motion_tracker = None
        self.skin_detector = None
        self.tattoo_renderer = None
        self.performance_monitor = PerformanceMonitor()
        
        # Get service managers
        self.model_handler = get_model_handler()
        self.tattoo_generator = get_tattoo_generator()
        self.subscription_manager = get_subscription_manager()
        self.privacy_manager = get_privacy_manager()
        self.auth_manager = get_auth_manager()
        self.data_manager = DataManager()
        self.session_manager = get_session_manager()
        
        # Session info
        self.user_id = None
        self.user_token = None
        self.session_id = None
        
        # Design cache
        self.design_cache: Dict[str, Dict[str, Any]] = {}
        
        # Temp directory
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("AR Visualizer initialized")
    
    @handle_errors()
    async def initialize(self) -> bool:
        """
        Initialize AR subsystems.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Detect device capabilities
            self.state.device_capabilities = self._detect_device_capabilities()
            
            # Initialize tracking
            tracking_success = await self._initialize_tracking()
            
            # Initialize skin detection
            skin_success = self._initialize_skin_detection()
            
            # Initialize rendering
            rendering_success = self._initialize_rendering()
            
            # Initialize camera if needed for preview
            camera_success = self._initialize_camera()
            
            # Set initialization status
            self.state.is_active = tracking_success and skin_success and rendering_success
            
            if self.state.is_active:
                logger.info("AR Visualizer successfully initialized")
            else:
                logger.error("AR Visualizer initialization failed")
                components = {
                    "tracking": tracking_success,
                    "skin_detection": skin_success,
                    "rendering": rendering_success,
                    "camera": camera_success
                }
                logger.error(f"Component status: {components}")
            
            return self.state.is_active
            
        except Exception as e:
            logger.error(f"Error initializing AR Visualizer: {str(e)}")
            self.state.is_active = False
            return False
    
    async def _initialize_tracking(self) -> bool:
        """
        Initialize motion tracking.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine optimal tracking model based on device capabilities
            if DeviceCapability.HIGH_PERFORMANCE_GPU in self.state.device_capabilities:
                model_type = TrackingModel.BLAZEPOSE
            elif DeviceCapability.LIDAR in self.state.device_capabilities or DeviceCapability.DEPTH_SENSOR in self.state.device_capabilities:
                model_type = TrackingModel.MEDIAPIPE
            else:
                model_type = TrackingModel.MEDIAPIPE  # Most compatible
            
            # Create motion tracker
            self.motion_tracker = MotionTracker(model_type=model_type)
            
            # Initialize tracking model
            success = await self.motion_tracker.initialize_model()
            
            logger.info(f"Motion tracking initialized with {model_type.value} model")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to initialize motion tracking: {str(e)}")
            return False
    
    def _initialize_skin_detection(self) -> bool:
        """
        Initialize skin detection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine optimal color model based on device capabilities
            if DeviceCapability.HIGH_PERFORMANCE_GPU in self.state.device_capabilities:
                color_model = SkinColorModel.ADAPTIVE
            else:
                color_model = SkinColorModel.HSV  # Faster
            
            # Create skin detector
            self.skin_detector = SkinDetector(
                threshold=self.config.skin_detection.detection_threshold,
                color_model=color_model,
                adapt_to_lighting=self.config.skin_detection.adapt_to_lighting
            )
            
            logger.info(f"Skin detection initialized with {color_model.value} model")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize skin detection: {str(e)}")
            return False
    
    def _initialize_rendering(self) -> bool:
        """
        Initialize tattoo rendering.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create tattoo renderer
            self.tattoo_renderer = TattooRenderer(
                opacity=self.config.visualization.tattoo_overlay_opacity,
                blend_mode=BlendMode(self.config.visualization.tattoo_blend_mode),
                adapt_to_lighting=True,
                high_quality=DeviceCapability.HIGH_PERFORMANCE_GPU in self.state.device_capabilities
            )
            
            logger.info("Tattoo rendering initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize tattoo rendering: {str(e)}")
            return False
    
    def _initialize_camera(self) -> bool:
        """
        Initialize camera capture.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Skip if camera already initialized
            if self.camera is not None and self.camera.isOpened():
                return True
            
            # Initialize camera
            self.camera = cv2.VideoCapture(self.camera_id)
            
            # Configure camera settings
            width, height = self.config.camera.resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
            
            # Check if camera is opened
            if not self.camera.isOpened():
                logger.error("Failed to open camera")
                return False
            
            # Log camera info
            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {str(e)}")
            return False
    
    def _detect_device_capabilities(self) -> List[DeviceCapability]:
        """
        Detect device capabilities.
        
        Returns:
            List of detected device capabilities
        """
        capabilities = [DeviceCapability.STANDARD_CAMERA]
        
        # Check for GPU
        if torch.cuda.is_available():
            capabilities.append(DeviceCapability.HIGH_PERFORMANCE_GPU)
        
        # For other capabilities, we would need platform-specific checks
        # This is a simplified implementation
        
        return capabilities
    
    @handle_errors()
    async def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a camera frame to apply AR effects.
        
        Args:
            frame: Input camera frame
            
        Returns:
            Tuple of (processed frame, performance metrics)
            
        Raises:
            ARError: If frame processing fails
        """
        if not self.state.is_active:
            return frame, {"error": "AR system not active"}
        
        try:
            # Start timing
            start_time = time.time()
            
            # Update frame count
            self.state.frame_count += 1
            
            # Update last activity
            self.state.last_activity = datetime.now()
            
            # Update AR session tracking
            if self.state.ar_session_start is None:
                self.state.ar_session_start = datetime.now()
            
            # Get performance metrics
            metrics = self.performance_monitor.update()
            
            # Update session metrics if in a session
            if self.session_id:
                self.session_manager.update_session(self.session_id, {
                    'frames_processed': 1,
                    'processing_time': metrics.get('processing_time', 0)
                })
            
            # Update settings based on performance
            self._update_performance_settings(
                metrics['battery_level'],
                metrics['temperature']
            )
            
            # Skip if not tracking or no overlay
            if not self.state.is_tracking or self.state.current_overlay is None:
                # Return original frame with metrics
                metrics['processing_time'] = time.time() - start_time
                return frame, metrics
            
            # Track motion and get keypoints
            tracking_start = time.time()
            keypoints, tracked_frame = await self.motion_tracker.track_frame(frame)
            metrics['tracking_time'] = time.time() - tracking_start
            
            # Update detected body parts
            self.state.detected_body_parts = [kp.get("part") for kp in keypoints if kp.get("confidence", 0) > 0.7]
            
            # Apply skin detection
            skin_start = time.time()
            skin_mask, confidence = await self.skin_detector.detect(tracked_frame)
            metrics['skin_detection_time'] = time.time() - skin_start
            
            # Update skin confidence
            self.state.skin_confidence = confidence
            
            # Refine skin mask using keypoints
            if keypoints:
                refined_mask = await self.skin_detector.refine_mask(skin_mask, keypoints)
            else:
                refined_mask = skin_mask
            
            # Apply tattoo overlay if available and confidence is high enough
            rendering_start = time.time()
            if self.state.current_overlay is not None and keypoints and confidence >= self.config.skin_detection.detection_threshold:
                result_frame = await self._apply_overlay(tracked_frame, refined_mask, keypoints)
            else:
                result_frame = tracked_frame
            
            metrics['rendering_time'] = time.time() - rendering_start
            
            # Update FPS
            frame_time = time.time() - start_time
            alpha = 0.2  # Smoothing factor
            fps = 1.0 / max(frame_time, 0.001)
            self.state.fps = alpha * fps + (1 - alpha) * self.state.fps
            
            # Add metrics
            metrics['fps'] = self.state.fps
            metrics['frame_time'] = frame_time * 1000  # Convert to ms
            metrics['processing_time'] = frame_time
            metrics['skin_confidence'] = confidence
            metrics['detected_body_parts'] = self.state.detected_body_parts
            
            return result_frame, metrics
            
        except Exception as e:
            error_msg = f"Error processing frame: {str(e)}"
            logger.error(error_msg)
            
            # Log error with context
            error_data = {
                'frame_shape': frame.shape if frame is not None else None,
                'tracking_status': self.state.is_tracking,
                'overlay_id': self.state.overlay_id
            }
            
            logger.error(error_msg, extra=error_data)
            
            return frame, {"error": error_msg}
    
    async def _apply_overlay(self, frame: np.ndarray, skin_mask: np.ndarray, keypoints: List[Dict[str, Any]]) -> np.ndarray:
        """
        Apply tattoo overlay to the detected skin regions.
        
        Args:
            frame: Input camera frame
            skin_mask: Detected skin mask
            keypoints: Detected body keypoints
            
        Returns:
            Frame with tattoo overlay
        """
        # Skip if no current overlay
        if self.state.current_overlay is None:
            return frame
        
        # Check if target body part is detected
        if self.state.target_body_part is not None:
            target_keypoints = [kp for kp in keypoints if kp.get("part") == self.state.target_body_part]
            if not target_keypoints:
                # Target body part not detected
                return frame
            
            # Filter skin mask to only include target body part
            target_mask = np.zeros_like(skin_mask)
            for kp in target_keypoints:
                if "bounds" in kp:
                    bounds = kp["bounds"]
                    x, y = int(bounds["x"]), int(bounds["y"])
                    w, h = int(bounds["width"]), int(bounds["height"])
                    # Create body part mask
                    cv2.rectangle(target_mask, (x, y), (x + w, y + h), 255, -1)
            
            # Combine with skin mask
            skin_mask = cv2.bitwise_and(skin_mask, target_mask)
        
        # Get the overlay image and apply transformations
        overlay = self.state.current_overlay.copy()
        
        # Apply scale and rotation if specified
        if self.state.overlay_scale != 1.0 or self.state.overlay_rotation != 0.0:
            # Get overlay center
            h, w = overlay.shape[:2]
            center = (w // 2, h // 2)
            
            # Create transformation matrix
            M = cv2.getRotationMatrix2D(center, self.state.overlay_rotation, self.state.overlay_scale)
            
            # Apply transformation
            overlay = cv2.warpAffine(overlay, M, (w, h))
        
        # Apply position offset if specified
        position_offset = self.state.overlay_position
        if position_offset is not None:
            # TODO: Implement position adjustment logic
            # This would require more complex transformations
            pass
        
        # Render the tattoo
        result_frame = await self.tattoo_renderer.render(frame, overlay, skin_mask, keypoints)
        
        return result_frame
    
    @handle_errors()
    async def set_tattoo_overlay(self, image_data: Union[np.ndarray, bytes, str], 
                            overlay_id: Optional[str] = None,
                            target_part: Optional[str] = None) -> bool:
        """
        Set the tattoo overlay image.
        
        Args:
            image_data: Tattoo image data (numpy array, bytes, or file path)
            overlay_id: Optional identifier for the overlay
            target_part: Optional target body part
            
        Returns:
            True if overlay was set successfully, False otherwise
            
        Raises:
            ARError: If setting overlay fails
        """
        try:
            # Convert image data to numpy array
            if isinstance(image_data, str):
                # Load from file path
                overlay = cv2.imread(image_data, cv2.IMREAD_UNCHANGED)
                if overlay is None:
                    raise ARError(f"Failed to load overlay image from {image_data}", ErrorLevel.ERROR)
            elif isinstance(image_data, bytes):
                # Load from bytes
                import numpy as np
                overlay = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_UNCHANGED)
                if overlay is None:
                    raise ARError("Failed to decode overlay image from bytes", ErrorLevel.ERROR)
            else:
                # Assume numpy array
                overlay = image_data
            
            # Add alpha channel if not present
            if overlay.shape[2] == 3:
                b, g, r = cv2.split(overlay)
                alpha = np.ones(b.shape, dtype=b.dtype) * 255
                overlay = cv2.merge((b, g, r, alpha))
            
            # Generate overlay ID if not provided
            if overlay_id is None:
                overlay_id = str(uuid.uuid4())
            
            # Set overlay
            self.state.current_overlay = overlay
            self.state.overlay_id = overlay_id
            self.state.target_body_part = target_part
            
            # Reset transformations
            self.state.overlay_scale = 1.0
            self.state.overlay_rotation = 0.0
            self.state.overlay_position = None
            
            # Set tracking to active
            self.state.is_tracking = True
            
            logger.info(f"Set tattoo overlay: {overlay_id}, size: {overlay.shape}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to set tattoo overlay: {str(e)}"
            logger.error(error_msg)
            raise ARError(error_msg, ErrorLevel.ERROR)
    
    @handle_errors()
    async def generate_and_set_tattoo(self, prompt: str, style: str = "traditional", 
                                target_part: Optional[str] = None) -> Tuple[bool, str]:
        """
        Generate a tattoo from a text description and set as overlay.
        
        Args:
            prompt: Text description of desired tattoo
            style: Tattoo style (traditional, realistic, etc.)
            target_part: Optional target body part
            
        Returns:
            Tuple of (success status, overlay ID or error message)
            
        Raises:
            ARError: If generation fails
        """
        try:
            # Check authentication
            if not self.user_id or not self.user_token:
                raise ARError("User authentication required", ErrorLevel.ERROR)
            
            # Check subscription limits
            if not self.subscription_manager.check_limit(self.user_id, "daily_generations"):
                raise ARError("Daily generation limit reached", ErrorLevel.ERROR)
            
            # Log activity
            logger.info(f"Generating tattoo for user {self.user_id}: {prompt[:30]}...")
            
            # Get subscription tier for quality settings
            tier = self.subscription_manager.get_user_subscription(self.user_id)
            
            # Determine image size based on tier
            if tier == SubscriptionTier.PRO:
                size = (1024, 1024)
            elif tier == SubscriptionTier.PREMIUM:
                size = (768, 768)
            else:
                size = (512, 512)
            
            # Generate the tattoo
            image, metadata = await self.tattoo_generator.generate_tattoo(
                prompt=prompt,
                style=style,
                size=size,
                color=True
            )
            
            # Generate ID for the design
            design_id = str(uuid.uuid4())
            
            # Save the generated image to a temporary file
            temp_path = self.temp_dir / f"{design_id}.png"
            cv2.imwrite(str(temp_path), image)
            
            # Set as current overlay
            success = await self.set_tattoo_overlay(
                image_data=image,
                overlay_id=design_id,
                target_part=target_part
            )
            
            if not success:
                raise ARError("Failed to set generated tattoo as overlay", ErrorLevel.ERROR)
            
            # Cache design details
            self.design_cache[design_id] = {
                "prompt": prompt,
                "style": style,
                "generated_at": datetime.now().isoformat(),
                "metadata": metadata,
                "path": str(temp_path)
            }
            
            # Record generation for subscription tracking
            self.subscription_manager.increment_usage(self.user_id, "daily_generations")
            
            # Record generation for user activity
            if self.session_id:
                self.session_manager.update_session(self.session_id, {
                    "designs_generated": 1
                })
            
            return True, design_id
            
        except ARError as e:
            # Re-raise specific AR errors
            raise
        except Exception as e:
            error_msg = f"Failed to generate tattoo: {str(e)}"
            logger.error(error_msg)
            raise ARError(error_msg, ErrorLevel.ERROR)
    
    @handle_errors()
    async def adjust_overlay(self, scale: Optional[float] = None, 
                        rotation: Optional[float] = None,
                        position: Optional[Dict[str, float]] = None) -> bool:
        """
        Adjust the current tattoo overlay.
        
        Args:
            scale: Scale factor (1.0 = original size)
            rotation: Rotation angle in degrees
            position: Position offset {x, y}
            
        Returns:
            True if adjustment was applied successfully
            
        Raises:
            ARError: If adjustment fails
        """
        if not self.state.current_overlay:
            raise ARError("No active overlay to adjust", ErrorLevel.WARNING)
        
        try:
            # Apply adjustments
            if scale is not None:
                self.state.overlay_scale = max(0.1, min(3.0, scale))
            
            if rotation is not None:
                self.state.overlay_rotation = rotation % 360
            
            if position is not None:
                self.state.overlay_position = position
            
            # Record adjustment for analytics
            if self.session_id:
                self.session_manager.update_session(self.session_id, {
                    "tattoo_adjustments": 1
                })
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to adjust overlay: {str(e)}"
            logger.error(error_msg)
            raise ARError(error_msg, ErrorLevel.ERROR)
    
    @handle_errors()
    async def save_design(self, name: str, description: Optional[str] = None) -> Optional[str]:
        """
        Save the current design to the user's collection.
        
        Args:
            name: Name for the saved design
            description: Optional description
            
        Returns:
            Design ID if saved successfully, None otherwise
            
        Raises:
            ARError: If saving fails
        """
        if not self.state.current_overlay or not self.state.overlay_id:
            raise ARError("No active overlay to save", ErrorLevel.WARNING)
        
        if not self.user_id:
            raise ARError("User authentication required", ErrorLevel.ERROR)
        
        try:
            # Check if design already in cache
            overlay_id = self.state.overlay_id
            design_info = self.design_cache.get(overlay_id, {})
            
            # Generate save path
            save_id = str(uuid.uuid4())
            save_dir = Path(f"data/users/{self.user_id}/designs")
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{save_id}.png"
            
            # Save the current overlay
            cv2.imwrite(str(save_path), self.state.current_overlay)
            
            # Save design metadata
            metadata = {
                "id": save_id,
                "name": name,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "original_id": overlay_id,
                "style": design_info.get("style", "custom"),
                "prompt": design_info.get("prompt", ""),
                "metadata": design_info.get("metadata", {}),
                "path": str(save_path)
            }
            
            metadata_path = save_dir / f"{save_id}.json"
            with open(metadata_path, "w") as f:
                import json
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved design {save_id} for user {self.user_id}")
            
            return save_id
            
        except Exception as e:
            error_msg = f"Failed to save design: {str(e)}"
            logger.error(error_msg)
            raise ARError(error_msg, ErrorLevel.ERROR)
    
    @handle_errors()
    async def load_design(self, design_id: str) -> bool:
        """
        Load a saved design as the current overlay.
        
        Args:
            design_id: ID of the design to load
            
        Returns:
            True if design was loaded successfully
            
        Raises:
            ARError: If loading fails
        """
        if not self.user_id:
            raise ARError("User authentication required", ErrorLevel.ERROR)
        
        try:
            # Check if design is in cache
            if design_id in self.design_cache:
                # Load from cache
                design_path = self.design_cache[design_id].get("path")
                if design_path:
                    return await self.set_tattoo_overlay(
                        image_data=design_path,
                        overlay_id=design_id
                    )
            
            # Check for saved design
            design_path = Path(f"data/users/{self.user_id}/designs/{design_id}.png")
            if design_path.exists():
                return await self.set_tattoo_overlay(
                    image_data=str(design_path),
                    overlay_id=design_id
                )
            
            # Check for marketplace design
            # This would require integration with the marketplace API
            
            raise ARError(f"Design not found: {design_id}", ErrorLevel.WARNING)
            
        except ARError:
            # Re-raise specific AR errors
            raise
        except Exception as e:
            error_msg = f"Failed to load design: {str(e)}"
            logger.error(error_msg)
            raise ARError(error_msg, ErrorLevel.ERROR)
    
    @handle_errors()
    async def clear_overlay(self) -> bool:
        """
        Clear the current tattoo overlay.
        
        Returns:
            True if overlay was cleared
        """
        self.state.current_overlay = None
        self.state.overlay_id = None
        self.state.target_body_part = None
        self.state.overlay_scale = 1.0
        self.state.overlay_rotation = 0.0
        self.state.overlay_position = None
        
        logger.info("Cleared tattoo overlay")
        return True
    
    @handle_errors()
    async def start_tracking(self) -> bool:
        """
        Start body tracking.
        
        Returns:
            True if tracking started successfully
            
        Raises:
            ARError: If starting tracking fails
        """
        if not self.state.is_active:
            raise ARError("AR system not initialized", ErrorLevel.ERROR)
        
        try:
            # Start AR session tracking if not already started
            if self.state.ar_session_start is None:
                self.state.ar_session_start = datetime.now()
            
            # Set tracking state
            self.state.is_tracking = True
            
            # Update session metrics
            if self.session_id:
                self.session_manager.update_session(self.session_id, {
                    "ar_sessions": 1
                })
            
            logger.info("Started body tracking")
            return True
            
        except Exception as e:
            error_msg = f"Failed to start tracking: {str(e)}"
            logger.error(error_msg)
            raise ARError(error_msg, ErrorLevel.ERROR)
    
    @handle_errors()
    async def stop_tracking(self) -> bool:
        """
        Stop body tracking.
        
        Returns:
            True if tracking stopped successfully
        """
        # Update AR session duration
        if self.state.ar_session_start is not None:
            session_duration = (datetime.now() - self.state.ar_session_start).total_seconds()
            self.state.ar_session_start = None
            
            # Update session metrics
            if self.session_id:
                self.session_manager.update_session(self.session_id, {
                    "ar_duration": session_duration
                })
        
        # Set tracking state
        self.state.is_tracking = False
        
        logger.info("Stopped body tracking")
        return True
    
    def _update_performance_settings(self, battery_level: float, temperature: float) -> None:
        """
        Update AR settings based on device performance.
        
        Args:
            battery_level: Current battery level percentage
            temperature: Current device temperature in Celsius
        """
        # Update settings based on battery and temperature
        self.config.adjust_quality_for_performance(battery_level, temperature, self.state.fps)
    
    @handle_errors()
    async def set_user(self, user_id: str, token: str) -> bool:
        """
        Set the current user for the AR session.
        
        Args:
            user_id: User ID
            token: Authentication token
            
        Returns:
            True if user was authenticated successfully
            
        Raises:
            ARError: If authentication fails
        """
        try:
            # Verify token with auth manager
            if not self.auth_manager.verify_token(token):
                raise ARError("Invalid authentication token", ErrorLevel.ERROR)
            
            # Set user information
            self.user_id = user_id
            self.user_token = token
            
            # Create session if not exists
            if not self.session_id:
                # Get device info
                device_info = {
                    "capabilities": [cap.value for cap in self.state.device_capabilities],
                    "platform": "unknown",  # Would be set based on platform
                    "app_version": "1.0.0"  # Would be set based on app version
                }
                
                # Create session
                self.session_id = self.session_manager.create_session(user_id, device_info)
            
            logger.info(f"User set to {user_id}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to set user: {str(e)}"
            logger.error(error_msg)
            raise ARError(error_msg, ErrorLevel.ERROR)
    
    @handle_errors()
    async def get_state(self) -> Dict[str, Any]:
        """
        Get the current AR state.
        
        Returns:
            Dictionary with AR state information
        """
        # Create state information dictionary
        state_info = {
            "is_active": self.state.is_active,
            "is_tracking": self.state.is_tracking,
            "fps": self.state.fps,
            "frame_count": self.state.frame_count,
            "has_overlay": self.state.current_overlay is not None,
            "overlay_id": self.state.overlay_id,
            "detected_body_parts": self.state.detected_body_parts,
            "skin_confidence": self.state.skin_confidence,
            "device_capabilities": [cap.value for cap in self.state.device_capabilities],
            "last_activity": self.state.last_activity.isoformat() if self.state.last_activity else None,
            "target_body_part": self.state.target_body_part,
            "overlay_scale": self.state.overlay_scale,
            "overlay_rotation": self.state.overlay_rotation,
            "authenticated": self.user_id is not None and self.user_token is not None,
            "user_id": self.user_id
        }
        
        # Add subscription info if authenticated
        if self.user_id:
            tier = self.subscription_manager.get_user_subscription(self.user_id)
            features = self.subscription_manager.get_features(self.user_id)
            
            state_info["subscription"] = {
                "tier": tier.value,
                "has_nerf_avatar": features.nerf_avatar,
                "has_advanced_styles": features.advanced_styles,
                "max_resolution": features.max_resolution
            }
        
        return state_info
    
    @handle_errors()
    async def capture_screenshot(self) -> Optional[np.ndarray]:
        """
        Capture the current AR view.
        
        Returns:
            Screenshot image or None if capture fails
            
        Raises:
            ARError: If capture fails
        """
        # Check if preview mode is active or camera is available
        if not self.state.preview_mode or self.camera is None:
            raise ARError("No active camera to capture screenshot", ErrorLevel.WARNING)
        
        try:
            # Capture frame
            ret, frame = self.camera.read()
            
            if not ret:
                raise ARError("Failed to capture frame from camera", ErrorLevel.ERROR)
            
            # Process frame with AR effects
            processed_frame, _ = await self.process_frame(frame)
            
            # Increment counter if in session
            if self.session_id:
                self.session_manager.update_session(self.session_id, {
                    "total_exports": 1
                })
            
            return processed_frame
            
        except Exception as e:
            error_msg = f"Failed to capture screenshot: {str(e)}"
            logger.error(error_msg)
            raise ARError(error_msg, ErrorLevel.ERROR)
    
    @handle_errors()
    async def save_screenshot(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Save the current AR view as an image.
        
        Args:
            filename: Optional filename to save as
            
        Returns:
            Path to saved screenshot or None if save fails
            
        Raises:
            ARError: If saving fails
        """
        try:
            # Capture screenshot
            screenshot = await self.capture_screenshot()
            
            if screenshot is None:
                raise ARError("Failed to capture screenshot", ErrorLevel.ERROR)
            
            # Generate filename if not provided
            if filename is None:
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            # Ensure screenshots directory exists
            screenshots_dir = Path("screenshots")
            screenshots_dir.mkdir(exist_ok=True)
            
            # Create save path
            save_path = screenshots_dir / filename
            
            # Save image
            cv2.imwrite(str(save_path), screenshot)
            
            logger.info(f"Saved screenshot to {save_path}")
            return str(save_path)
            
        except Exception as e:
            error_msg = f"Failed to save screenshot: {str(e)}"
            logger.error(error_msg)
            raise ARError(error_msg, ErrorLevel.ERROR)
    
    @handle_errors()
    async def get_camera_frame(self) -> Optional[np.ndarray]:
        """
        Get a raw frame from the camera.
        
        Returns:
            Camera frame or None if capture fails
            
        Raises:
            ARError: If camera is not available
        """
        if self.camera is None or not self.camera.isOpened():
            raise ARError("Camera not available", ErrorLevel.ERROR)
        
        try:
            # Capture frame
            ret, frame = self.camera.read()
            
            if not ret:
                raise ARError("Failed to capture frame from camera", ErrorLevel.ERROR)
            
            return frame
            
        except Exception as e:
            error_msg = f"Failed to get camera frame: {str(e)}"
            logger.error(error_msg)
            raise ARError(error_msg, ErrorLevel.ERROR)
    
    @handle_errors()
    async def toggle_camera(self) -> bool:
        """
        Toggle between front and back camera.
        
        Returns:
            True if camera was toggled successfully
            
        Raises:
            ARError: If toggling fails
        """
        try:
            # Release current camera
            if self.camera is not None:
                self.camera.release()
            
            # Toggle camera ID
            self.camera_id = 1 if self.camera_id == 0 else 0
            
            # Initialize new camera
            success = self._initialize_camera()
            
            if not success:
                # Revert to previous camera if failed
                self.camera_id = 1 if self.camera_id == 0 else 0
                self._initialize_camera()
                raise ARError("Failed to switch camera", ErrorLevel.WARNING)
            
            logger.info(f"Switched to camera {self.camera_id}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to toggle camera: {str(e)}"
            logger.error(error_msg)
            raise ARError(error_msg, ErrorLevel.ERROR)
    
    @handle_errors()
    async def set_preview_mode(self, enabled: bool) -> bool:
        """
        Enable or disable preview mode.
        
        Args:
            enabled: Whether preview mode is enabled
            
        Returns:
            True if mode was set successfully
        """
        # Set preview mode
        self.state.preview_mode = enabled
        
        # Initialize or release camera as needed
        if enabled and (self.camera is None or not self.camera.isOpened()):
            self._initialize_camera()
        elif not enabled and self.camera is not None:
            self.camera.release()
            self.camera = None
        
        logger.info(f"Preview mode {'enabled' if enabled else 'disabled'}")
        return True
    
    @handle_errors()
    async def generate_nerf_avatar(self) -> bool:
        """
        Generate a NeRF Metahuman Avatar for more advanced AR visualization.
        
        Returns:
            True if avatar generation was initiated successfully
            
        Raises:
            ARError: If generation fails or is not available
        """
        if not self.user_id:
            raise ARError("User authentication required", ErrorLevel.ERROR)
        
        # Check if user has NeRF avatar feature
        if not self.subscription_manager.has_feature(self.user_id, "nerf_avatar"):
            raise ARError("NeRF avatar feature requires premium subscription", ErrorLevel.ERROR)
        
        try:
            # This would integrate with the NeRF avatar generation system
            # For now, it's a placeholder
            
            logger.info(f"NeRF avatar generation initiated for user {self.user_id}")
            
            # Return success (actual implementation would be asynchronous)
            return True
            
        except Exception as e:
            error_msg = f"Failed to generate NeRF avatar: {str(e)}"
            logger.error(error_msg)
            raise ARError(error_msg, ErrorLevel.ERROR)
    
    @handle_errors()
    async def cleanup(self) -> None:
        """
        Clean up resources used by the AR visualizer.
        
        This method should be called when the application is shutting down.
        """
        try:
            # Release camera
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            
            # End session if active
            if self.session_id:
                self.session_manager.end_session(self.session_id)
                self.session_id = None
            
            # Unload models
            if self.model_handler:
                await self.model_handler.unload_all_models()
            
            # Clean up temporary files
            # This would delete temporary files in a production implementation
            
            logger.info("AR visualizer resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


# Create a singleton instance
_ar_visualizer = None

def get_ar_visualizer(config: Optional[ARConfig] = None) -> ARVisualizer:
    """
    Get the singleton AR visualizer instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        ARVisualizer: The AR visualizer instance
    """
    global _ar_visualizer
    if _ar_visualizer is None:
        _ar_visualizer = ARVisualizer(config)
    return _ar_visualizer