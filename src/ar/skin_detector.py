"""
Skin detection module for the Th.ink AR application.

This module provides functionality to detect skin regions in images
using various computer vision techniques, with support for different skin tones,
lighting conditions, and body parts.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum

from ..utils.performance_monitor import measure_time
from ..errors.error_handler import handle_errors, CameraError

# Configure logger
logger = logging.getLogger("think.ar.skin")


class SkinColorModel(Enum):
    """Color models for skin detection."""
    
    RGB = "rgb"
    HSV = "hsv"
    YCrCb = "ycrcb"
    LAB = "lab"
    ADAPTIVE = "adaptive"


@dataclass
class SkinThresholds:
    """Threshold values for skin detection in different color spaces."""
    
    # HSV thresholds
    hsv_lower: np.ndarray = np.array([0, 10, 60], dtype=np.uint8)
    hsv_upper: np.ndarray = np.array([25, 150, 255], dtype=np.uint8)
    
    # YCrCb thresholds
    ycrcb_lower: np.ndarray = np.array([0, 135, 85], dtype=np.uint8)
    ycrcb_upper: np.ndarray = np.array([255, 180, 135], dtype=np.uint8)
    
    # RGB thresholds (normalized)
    rgb_r_min: float = 0.4
    rgb_ratio: float = 1.185
    rgb_diff: float = 0.2
    
    # LAB thresholds
    lab_lower: np.ndarray = np.array([20, 130, 110], dtype=np.uint8)
    lab_upper: np.ndarray = np.array([220, 175, 155], dtype=np.uint8)


class SkinDetector:
    """
    Skin detection system for AR tattoo visualization.
    
    This class provides methods to detect and segment skin regions in images
    using various computer vision techniques, with adaptive handling of different
    skin tones and lighting conditions.
    """
    
    def __init__(self, 
                threshold: float = 0.85, 
                color_model: SkinColorModel = SkinColorModel.ADAPTIVE,
                thresholds: Optional[SkinThresholds] = None,
                use_face_landmarks: bool = True,
                adapt_to_lighting: bool = True):
        """
        Initialize the skin detector.
        
        Args:
            threshold: Confidence threshold for skin detection (0.0 to 1.0)
            color_model: Color model to use for skin detection
            thresholds: Custom threshold values for different color spaces
            use_face_landmarks: Whether to use face landmarks to refine skin detection
            adapt_to_lighting: Whether to adapt to lighting conditions
        """
        self.threshold = threshold
        self.color_model = color_model
        self.thresholds = thresholds if thresholds is not None else SkinThresholds()
        self.use_face_landmarks = use_face_landmarks
        self.adapt_to_lighting = adapt_to_lighting
        
        # For adaptive model
        self.skin_samples: List[np.ndarray] = []
        self.lighting_samples: List[float] = []
        self.max_samples = 100  # Maximum number of samples to store
        
        # For statistics
        self.detection_stats = {
            "frames_processed": 0,
            "avg_skin_percentage": 0.0,
            "avg_confidence": 0.0
        }
        
        logger.info(f"Skin detector initialized with {color_model.value} color model")
    
    @measure_time("skin_detection")
    @handle_errors()
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect skin regions in the frame.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Tuple of (skin mask, confidence score)
            
        Raises:
            CameraError: If frame processing fails
        """
        if frame is None or frame.size == 0:
            raise CameraError("Empty frame provided to skin detector")
        
        # Track statistics
        self.detection_stats["frames_processed"] += 1
        
        # Select detection method based on color model
        if self.color_model == SkinColorModel.HSV:
            mask, confidence = self._detect_hsv(frame)
        elif self.color_model == SkinColorModel.YCrCb:
            mask, confidence = self._detect_ycrcb(frame)
        elif self.color_model == SkinColorModel.RGB:
            mask, confidence = self._detect_rgb(frame)
        elif self.color_model == SkinColorModel.LAB:
            mask, confidence = self._detect_lab(frame)
        elif self.color_model == SkinColorModel.ADAPTIVE:
            mask, confidence = self._detect_adaptive(frame)
        else:
            # Default to HSV if unknown model
            mask, confidence = self._detect_hsv(frame)
        
        # Refine mask with morphological operations
        mask = self._refine_mask(mask)
        
        # Update statistics
        self._update_statistics(mask, confidence)
        
        return mask, confidence
    
    def _detect_hsv(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect skin using HSV color space.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Tuple of (skin mask, confidence score)
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Adjust thresholds based on lighting if enabled
        lower_threshold = self.thresholds.hsv_lower
        upper_threshold = self.thresholds.hsv_upper
        
        if self.adapt_to_lighting:
            # Analyze frame brightness
            brightness = np.mean(hsv[:, :, 2])
            
            # Adjust value (brightness) thresholds
            brightness_factor = brightness / 128.0  # Normalize to 1.0 at middle brightness
            
            adjusted_lower = lower_threshold.copy()
            adjusted_upper = upper_threshold.copy()
            
            # Adjust value (V) component
            if brightness_factor < 1.0:
                # Darker image: lower the min value threshold
                adjusted_lower[2] = max(20, int(lower_threshold[2] * brightness_factor))
            else:
                # Brighter image: raise the max value threshold
                adjusted_upper[2] = min(255, int(upper_threshold[2] * brightness_factor))
            
            lower_threshold = adjusted_lower
            upper_threshold = adjusted_upper
        
        # Create mask for skin color range
        mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
        
        # Calculate confidence score
        confidence = np.sum(mask > 0) / mask.size
        
        return mask, confidence
    
    def _detect_ycrcb(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect skin using YCrCb color space.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Tuple of (skin mask, confidence score)
        """
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Create mask for skin color range
        mask = cv2.inRange(ycrcb, self.thresholds.ycrcb_lower, self.thresholds.ycrcb_upper)
        
        # Calculate confidence score
        confidence = np.sum(mask > 0) / mask.size
        
        return mask, confidence
    
    def _detect_rgb(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect skin using RGB color space with normalized RGB.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Tuple of (skin mask, confidence score)
        """
        # BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to float and normalize
        rgb_float = rgb.astype(np.float32) / 255.0
        
        # Avoid division by zero
        rgb_sum = np.sum(rgb_float, axis=2)
        rgb_sum = np.maximum(rgb_sum, 0.00001)
        
        # Normalized RGB
        r_normalized = rgb_float[:, :, 0] / rgb_sum
        g_normalized = rgb_float[:, :, 1] / rgb_sum
        b_normalized = rgb_float[:, :, 2] / rgb_sum
        
        # Create mask using normalized RGB rules
        # R > 0.4, R > G*ratio, R > B*ratio, |G-B| < diff
        mask1 = r_normalized > self.thresholds.rgb_r_min
        mask2 = r_normalized > (g_normalized * self.thresholds.rgb_ratio)
        mask3 = r_normalized > (b_normalized * self.thresholds.rgb_ratio)
        mask4 = np.abs(g_normalized - b_normalized) < self.thresholds.rgb_diff
        
        # Combine masks
        mask = np.logical_and.reduce((mask1, mask2, mask3, mask4))
        mask = (mask * 255).astype(np.uint8)
        
        # Calculate confidence score
        confidence = np.sum(mask > 0) / mask.size
        
        return mask, confidence
    
    def _detect_lab(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect skin using LAB color space.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Tuple of (skin mask, confidence score)
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Create mask for skin color range
        mask = cv2.inRange(lab, self.thresholds.lab_lower, self.thresholds.lab_upper)
        
        # Calculate confidence score
        confidence = np.sum(mask > 0) / mask.size
        
        return mask, confidence
    
    def _detect_adaptive(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect skin using an adaptive approach combining multiple color spaces.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Tuple of (skin mask, confidence score)
        """
        # Get masks from different color spaces
        hsv_mask, hsv_conf = self._detect_hsv(frame)
        ycrcb_mask, ycrcb_conf = self._detect_ycrcb(frame)
        
        # Combine masks with weighted average based on confidence
        if hsv_conf > ycrcb_conf:
            # If HSV is more confident, give it more weight
            weight_hsv = 0.7
            weight_ycrcb = 0.3
        else:
            # If YCrCb is more confident, give it more weight
            weight_hsv = 0.3
            weight_ycrcb = 0.7
        
        # Create combined mask
        combined_mask = cv2.addWeighted(hsv_mask, weight_hsv, ycrcb_mask, weight_ycrcb, 0)
        
        # Apply threshold to combined mask
        _, mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Calculate overall confidence (weighted average of individual confidences)
        confidence = (hsv_conf * weight_hsv) + (ycrcb_conf * weight_ycrcb)
        
        return mask, confidence
    
    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Refine skin mask with morphological operations.
        
        Args:
            mask: Initial skin mask
            
        Returns:
            Refined skin mask
        """
        # Create kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Apply morphological operations to refine mask
        # First, remove small noise with erosion
        mask = cv2.erode(mask, kernel, iterations=1)
        
        # Then, fill holes and connect regions with dilation
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Final erosion to sharpen edges
        mask = cv2.erode(mask, kernel, iterations=1)
        
        return mask
    
    @handle_errors()
    def refine_mask(self, mask: np.ndarray, keypoints: List[Dict[str, Any]]) -> np.ndarray:
        """
        Refine skin mask using detected keypoints.
        
        Args:
            mask: Skin detection mask
            keypoints: List of detected keypoints from body tracking
            
        Returns:
            Refined skin mask
        """
        if not keypoints:
            return mask
        
        refined_mask = mask.copy()
        
        # Create a mask from body keypoints
        body_mask = np.zeros_like(mask)
        
        # Draw body parts based on keypoints
        for keypoint in keypoints:
            if keypoint.get("part") in ["face", "arm", "hand", "leg", "torso"] and "position" in keypoint:
                # Get position and confidence
                pos = keypoint["position"]
                conf = keypoint.get("confidence", 0.0)
                
                # Only use high-confidence keypoints
                if conf >= self.threshold:
                    # Draw body part region on body mask
                    if keypoint.get("part") == "face" and "bounds" in keypoint:
                        # For face, use bounding box
                        bounds = keypoint["bounds"]
                        cv2.rectangle(body_mask, 
                                     (int(bounds["x"]), int(bounds["y"])),
                                     (int(bounds["x"] + bounds["width"]), int(bounds["y"] + bounds["height"])),
                                     255, -1)
                    else:
                        # For other body parts, draw circle at keypoint
                        cv2.circle(body_mask, 
                                  (int(pos["x"]), int(pos["y"])),
                                  25,  # Radius
                                  255, -1)
        
        # Dilate body mask to cover surrounding areas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        body_mask = cv2.dilate(body_mask, kernel, iterations=3)
        
        # Combine masks: only keep skin detections that overlap with body parts
        refined_mask = cv2.bitwise_and(refined_mask, body_mask)
        
        return refined_mask
    
    def update_thresholds(self, samples: List[np.ndarray]) -> None:
        """
        Update skin detection thresholds based on skin samples.
        
        Args:
            samples: List of skin sample images (BGR format)
        """
        if not samples:
            return
        
        # Analyze samples to adjust thresholds
        hsv_samples = []
        ycrcb_samples = []
        
        for sample in samples:
            hsv_samples.append(cv2.cvtColor(sample, cv2.COLOR_BGR2HSV))
            ycrcb_samples.append(cv2.cvtColor(sample, cv2.COLOR_BGR2YCrCb))
        
        # Update HSV thresholds
        if hsv_samples:
            hsv_values = np.vstack([sample.reshape(-1, 3) for sample in hsv_samples])
            h_mean, s_mean, v_mean = np.mean(hsv_values, axis=0)
            h_std, s_std, v_std = np.std(hsv_values, axis=0)
            
            # Update lower and upper bounds with 2 standard deviations
            self.thresholds.hsv_lower = np.array([
                max(0, h_mean - 2 * h_std),
                max(0, s_mean - 2 * s_std),
                max(0, v_mean - 2 * v_std)
            ], dtype=np.uint8)
            
            self.thresholds.hsv_upper = np.array([
                min(179, h_mean + 2 * h_std),
                min(255, s_mean + 2 * s_std),
                min(255, v_mean + 2 * v_std)
            ], dtype=np.uint8)
        
        # Update YCrCb thresholds
        if ycrcb_samples:
            ycrcb_values = np.vstack([sample.reshape(-1, 3) for sample in ycrcb_samples])
            y_mean, cr_mean, cb_mean = np.mean(ycrcb_values, axis=0)
            y_std, cr_std, cb_std = np.std(ycrcb_values, axis=0)
            
            # Update lower and upper bounds with 2 standard deviations
            self.thresholds.ycrcb_lower = np.array([
                max(0, y_mean - 2 * y_std),
                max(0, cr_mean - 2 * cr_std),
                max(0, cb_mean - 2 * cb_std)
            ], dtype=np.uint8)
            
            self.thresholds.ycrcb_upper = np.array([
                min(255, y_mean + 2 * y_std),
                min(255, cr_mean + 2 * cr_std),
                min(255, cb_mean + 2 * cb_std)
            ], dtype=np.uint8)
        
        logger.info("Skin detection thresholds updated based on samples")
    
    def add_skin_sample(self, sample: np.ndarray) -> None:
        """
        Add a skin sample for adaptive threshold adjustment.
        
        Args:
            sample: Skin sample image (BGR format)
        """
        # Only store a limited number of samples
        if len(self.skin_samples) >= self.max_samples:
            # Remove oldest sample
            self.skin_samples.pop(0)
        
        self.skin_samples.append(sample)
        
        # If we have enough samples, update thresholds
        if len(self.skin_samples) >= 5:
            self.update_thresholds(self.skin_samples)
    
    def add_lighting_sample(self, brightness: float) -> None:
        """
        Add a lighting sample for adaptive threshold adjustment.
        
        Args:
            brightness: Average brightness value
        """
        # Only store a limited number of samples
        if len(self.lighting_samples) >= self.max_samples:
            # Remove oldest sample
            self.lighting_samples.pop(0)
        
        self.lighting_samples.append(brightness)
    
    def _update_statistics(self, mask: np.ndarray, confidence: float) -> None:
        """
        Update detection statistics.
        
        Args:
            mask: Skin detection mask
            confidence: Detection confidence
        """
        # Calculate skin percentage
        skin_percentage = np.sum(mask > 0) / mask.size * 100
        
        # Update running averages
        frames = self.detection_stats["frames_processed"]
        
        if frames == 1:
            # First frame, just set the values
            self.detection_stats["avg_skin_percentage"] = skin_percentage
            self.detection_stats["avg_confidence"] = confidence
        else:
            # Update running averages
            alpha = 0.05  # Weight for new value in running average
            self.detection_stats["avg_skin_percentage"] = (
                (1 - alpha) * self.detection_stats["avg_skin_percentage"] + 
                alpha * skin_percentage
            )
            self.detection_stats["avg_confidence"] = (
                (1 - alpha) * self.detection_stats["avg_confidence"] + 
                alpha * confidence
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current skin detection statistics.
        
        Returns:
            Dictionary with detection statistics
        """
        return self.detection_stats
    
    def set_color_model(self, model: SkinColorModel) -> None:
        """
        Set the color model for skin detection.
        
        Args:
            model: Color model to use
        """
        self.color_model = model
        logger.info(f"Skin detection color model set to {model.value}")
    
    def set_threshold(self, threshold: float) -> None:
        """
        Set the confidence threshold for skin detection.
        
        Args:
            threshold: Confidence threshold (0.0 to 1.0)
        """
        self.threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Skin detection threshold set to {self.threshold}")


# Factory function for creating skin detector with different configurations
def create_skin_detector(
    config_type: str = "standard",
    threshold: float = 0.85,
    color_model: str = "adaptive"
) -> SkinDetector:
    """
    Create a skin detector with specific configuration.
    
    Args:
        config_type: Configuration type ("standard", "high_quality", "performance")
        threshold: Confidence threshold (0.0 to 1.0)
        color_model: Color model name ("hsv", "ycrcb", "rgb", "lab", "adaptive")
        
    Returns:
        Configured skin detector instance
    """
    # Convert color model string to enum
    try:
        model = SkinColorModel(color_model.lower())
    except ValueError:
        logger.warning(f"Invalid color model: {color_model}, using adaptive")
        model = SkinColorModel.ADAPTIVE
    
    # Create detector with basic settings
    detector = SkinDetector(threshold=threshold, color_model=model)
    
    # Apply configuration based on type
    if config_type == "high_quality":
        # High quality settings prioritize accuracy
        detector.use_face_landmarks = True
        detector.adapt_to_lighting = True
        # Custom thresholds for better accuracy
        detector.thresholds = SkinThresholds(
            hsv_lower=np.array([0, 15, 50], dtype=np.uint8),
            hsv_upper=np.array([30, 170, 255], dtype=np.uint8),
            ycrcb_lower=np.array([0, 130, 80], dtype=np.uint8),
            ycrcb_upper=np.array([255, 185, 140], dtype=np.uint8)
        )
    elif config_type == "performance":
        # Performance settings prioritize speed
        detector.use_face_landmarks = False
        detector.adapt_to_lighting = False
        detector.color_model = SkinColorModel.HSV  # HSV is faster than adaptive
    
    return detector