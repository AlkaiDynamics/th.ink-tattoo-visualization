"""
Tattoo rendering module for the Th.ink AR application.

This module provides functionality to render tattoo designs onto detected skin
regions with realistic lighting, perspective, and blending effects.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from enum import Enum
import math

from ..utils.performance_monitor import measure_time
from ..errors.error_handler import handle_errors, RenderingError

# Configure logger
logger = logging.getLogger("think.ar.renderer")


class BlendMode(Enum):
    """Blending modes for tattoo rendering."""
    
    NORMAL = "normal"
    MULTIPLY = "multiply"
    OVERLAY = "overlay"
    SCREEN = "screen"
    SOFT_LIGHT = "soft_light"
    COLOR = "color"
    LUMINOSITY = "luminosity"


class TattooStyle(Enum):
    """Tattoo design styles."""
    
    TRADITIONAL = "traditional"
    REALISTIC = "realistic"
    BLACKWORK = "blackwork"
    WATERCOLOR = "watercolor"
    TRIBAL = "tribal"
    JAPANESE = "japanese"
    NEW_SCHOOL = "new_school"
    MINIMALIST = "minimalist"


class TattooRenderer:
    """
    Tattoo rendering system for AR visualization.
    
    This class provides methods to render tattoo designs onto detected skin regions
    with realistic lighting, perspective, and blending effects.
    """
    
    def __init__(self, 
                opacity: float = 0.9, 
                blend_mode: BlendMode = BlendMode.MULTIPLY,
                adapt_to_lighting: bool = True,
                high_quality: bool = True):
        """
        Initialize the tattoo renderer.
        
        Args:
            opacity: Opacity of rendered tattoo (0.0 to 1.0)
            blend_mode: Blending mode for tattoo rendering
            adapt_to_lighting: Whether to adapt to lighting conditions
            high_quality: Whether to use high quality rendering
        """
        self.opacity = max(0.0, min(1.0, opacity))
        self.blend_mode = blend_mode
        self.adapt_to_lighting = adapt_to_lighting
        self.high_quality = high_quality
        self.current_design = None
        self.perspective_matrix = None
        self.last_keypoints = None
        self.design_metrics = {
            "width": 0,
            "height": 0,
            "channels": 0,
            "has_alpha": False
        }
        self.shadow_strength = 0.3
        self.highlight_strength = 0.2
        self.texture_strength = 0.15
        
        logger.info(f"Tattoo renderer initialized with {blend_mode.value} blend mode")
    
    @measure_time("set_design")
    @handle_errors()
    def set_design(self, design: np.ndarray) -> bool:
        """
        Set the tattoo design to be rendered.
        
        Args:
            design: Tattoo design image (BGRA or BGR format)
            
        Returns:
            True if design was set successfully, False otherwise
            
        Raises:
            RenderingError: If design processing fails
        """
        if design is None or design.size == 0:
            logger.warning("Attempted to set empty design")
            return False
        
        try:
            # Store original design
            self.current_design = design.copy()
            
            # Store design metrics
            self.design_metrics = {
                "width": design.shape[1],
                "height": design.shape[0],
                "channels": design.shape[2] if len(design.shape) > 2 else 1,
                "has_alpha": design.shape[2] == 4 if len(design.shape) > 2 else False
            }
            
            # Create alpha channel if not present
            if not self.design_metrics["has_alpha"]:
                logger.info("Adding alpha channel to design")
                # Create alpha channel (white areas transparent, black areas opaque)
                if self.design_metrics["channels"] == 3:
                    # Convert to grayscale for alpha calculation
                    gray = cv2.cvtColor(design, cv2.COLOR_BGR2GRAY)
                    # Invert so black becomes transparent, white becomes opaque
                    alpha = cv2.bitwise_not(gray)
                    # Create BGRA image
                    self.current_design = cv2.cvtColor(design, cv2.COLOR_BGR2BGRA)
                    # Set alpha channel
                    self.current_design[:, :, 3] = alpha
                    self.design_metrics["has_alpha"] = True
                    self.design_metrics["channels"] = 4
            
            logger.info(f"Design set: {self.design_metrics['width']}x{self.design_metrics['height']} "
                      f"with {self.design_metrics['channels']} channels")
            
            # Reset perspective matrix
            self.perspective_matrix = None
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set design: {e}")
            raise RenderingError(f"Failed to process tattoo design: {str(e)}")
    
    @measure_time("render")
    @handle_errors()
    def render(self, frame: np.ndarray, mask: np.ndarray, keypoints: List[Dict]) -> np.ndarray:
        """
        Render tattoo design onto the frame based on mask and keypoints.
        
        Args:
            frame: Input image frame (BGR format)
            mask: Skin detection mask
            keypoints: List of detected keypoints from body tracking
            
        Returns:
            Frame with rendered tattoo
            
        Raises:
            RenderingError: If rendering fails
        """
        if self.current_design is None:
            return frame
        
        if mask is None or np.sum(mask > 0) == 0:
            return frame
        
        try:
            # Calculate perspective transform if keypoints available
            if keypoints and keypoints != self.last_keypoints:
                self.perspective_matrix = self._calculate_perspective(keypoints)
                self.last_keypoints = keypoints
            
            # Transform design according to perspective
            warped_design = self._warp_design(self.current_design)
            
            # Apply lighting adjustment
            adjusted_design = self._adjust_lighting(warped_design, frame, mask)
            
            # Blend design with frame
            result = self._blend_overlay(frame, adjusted_design, mask)
            
            return result
            
        except Exception as e:
            logger.error(f"Rendering error: {e}")
            raise RenderingError(f"Failed to render tattoo: {str(e)}")
    
    def _calculate_perspective(self, keypoints: List[Dict]) -> np.ndarray:
        """
        Calculate perspective transform matrix from keypoints.
        
        Args:
            keypoints: List of detected keypoints from body tracking
            
        Returns:
            Perspective transform matrix
        """
        # Default to identity matrix (no transformation)
        perspective_matrix = np.eye(3, dtype=np.float32)
        
        # Extract relevant keypoints for the body part
        target_parts = ["torso", "left_arm", "right_arm", "left_leg", "right_leg"]
        relevant_keypoints = [kp for kp in keypoints if kp.get("part") in target_parts]
        
        if not relevant_keypoints:
            return perspective_matrix
        
        # Find the body part with highest confidence
        best_keypoint = max(relevant_keypoints, key=lambda kp: kp.get("confidence", 0))
        bounds = best_keypoint.get("bounds")
        
        if not bounds:
            return perspective_matrix
        
        # Get design size
        design_width = self.design_metrics["width"]
        design_height = self.design_metrics["height"]
        
        # Define source points (corners of design)
        src_points = np.array([
            [0, 0],  # Top-left
            [design_width, 0],  # Top-right
            [design_width, design_height],  # Bottom-right
            [0, design_height]  # Bottom-left
        ], dtype=np.float32)
        
        # Define target points based on body part bounds
        x, y = bounds.get("x", 0), bounds.get("y", 0)
        w, h = bounds.get("width", 0), bounds.get("height", 0)
        
        # Scale design to fit target area
        scale_factor = min(w / design_width, h / design_height)
        target_width = design_width * scale_factor
        target_height = design_height * scale_factor
        
        # Center design on target area
        x_center = x + w/2
        y_center = y + h/2
        x_start = x_center - target_width/2
        y_start = y_center - target_height/2
        
        # Add perspective effect based on body part
        part = best_keypoint.get("part")
        if part in ["left_arm", "right_arm"]:
            # Curve design to follow arm
            curve_factor = 0.15
            dst_points = np.array([
                [x_start, y_start],  # Top-left
                [x_start + target_width, y_start - target_height * curve_factor],  # Top-right
                [x_start + target_width, y_start + target_height + target_height * curve_factor],  # Bottom-right
                [x_start, y_start + target_height]  # Bottom-left
            ], dtype=np.float32)
        elif part in ["left_leg", "right_leg"]:
            # Taper design to follow leg
            taper_factor = 0.1
            dst_points = np.array([
                [x_start - target_width * taper_factor, y_start],  # Top-left
                [x_start + target_width + target_width * taper_factor, y_start],  # Top-right
                [x_start + target_width, y_start + target_height],  # Bottom-right
                [x_start, y_start + target_height]  # Bottom-left
            ], dtype=np.float32)
        else:
            # Standard rectangle
            dst_points = np.array([
                [x_start, y_start],  # Top-left
                [x_start + target_width, y_start],  # Top-right
                [x_start + target_width, y_start + target_height],  # Bottom-right
                [x_start, y_start + target_height]  # Bottom-left
            ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        return perspective_matrix
    
    def _warp_design(self, design: np.ndarray) -> np.ndarray:
        """
        Apply perspective warp to design.
        
        Args:
            design: Tattoo design image
            
        Returns:
            Warped design image
        """
        if self.perspective_matrix is None:
            return design
        
        # Get frame size from design or perspective matrix
        if hasattr(self, "frame_size"):
            frame_size = self.frame_size
        else:
            # Estimate output size based on design size and perspective matrix
            h, w = design.shape[:2]
            frame_size = (w*2, h*2)  # Double size as a safe estimate
        
        # Apply perspective transformation
        warped = cv2.warpPerspective(
            design,
            self.perspective_matrix,
            (frame_size[0], frame_size[1]),
            flags=cv2.INTER_LANCZOS4
        )
        
        return warped
    
    def _adjust_lighting(self, design: np.ndarray, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Adjust design lighting to match frame.
        
        Args:
            design: Tattoo design image
            frame: Input image frame
            mask: Skin detection mask
            
        Returns:
            Design with adjusted lighting
        """
        if not self.adapt_to_lighting:
            return design
        
        # Extract lighting information from frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masked_frame = cv2.bitwise_and(gray_frame, mask)
        
        # Calculate average lighting in masked region
        mask_pixels = mask > 0
        if np.any(mask_pixels):
            mean_lighting = np.mean(masked_frame[mask_pixels])
            
            # Calculate lighting variation (for shadows and highlights)
            std_lighting = np.std(masked_frame[mask_pixels])
        else:
            mean_lighting = 128  # Default to middle gray
            std_lighting = 30    # Default standard deviation
        
        # Normalize mean lighting to 0-1 range
        normalized_lighting = mean_lighting / 255.0
        
        # Apply lighting adjustment to design
        adjusted = design.copy()
        
        # Create lighting adjustment layer (darker or brighter)
        lighting_factor = normalized_lighting * 2.0  # Adjust to 0-2 range
        
        # Split design into color channels and alpha
        if design.shape[2] == 4:  # BGRA
            b, g, r, a = cv2.split(adjusted)
            
            # Apply global lighting
            b = cv2.multiply(b, lighting_factor)
            g = cv2.multiply(g, lighting_factor)
            r = cv2.multiply(r, lighting_factor)
            
            # Add skin texture variation if high quality is enabled
            if self.high_quality:
                # Create texture variation map
                texture_map = self._create_skin_texture_map(frame.shape[:2], std_lighting)
                
                # Apply texture variation
                texture_strength = self.texture_strength
                b = cv2.addWeighted(b, 1.0, texture_map, texture_strength, 0)
                g = cv2.addWeighted(g, 1.0, texture_map, texture_strength, 0)
                r = cv2.addWeighted(r, 1.0, texture_map, texture_strength, 0)
            
            # Merge channels back
            adjusted = cv2.merge([b, g, r, a])
        
        return adjusted
    
    def _create_skin_texture_map(self, shape: Tuple[int, int], std_lighting: float) -> np.ndarray:
        """
        Create a skin texture variation map for realistic rendering.
        
        Args:
            shape: Size of the output texture map
            std_lighting: Standard deviation of lighting in the masked region
            
        Returns:
            Texture variation map
        """
        # Create noise texture
        texture = np.zeros(shape, dtype=np.float32)
        
        # Add Perlin-like noise
        scale = 0.1
        octaves = 4
        persistence = 0.5
        
        for y in range(shape[0]):
            for x in range(shape[1]):
                noise = 0
                amplitude = 1.0
                frequency = 1.0
                
                for i in range(octaves):
                    noise += self._smooth_noise(x * scale * frequency, 
                                              y * scale * frequency) * amplitude
                    
                    amplitude *= persistence
                    frequency *= 2
                
                texture[y, x] = noise
        
        # Normalize texture to 0-1 range
        texture = (texture - np.min(texture)) / (np.max(texture) - np.min(texture))
        
        # Scale texture by lighting variation
        texture_intensity = std_lighting / 255.0 * 30.0  # Scale factor
        texture = texture * texture_intensity
        
        return texture
    
    def _smooth_noise(self, x: float, y: float) -> float:
        """
        Generate smooth noise value at coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Noise value
        """
        # Simple implementation of smooth noise
        corners = (math.sin(x) * math.cos(y) * 10000) % 1.0
        sides = (math.cos(x + y) * 10000) % 1.0
        center = (math.sin(x * 0.7 + y * 0.3) * 10000) % 1.0
        
        return (corners + sides + center) / 3.0
    
    def _blend_overlay(self, frame: np.ndarray, design: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Blend design overlay with frame.
        
        Args:
            frame: Input image frame
            design: Tattoo design image
            mask: Skin detection mask
            
        Returns:
            Blended result
        """
        result = frame.copy()
        
        # Ensure design has the same size as the frame
        if design.shape[:2] != frame.shape[:2]:
            design_resized = np.zeros_like(frame, dtype=np.uint8)
            design_resized[:design.shape[0], :design.shape[1]] = design[:, :, :3]
            
            if design.shape[2] == 4:  # If design has alpha channel
                alpha = np.zeros(frame.shape[:2], dtype=np.uint8)
                alpha[:design.shape[0], :design.shape[1]] = design[:, :, 3]
            else:
                alpha = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        else:
            design_resized = design[:, :, :3]
            alpha = design[:, :, 3] if design.shape[2] == 4 else np.ones(frame.shape[:2], dtype=np.uint8) * 255
        
        # Create combined mask (skin mask AND design alpha)
        combined_mask = cv2.bitwise_and(mask, alpha)
        
        # Apply different blend modes
        if self.blend_mode == BlendMode.NORMAL:
            # Normal blending
            cv2.copyTo(design_resized, combined_mask, result)
            
        elif self.blend_mode == BlendMode.MULTIPLY:
            # Multiply blending
            blended = cv2.multiply(frame.astype(np.float32)/255.0, 
                                 design_resized.astype(np.float32)/255.0) * 255.0
            np.place(result, combined_mask[:, :, np.newaxis] > 0, blended.astype(np.uint8))
            
        elif self.blend_mode == BlendMode.OVERLAY:
            # Overlay blending
            low = 2 * frame.astype(np.float32)/255.0 * design_resized.astype(np.float32)/255.0
            high = 1 - 2 * (1 - frame.astype(np.float32)/255.0) * (1 - design_resized.astype(np.float32)/255.0)
            
            blended = np.where(
                frame.astype(np.float32)/255.0 < 0.5,
                low,
                high
            ) * 255.0
            
            np.place(result, combined_mask[:, :, np.newaxis] > 0, blended.astype(np.uint8))
            
        elif self.blend_mode == BlendMode.SCREEN:
            # Screen blending
            blended = (1 - (1 - frame.astype(np.float32)/255.0) * 
                       (1 - design_resized.astype(np.float32)/255.0)) * 255.0
            np.place(result, combined_mask[:, :, np.newaxis] > 0, blended.astype(np.uint8))
            
        elif self.blend_mode == BlendMode.SOFT_LIGHT:
            # Soft light blending
            low = 2 * frame.astype(np.float32)/255.0 * design_resized.astype(np.float32)/255.0
            high = 1 - 2 * (1 - design_resized.astype(np.float32)/255.0) * (1 - frame.astype(np.float32)/255.0)
            blended = np.where(
                design_resized.astype(np.float32)/255.0 < 0.5,
                frame.astype(np.float32)/255.0 * (0.5 + design_resized.astype(np.float32)/255.0),
                low * 0.5 + high * 0.5
            ) * 255.0
            np.place(result, combined_mask[:, :, np.newaxis] > 0, blended.astype(np.uint8))
            
        else:
            # Default to alpha blending
            cv2.copyTo(design_resized, combined_mask, result)
        
        # Apply opacity
        if self.opacity < 1.0:
            alpha_blend = cv2.addWeighted(
                frame, 1.0 - self.opacity,
                result, self.opacity,
                0
            )
            # Only blend in the mask region
            np.place(result, combined_mask[:, :, np.newaxis] > 0, 
                    alpha_blend[combined_mask > 0].reshape(-1, 3))
        
        return result
    
    def set_opacity(self, opacity: float) -> None:
        """
        Set the opacity of the rendered tattoo.
        
        Args:
            opacity: Opacity value (0.0 to 1.0)
        """
        self.opacity = max(0.0, min(1.0, opacity))
        logger.info(f"Tattoo opacity set to {self.opacity}")
    
    def set_blend_mode(self, blend_mode: Union[BlendMode, str]) -> None:
        """
        Set the blending mode for tattoo rendering.
        
        Args:
            blend_mode: Blending mode or mode name
        """
        if isinstance(blend_mode, str):
            try:
                blend_mode = BlendMode(blend_mode.lower())
            except ValueError:
                logger.warning(f"Invalid blend mode: {blend_mode}, using multiply")
                blend_mode = BlendMode.MULTIPLY
        
        self.blend_mode = blend_mode
        logger.info(f"Tattoo blend mode set to {self.blend_mode.value}")
    
    def get_design_info(self) -> Dict[str, Any]:
        """
        Get information about the current design.
        
        Returns:
            Dictionary with design information
        """
        info = {
            "loaded": self.current_design is not None,
            "metrics": self.design_metrics,
            "blend_mode": self.blend_mode.value if self.blend_mode else "none",
            "opacity": self.opacity,
            "high_quality": self.high_quality
        }
        
        return info


# Factory function for creating tattoo renderer with different configurations
def create_tattoo_renderer(
    style: Union[TattooStyle, str] = TattooStyle.TRADITIONAL,
    quality_preset: str = "balanced"
) -> TattooRenderer:
    """
    Create a tattoo renderer with specific configuration.
    
    Args:
        style: Tattoo style or style name
        quality_preset: Quality preset ("performance", "balanced", "quality")
        
    Returns:
        Configured tattoo renderer instance
    """
    # Convert style string to enum
    if isinstance(style, str):
        try:
            style = TattooStyle(style.lower())
        except ValueError:
            logger.warning(f"Invalid tattoo style: {style}, using traditional")
            style = TattooStyle.TRADITIONAL
    
    # Set initial rendering parameters based on style
    if style == TattooStyle.TRADITIONAL:
        opacity = 0.9
        blend_mode = BlendMode.MULTIPLY
    elif style == TattooStyle.REALISTIC:
        opacity = 0.85
        blend_mode = BlendMode.SOFT_LIGHT
    elif style == TattooStyle.BLACKWORK:
        opacity = 0.95
        blend_mode = BlendMode.MULTIPLY
    elif style == TattooStyle.WATERCOLOR:
        opacity = 0.7
        blend_mode = BlendMode.OVERLAY
    elif style == TattooStyle.TRIBAL:
        opacity = 0.9
        blend_mode = BlendMode.MULTIPLY
    else:
        opacity = 0.9
        blend_mode = BlendMode.MULTIPLY
    
    # Create renderer
    high_quality = quality_preset != "performance"
    adapt_to_lighting = quality_preset != "performance"
    
    renderer = TattooRenderer(
        opacity=opacity,
        blend_mode=blend_mode,
        adapt_to_lighting=adapt_to_lighting,
        high_quality=high_quality
    )
    
    return renderer