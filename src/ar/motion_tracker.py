"""
Motion tracking and body keypoint detection module for the Th.ink AR application.

This module provides functionality to track body parts and keypoints in real-time
camera frames, optimized for AR tattoo placement and visualization.
"""

import torch
import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
import time

from ..utils.performance_monitor import measure_time
from ..errors.error_handler import handle_errors, ModelError, CameraError
from ..config.model_config import get_config

# Configure logger
logger = logging.getLogger("think.ar.tracking")


class TrackingModel(Enum):
    """Available body tracking model types."""
    
    YOLO = "yolov5"
    MEDIAPIPE = "mediapipe"
    BLAZEPOSE = "blazepose"
    CUSTOM = "custom"


@dataclass
class Keypoint:
    """Body keypoint representation."""
    
    id: int
    name: str
    x: float
    y: float
    z: float = 0.0
    confidence: float = 0.0
    visible: bool = True


@dataclass
class BodyPart:
    """Body part representation with associated keypoints."""
    
    name: str
    keypoints: List[Keypoint]
    confidence: float = 0.0
    bounds: Optional[Dict[str, float]] = None


class MotionTracker:
    """
    Motion tracking and body keypoint detection for AR tattoo visualization.
    
    This class provides methods to track body parts and keypoints in real-time
    camera frames, with support for multiple tracking models and optimizations.
    """
    
    def __init__(self, model_type: TrackingModel = TrackingModel.MEDIAPIPE):
        """
        Initialize the motion tracker.
        
        Args:
            model_type: Tracking model type to use
        """
        self.model_type = model_type
        self.model = None
        self.config = get_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() and 
                                   self.config.ai.allow_gpu else "cpu")
        self.keypoints = []
        self.body_parts = []
        self.tracking_active = False
        self.last_detected_frame = 0
        self.frame_count = 0
        self.detection_interval = 3  # Process every Nth frame for performance
        
        # Performance metrics
        self.processing_times = []
        self.max_processing_times = 100  # Keep last N times
        
        logger.info(f"Motion tracker initialized with {model_type.value} model on {self.device}")
    
    @measure_time("model_initialization")
    @handle_errors()
    def initialize_model(self) -> bool:
        """
        Initialize tracking model with optimizations.
        
        Returns:
            True if initialization successful, False otherwise
            
        Raises:
            ModelError: If model initialization fails
        """
        try:
            if self.model_type == TrackingModel.YOLO:
                self._initialize_yolo_model()
            elif self.model_type == TrackingModel.MEDIAPIPE:
                self._initialize_mediapipe_model()
            elif self.model_type == TrackingModel.BLAZEPOSE:
                self._initialize_blazepose_model()
            elif self.model_type == TrackingModel.CUSTOM:
                self._initialize_custom_model()
            else:
                raise ModelError(f"Unsupported model type: {self.model_type.value}")
            
            self.tracking_active = True
            return True
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.tracking_active = False
            raise ModelError(f"Failed to initialize {self.model_type.value} model: {str(e)}")
    
    def _initialize_yolo_model(self) -> None:
        """Initialize YOLOv5 model with optimizations."""
        logger.info("Initializing YOLOv5 model")
        
        try:
            # Use PyTorch Hub to load YOLOv5
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.to(self.device)
            
            # Optimize model for inference
            self.model.eval()
            
            if self.config.ai.quantization:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
                )
                logger.info("Applied quantization to YOLOv5 model")
            
            # Set inference parameters
            self.model.conf = self.config.ai.min_confidence  # Confidence threshold
            self.model.classes = [0]  # Only detect people (class 0)
            
            logger.info("YOLOv5 model initialized successfully")
            
        except Exception as e:
            logger.error(f"YOLOv5 initialization error: {e}")
            raise ModelError(f"Failed to initialize YOLOv5: {str(e)}")
    
    def _initialize_mediapipe_model(self) -> None:
        """Initialize MediaPipe model."""
        logger.info("Initializing MediaPipe model")
        
        try:
            # Import here to avoid dependency if not used
            import mediapipe as mp
            
            self.mp = mp
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Initialize pose detection with performance settings
            self.model = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
                smooth_landmarks=True,
                min_detection_confidence=self.config.ai.min_confidence,
                min_tracking_confidence=self.config.ar.tracking.tracking_precision
            )
            
            logger.info("MediaPipe model initialized successfully")
            
        except Exception as e:
            logger.error(f"MediaPipe initialization error: {e}")
            raise ModelError(f"Failed to initialize MediaPipe: {str(e)}")
    
    def _initialize_blazepose_model(self) -> None:
        """Initialize BlazePose model."""
        logger.info("Initializing BlazePose model")
        
        try:
            # Import here to avoid dependency if not used
            import mediapipe as mp
            
            self.mp = mp
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Initialize BlazePose with high performance settings
            self.model = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,  # Use the heavy model for better accuracy
                smooth_landmarks=True,
                enable_segmentation=True,  # Enable segmentation for better results
                min_detection_confidence=self.config.ai.min_confidence,
                min_tracking_confidence=self.config.ar.tracking.tracking_precision
            )
            
            logger.info("BlazePose model initialized successfully")
            
        except Exception as e:
            logger.error(f"BlazePose initialization error: {e}")
            raise ModelError(f"Failed to initialize BlazePose: {str(e)}")
    
    def _initialize_custom_model(self) -> None:
        """Initialize custom tracking model."""
        logger.info("Initializing custom tracking model")
        
        model_path = Path(self.config.ai.model_path) / self.config.ai.motion_tracking_model
        
        if not model_path.exists():
            raise ModelError(f"Custom model file not found at {model_path}")
        
        try:
            # Load custom model (assuming TorchScript model)
            self.model = torch.jit.load(str(model_path))
            self.model.to(self.device)
            self.model.eval()
            
            if self.config.ai.quantization:
                # Apply quantization for better performance
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
                )
                logger.info("Applied quantization to custom model")
            
            logger.info("Custom model initialized successfully")
            
        except Exception as e:
            logger.error(f"Custom model initialization error: {e}")
            raise ModelError(f"Failed to initialize custom model: {str(e)}")
    
    @measure_time("track_frame")
    @handle_errors()
    def track_frame(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Track body keypoints in frame.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Tuple of (list of keypoints, visualization frame)
            
        Raises:
            CameraError: If frame processing fails
        """
        if not self.tracking_active or self.model is None:
            return [], frame
        
        self.frame_count += 1
        
        # Skip frames for performance if needed
        if self.frame_count % self.detection_interval != 0:
            # Return last detected keypoints
            return self.keypoints, self._visualize_tracking(frame, self.keypoints)
        
        try:
            start_time = time.time()
            
            if self.model_type == TrackingModel.YOLO:
                keypoints = self._process_yolo(frame)
            elif self.model_type in [TrackingModel.MEDIAPIPE, TrackingModel.BLAZEPOSE]:
                keypoints = self._process_mediapipe(frame)
            elif self.model_type == TrackingModel.CUSTOM:
                keypoints = self._process_custom(frame)
            else:
                keypoints = []
            
            # Update stored keypoints
            self.keypoints = keypoints
            self.last_detected_frame = self.frame_count
            
            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_times:
                self.processing_times.pop(0)
            
            # Log unusual processing times
            avg_time = sum(self.processing_times) / len(self.processing_times)
            if processing_time > avg_time * 2 and len(self.processing_times) > 10:
                logger.warning(f"Unusually long frame processing: {processing_time:.4f}s (avg: {avg_time:.4f}s)")
            
            # Generate visualization
            visualized_frame = self._visualize_tracking(frame, keypoints)
            
            return keypoints, visualized_frame
            
        except Exception as e:
            logger.error(f"Error tracking frame: {e}")
            raise CameraError(f"Failed to process frame: {str(e)}")
    
    def _process_yolo(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process frame with YOLOv5 model.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of detected keypoints
        """
        # Convert frame to RGB (YOLO expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        with torch.no_grad():
            results = self.model(rgb_frame)
        
        # Process results
        keypoints = []
        
        # Get detections
        detections = results.xyxy[0].cpu().numpy()
        
        # Process each person detection
        for *xyxy, conf, cls in detections:
            if int(cls) == 0:  # Person class
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Create keypoint for center of bounding box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                keypoints.append({
                    "part": "torso",
                    "position": {"x": center_x, "y": center_y},
                    "confidence": float(conf),
                    "bounds": {
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1
                    }
                })
        
        return keypoints
    
    def _process_mediapipe(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process frame with MediaPipe model.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of detected keypoints
        """
        # Convert frame to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.model.process(rgb_frame)
        
        if not results.pose_landmarks:
            return []
        
        keypoints = []
        landmarks = results.pose_landmarks.landmark
        
        # MediaPipe body part definitions
        body_parts = {
            "face": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "left_arm": [11, 13, 15, 17, 19, 21],
            "right_arm": [12, 14, 16, 18, 20, 22],
            "torso": [11, 12, 23, 24],
            "left_leg": [23, 25, 27, 29, 31],
            "right_leg": [24, 26, 28, 30, 32]
        }
        
        # Extract landmarks for each body part
        for part_name, part_indices in body_parts.items():
            # Skip if any landmark has low confidence
            part_landmarks = [landmarks[i] for i in part_indices]
            
            # Calculate average confidence
            avg_confidence = sum(lm.visibility for lm in part_landmarks) / len(part_landmarks)
            
            # Skip if confidence is too low
            if avg_confidence < self.config.ai.min_confidence:
                continue
            
            # Calculate bounding box for the body part
            x_coords = [lm.x * frame.shape[1] for lm in part_landmarks]
            y_coords = [lm.y * frame.shape[0] for lm in part_landmarks]
            
            if not x_coords or not y_coords:
                continue
                
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            width = x_max - x_min
            height = y_max - y_min
            
            # Add some padding
            padding = 0.2
            x_min -= width * padding
            y_min -= height * padding
            width += width * padding * 2
            height += height * padding * 2
            
            # Create keypoint for body part
            keypoints.append({
                "part": part_name,
                "position": {
                    "x": (x_min + width / 2),
                    "y": (y_min + height / 2)
                },
                "confidence": float(avg_confidence),
                "bounds": {
                    "x": float(max(0, x_min)),
                    "y": float(max(0, y_min)),
                    "width": float(width),
                    "height": float(height)
                }
            })
        
        return keypoints
    
    def _process_custom(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process frame with custom model.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of detected keypoints
        """
        # Convert frame to tensor
        img = torch.from_numpy(frame).to(self.device)
        img = img.permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(img)
        
        # Process outputs (format depends on custom model)
        # This is a placeholder - adjust based on your model's output format
        keypoints = []
        
        # Assuming outputs is a tensor with keypoint coordinates and confidences
        if isinstance(outputs, torch.Tensor) and outputs.numel() > 0:
            # Convert to numpy for processing
            detections = outputs[0].cpu().numpy()
            
            # Process each keypoint
            for i, detection in enumerate(detections):
                # Assuming format [x, y, confidence]
                if len(detection) >= 3:
                    x, y, conf = detection[:3]
                    
                    # Skip low confidence detections
                    if conf < self.config.ai.min_confidence:
                        continue
                    
                    # Map index to body part
                    part_name = self._map_index_to_body_part(i)
                    
                    keypoints.append({
                        "part": part_name,
                        "position": {"x": float(x), "y": float(y)},
                        "confidence": float(conf)
                    })
        
        return keypoints
    
    def _map_index_to_body_part(self, index: int) -> str:
        """
        Map keypoint index to body part name.
        
        Args:
            index: Keypoint index
            
        Returns:
            Body part name
        """
        # This mapping should be adjusted based on your custom model's output format
        part_mapping = {
            0: "face",
            1: "neck",
            2: "right_shoulder",
            3: "right_elbow",
            4: "right_wrist",
            5: "left_shoulder",
            6: "left_elbow",
            7: "left_wrist",
            8: "right_hip",
            9: "right_knee",
            10: "right_ankle",
            11: "left_hip",
            12: "left_knee",
            13: "left_ankle",
            14: "torso"
        }
        
        return part_mapping.get(index, "unknown")
    
    def _visualize_tracking(self, frame: np.ndarray, keypoints: List[Dict[str, Any]]) -> np.ndarray:
        """
        Visualize tracking results on frame.
        
        Args:
            frame: Input image frame (BGR format)
            keypoints: List of detected keypoints
            
        Returns:
            Visualization frame
        """
        visualization = frame.copy()
        
        # Visualization colors
        colors = {
            "face": (0, 255, 0),       # Green
            "torso": (0, 0, 255),      # Red
            "left_arm": (255, 0, 0),   # Blue
            "right_arm": (255, 0, 0),  # Blue
            "left_leg": (0, 255, 255), # Yellow
            "right_leg": (0, 255, 255) # Yellow
        }
        
        # Draw each keypoint
        for kp in keypoints:
            part = kp.get("part", "unknown")
            pos = kp.get("position", {})
            conf = kp.get("confidence", 0.0)
            bounds = kp.get("bounds")
            
            # Get color for body part
            color = colors.get(part, (255, 255, 255))
            
            # Draw position dot
            if "x" in pos and "y" in pos:
                x, y = int(pos["x"]), int(pos["y"])
                cv2.circle(visualization, (x, y), 5, color, -1)
                
                # Draw part name and confidence
                cv2.putText(
                    visualization,
                    f"{part}: {conf:.2f}",
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA
                )
            
            # Draw bounding box if available
            if bounds:
                x = int(bounds.get("x", 0))
                y = int(bounds.get("y", 0))
                w = int(bounds.get("width", 0))
                h = int(bounds.get("height", 0))
                
                cv2.rectangle(visualization, (x, y), (x + w, y + h), color, 2)
        
        return visualization
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            "avg_processing_time": 0.0,
            "max_processing_time": 0.0,
            "min_processing_time": 0.0,
            "detection_rate": 0.0
        }
        
        if self.processing_times:
            stats["avg_processing_time"] = sum(self.processing_times) / len(self.processing_times)
            stats["max_processing_time"] = max(self.processing_times)
            stats["min_processing_time"] = min(self.processing_times)
        
        if self.frame_count > 0:
            stats["detection_rate"] = self.last_detected_frame / self.frame_count
        
        return stats
    
    def set_detection_interval(self, interval: int) -> None:
        """
        Set frame detection interval for performance optimization.
        
        Args:
            interval: Process every Nth frame (1 = process all frames)
        """
        self.detection_interval = max(1, interval)
        logger.info(f"Motion tracking detection interval set to {self.detection_interval}")
    
    def release(self) -> None:
        """Release resources."""
        if self.model_type in [TrackingModel.MEDIAPIPE, TrackingModel.BLAZEPOSE] and self.model:
            self.model.close()
        
        self.model = None
        self.tracking_active = False
        logger.info("Motion tracker resources released")


# Factory function for creating motion tracker with different configurations
def create_motion_tracker(
    model_type: str = "mediapipe",
    quality_preset: str = "balanced"
) -> MotionTracker:
    """
    Create a motion tracker with specific configuration.
    
    Args:
        model_type: Model type name ("yolov5", "mediapipe", "blazepose", "custom")
        quality_preset: Quality preset ("performance", "balanced", "quality")
        
    Returns:
        Configured motion tracker instance
    """
    # Convert model type string to enum
    try:
        tracker_model = TrackingModel(model_type.lower())
    except ValueError:
        logger.warning(f"Invalid model type: {model_type}, using MediaPipe")
        tracker_model = TrackingModel.MEDIAPIPE
    
    # Create tracker
    tracker = MotionTracker(model_type=tracker_model)
    
    # Apply quality preset
    if quality_preset == "performance":
        tracker.set_detection_interval(4)  # Process every 4th frame
    elif quality_preset == "balanced":
        tracker.set_detection_interval(2)  # Process every 2nd frame
    elif quality_preset == "quality":
        tracker.set_detection_interval(1)  # Process every frame
    
    return tracker