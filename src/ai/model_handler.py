"""
AI model handler for the Th.ink AR application.

This module provides functionality to load, manage, and use AI models for 
tattoo generation, skin detection, and motion tracking with optimized performance.
"""

import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import os
import asyncio
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

from ..config.model_config import get_config
from ..utils.performance_monitor import measure_time
from ..errors.error_handler import handle_errors, ModelError

# Configure logger
logger = logging.getLogger(__name__)


class ModelHandler:
    """
    Handler for AI model operations in the AR application.
    
    This class provides methods to load, manage, and use AI models with
    features like model quantization, caching, and dynamic resource allocation.
    """

    def __init__(self, config=None):
        """
        Initialize the model handler.
        
        Args:
            config: Configuration object or None to use default
        """
        self.config = config or get_config()
        self.device = self._get_optimal_device()
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.inference_cache = {}
        self.model_lock = asyncio.Lock()
        
        # Setup model paths
        self._setup_model_paths()
        
        # Performance tracking
        self.execution_times: Dict[str, List[float]] = {}
        self.memory_usage = {}
        
        logger.info(f"Model handler initialized with device: {self.device}")

    def _setup_model_paths(self):
        """Set up model paths based on configuration."""
        base_path = self.config.ai.model_path
        self.model_paths = {
            "skin_detection": Path(base_path) / self.config.ai.skin_detection_model,
            "tattoo_generator": Path(base_path) / self.config.ai.tattoo_generator_model,
            "motion_tracking": Path(base_path) / self.config.ai.motion_tracking_model
        }
        
        # Ensure the model directory exists
        for path in self.model_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)

    def _get_optimal_device(self) -> torch.device:
        """
        Get the optimal device for model execution based on available hardware.
        
        Returns:
            torch.device: The optimal device (CUDA, MPS, or CPU)
        """
        if self.config.ai.allow_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            
            # Set CUDA performance optimizations
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.benchmark = True
            
        elif self.config.ai.allow_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # For Apple Silicon (M1/M2) devices
            device = torch.device("mps")
            logger.info("Using Apple Metal Performance Shaders (MPS)")
            
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for model inference")
            
            # Set number of threads for CPU execution
            if hasattr(torch, 'set_num_threads'):
                num_threads = os.cpu_count() or 4
                torch.set_num_threads(min(num_threads, 8))  # Limit to 8 threads max
                logger.info(f"Set PyTorch to use {min(num_threads, 8)} CPU threads")
        
        return device

    @measure_time("model_loading")
    @handle_errors()
    async def load_model(self, model_name: str) -> bool:
        """
        Load an AI model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            bool: True if model loaded successfully, False otherwise
            
        Raises:
            ModelError: If model loading fails
        """
        # Use lock to prevent concurrent model loading
        async with self.model_lock:
            # Skip if model is already loaded
            if model_name in self.models:
                logger.debug(f"Model {model_name} already loaded")
                return True
            
            try:
                # Get model path
                if model_name not in self.model_paths:
                    raise ModelError(f"Unknown model: {model_name}")
                
                model_path = self.model_paths[model_name]
                
                if not model_path.exists():
                    raise ModelError(f"Model {model_name} not found at {model_path}")
                
                # Run model loading in a separate thread to avoid blocking
                start_time = time.time()
                model = await asyncio.to_thread(self._load_model_file, model_path, model_name)
                load_time = time.time() - start_time
                
                # Store model and metadata
                self.models[model_name] = model
                self.model_info[model_name] = {
                    "path": str(model_path),
                    "load_time": load_time,
                    "loaded_at": time.time(),
                    "device": str(self.device),
                    "memory": self._get_model_memory_usage(model)
                }
                
                logger.info(f"Successfully loaded model {model_name} in {load_time:.2f}s")
                return True
                
            except Exception as e:
                error_msg = f"Failed to load model {model_name}: {str(e)}"
                logger.error(error_msg)
                raise ModelError(error_msg)

    def _load_model_file(self, model_path: Path, model_name: str) -> Any:
        """
        Load model file based on its type.
        
        Args:
            model_path: Path to the model file
            model_name: Name of the model
            
        Returns:
            The loaded model
            
        Raises:
            Exception: If loading fails
        """
        # Determine model format from extension
        if model_path.suffix == ".pth":
            # Load TorchScript model
            model = torch.jit.load(str(model_path))
            model.to(self.device)
            
            # Apply quantization if configured and on CPU
            if self.config.ai.quantization and self.device.type == "cpu":
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
                )
                logger.info(f"Applied quantization to model {model_name}")
                
            # Set eval mode for inference
            model.eval()
            
        elif model_path.suffix == ".onnx":
            # Load ONNX model
            try:
                import onnxruntime as ort
                
                # Configure ONNX runtime session
                session_options = ort.SessionOptions()
                
                # Set execution provider based on device
                providers = []
                if self.device.type == "cuda":
                    providers.append('CUDAExecutionProvider')
                elif self.device.type == "mps":
                    providers.append('CoreMLExecutionProvider')
                providers.append('CPUExecutionProvider')
                
                # Create ONNX session
                model = ort.InferenceSession(
                    str(model_path),
                    sess_options=session_options,
                    providers=providers
                )
                
            except ImportError:
                raise ImportError("ONNX Runtime not installed. Please install onnxruntime package.")
                
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
            
        return model

    @measure_time("model_unloading")
    @handle_errors()
    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a model to free resources.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            bool: True if model was unloaded, False if not found
        """
        async with self.model_lock:
            if model_name in self.models:
                # Remove model from dictionary
                del self.models[model_name]
                
                # Clean up model info
                if model_name in self.model_info:
                    del self.model_info[model_name]
                
                # Clear cache entries for this model
                self._clear_model_cache(model_name)
                
                # Force CUDA memory cleanup if applicable
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                
                logger.info(f"Unloaded model {model_name}")
                return True
            
            logger.warning(f"Model {model_name} not loaded, nothing to unload")
            return False

    @measure_time("predict")
    @handle_errors()
    async def predict(self, model_name: str, input_data: Union[torch.Tensor, np.ndarray, Dict[str, Any]], 
                cache_key: Optional[str] = None) -> Optional[Any]:
        """
        Run inference with the specified model.
        
        Args:
            model_name: Name of the model to use
            input_data: Input data for the model (tensor, numpy array, or dict for ONNX)
            cache_key: Optional key for caching results
            
        Returns:
            Model output or None if prediction fails
            
        Raises:
            ModelError: If prediction fails
        """
        # Check if model is loaded
        if model_name not in self.models:
            # Try to load the model
            if not await self.load_model(model_name):
                raise ModelError(f"Model {model_name} could not be loaded")
                
        # Check cache if enabled and cache_key provided
        if self.config.performance.caching.enabled and cache_key:
            cache_result = self._check_cache(model_name, cache_key)
            if cache_result is not None:
                logger.debug(f"Cache hit for {model_name} with key {cache_key[:10]}...")
                return cache_result
                
        try:
            model = self.models[model_name]
            
            # Prepare input based on model type
            if isinstance(model, torch.jit.ScriptModule):
                # For PyTorch models
                if isinstance(input_data, np.ndarray):
                    input_tensor = torch.from_numpy(input_data)
                elif isinstance(input_data, torch.Tensor):
                    input_tensor = input_data
                else:
                    raise ValueError(f"Unsupported input type for PyTorch model: {type(input_data)}")
                
                # Move input to correct device
                input_tensor = input_tensor.to(self.device)
                
                # Run inference in no_grad context for efficiency
                with torch.no_grad():
                    start_time = time.time()
                    output = model(input_tensor)
                    inference_time = time.time() - start_time
                
            elif hasattr(model, 'run'):
                # For ONNX models
                if isinstance(input_data, dict):
                    # ONNX takes named inputs
                    start_time = time.time()
                    output = model.run(None, input_data)
                    inference_time = time.time() - start_time
                else:
                    raise ValueError("ONNX models require dictionary input with named tensors")
                    
            else:
                raise ModelError(f"Unsupported model type: {type(model)}")
                
            # Track execution time
            self._track_execution_time(model_name, inference_time)
            
            # Cache result if caching is enabled
            if self.config.performance.caching.enabled and cache_key:
                self._cache_result(model_name, cache_key, output)
                
            return output
            
        except Exception as e:
            error_msg = f"Prediction failed for model {model_name}: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg)

    def _check_cache(self, model_name: str, cache_key: str) -> Optional[Any]:
        """
        Check if result is in cache.
        
        Args:
            model_name: Model name
            cache_key: Cache key
            
        Returns:
            Cached result or None if not found
        """
        cache_dict = self.inference_cache.get(model_name, {})
        if cache_key in cache_dict:
            cache_entry = cache_dict[cache_key]
            # Check if cache entry is still valid
            if time.time() - cache_entry["timestamp"] < self.config.performance.caching.ttl_seconds:
                return cache_entry["result"]
            
            # Remove expired entry
            del cache_dict[cache_key]
            
        return None

    def _cache_result(self, model_name: str, cache_key: str, result: Any) -> None:
        """
        Cache inference result.
        
        Args:
            model_name: Model name
            cache_key: Cache key
            result: Result to cache
        """
        if model_name not in self.inference_cache:
            self.inference_cache[model_name] = {}
            
        self.inference_cache[model_name][cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
        
        # Check cache size and prune if needed
        self._prune_cache_if_needed()

    def _clear_model_cache(self, model_name: str) -> None:
        """
        Clear cache entries for specific model.
        
        Args:
            model_name: Model name
        """
        if model_name in self.inference_cache:
            del self.inference_cache[model_name]

    def _prune_cache_if_needed(self) -> None:
        """Prune cache if it exceeds configured size limit."""
        # Calculate current cache size (rough estimate)
        total_entries = sum(len(entries) for entries in self.inference_cache.values())
        
        if total_entries > self.config.performance.caching.max_size:
            logger.info(f"Pruning cache with {total_entries} entries")
            
            # Remove oldest entries first
            for model_name in list(self.inference_cache.keys()):
                cache_dict = self.inference_cache[model_name]
                
                # Sort by timestamp
                sorted_entries = sorted(
                    cache_dict.items(), 
                    key=lambda x: x[1]["timestamp"]
                )
                
                # Remove oldest 25% of entries
                num_to_remove = max(1, len(sorted_entries) // 4)
                for i in range(num_to_remove):
                    if i < len(sorted_entries):
                        key = sorted_entries[i][0]
                        del cache_dict[key]

    def _track_execution_time(self, model_name: str, execution_time: float) -> None:
        """
        Track model execution time for performance monitoring.
        
        Args:
            model_name: Model name
            execution_time: Execution time in seconds
        """
        if model_name not in self.execution_times:
            self.execution_times[model_name] = []
            
        times_list = self.execution_times[model_name]
        times_list.append(execution_time)
        
        # Keep only last 100 times
        if len(times_list) > 100:
            times_list.pop(0)

    def _get_model_memory_usage(self, model) -> Dict[str, Any]:
        """
        Get memory usage information for a model.
        
        Args:
            model: The model object
            
        Returns:
            Dict with memory usage information
        """
        memory_info = {"unit": "bytes"}
        
        try:
            # For PyTorch models
            if isinstance(model, torch.nn.Module) or isinstance(model, torch.jit.ScriptModule):
                total_size = 0
                for param in model.parameters():
                    # Calculate size in bytes
                    total_size += param.nelement() * param.element_size()
                    
                memory_info["model_size"] = total_size
                memory_info["size_mb"] = total_size / (1024 * 1024)
                
                # Get CUDA memory if applicable
                if self.device.type == "cuda":
                    memory_info["cuda_allocated"] = torch.cuda.memory_allocated()
                    memory_info["cuda_reserved"] = torch.cuda.memory_reserved()
            
            # For ONNX models, size estimation is more complex
            # and would require additional implementation
            
        except Exception as e:
            logger.warning(f"Error calculating model memory usage: {e}")
            memory_info["error"] = str(e)
            
        return memory_info

    def get_model_stats(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics for models.
        
        Args:
            model_name: Optional specific model name, or None for all models
            
        Returns:
            Dict with model statistics
        """
        stats = {}
        
        if model_name:
            # Stats for specific model
            if model_name in self.execution_times:
                times = self.execution_times[model_name]
                if times:
                    stats["avg_time"] = sum(times) / len(times)
                    stats["min_time"] = min(times)
                    stats["max_time"] = max(times)
                    stats["calls"] = len(times)
                    
            if model_name in self.model_info:
                stats.update(self.model_info[model_name])
                
        else:
            # Stats for all models
            stats["loaded_models"] = list(self.models.keys())
            stats["device"] = str(self.device)
            stats["model_count"] = len(self.models)
            stats["cache_entries"] = sum(len(entries) for entries in self.inference_cache.values())
            
            # Compute aggregate stats
            all_times = []
            for times in self.execution_times.values():
                all_times.extend(times)
                
            if all_times:
                stats["overall_avg_time"] = sum(all_times) / len(all_times)
                stats["overall_min_time"] = min(all_times)
                stats["overall_max_time"] = max(all_times)
                stats["total_calls"] = len(all_times)
                
        return stats

    async def load_all_models(self) -> Dict[str, bool]:
        """
        Load all configured models.
        
        Returns:
            Dict mapping model names to loading success status
        """
        results = {}
        for model_name in self.model_paths.keys():
            try:
                success = await self.load_model(model_name)
                results[model_name] = success
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
                results[model_name] = False
                
        return results

    async def unload_all_models(self) -> None:
        """Unload all loaded models to free resources."""
        for model_name in list(self.models.keys()):
            await self.unload_model(model_name)

    def clear_cache(self) -> int:
        """
        Clear all cached results.
        
        Returns:
            Number of cleared cache entries
        """
        total_entries = sum(len(entries) for entries in self.inference_cache.values())
        self.inference_cache = {}
        logger.info(f"Cleared {total_entries} cache entries")
        return total_entries

    def generate_cache_key(self, data: Any) -> str:
        """
        Generate a cache key for input data.
        
        Args:
            data: Input data
            
        Returns:
            Cache key string
        """
        if isinstance(data, torch.Tensor):
            # Use tensor shape and first/last values for faster hashing
            shape_str = str(data.shape)
            sample = f"{data.flatten()[0].item():.6f}_{data.flatten()[-1].item():.6f}"
            unique_str = f"{shape_str}_{sample}"
        elif isinstance(data, np.ndarray):
            shape_str = str(data.shape)
            sample = f"{data.flatten()[0]:.6f}_{data.flatten()[-1]:.6f}"
            unique_str = f"{shape_str}_{sample}"
        elif isinstance(data, dict):
            # For ONNX inputs
            key_parts = []
            for key, value in sorted(data.items()):
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    key_parts.append(f"{key}_{len(value)}")
                else:
                    key_parts.append(f"{key}_{value}")
            unique_str = "_".join(key_parts)
        else:
            # Fall back to JSON for other types
            try:
                unique_str = json.dumps(data, sort_keys=True)
            except (TypeError, ValueError):
                # If JSON fails, use string representation
                unique_str = str(data)
                
        # Generate hash
        return hashlib.md5(unique_str.encode('utf-8')).hexdigest()


# Create a singleton instance
_model_handler = None

def get_model_handler() -> ModelHandler:
    """
    Get the singleton model handler instance.
    
    Returns:
        ModelHandler: The model handler instance
    """
    global _model_handler
    if _model_handler is None:
        _model_handler = ModelHandler()
    return _model_handler