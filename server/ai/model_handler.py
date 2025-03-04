"""
Model handler for AI-powered tattoo image generation.

This module provides functionality to generate tattoo images using various AI models,
abstracting the implementation details and providing a consistent interface.
"""

import os
import logging
import requests
import base64
import json
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
import time
import asyncio
import boto3
from PIL import Image
from io import BytesIO

from ..config.model_config import get_config
from ..errors.error_handler import handle_errors, ModelError

# Configure logger
logger = logging.getLogger("think.ai.model_handler")

# Get configuration
config = get_config()


class ModelHandler:
    """
    Handler for AI model operations including tattoo generation, style transfer, and manipulation.
    
    This class provides a unified interface to multiple AI models and services,
    with fallback mechanisms and optimized resource usage.
    """
    
    def __init__(self):
        """Initialize the model handler with configuration."""
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.stability_api_key = os.getenv('STABILITY_API_KEY')
        self.models_path = Path(os.getenv('MODELS_PATH', 'models'))
        self.use_local_models = config.ai.use_local_models
        self.use_aws_sagemaker = config.ai.use_aws_sagemaker
        self.use_cache = config.performance.caching.enabled
        self.cache = {}  # Simple in-memory cache
        
        # Initialize AWS clients if using SageMaker
        if self.use_aws_sagemaker:
            self._init_aws_clients()
            
        # Load local models if configured
        if self.use_local_models:
            self._load_local_models()
            
        logger.info("Model handler initialized")
    
    def _init_aws_clients(self):
        """Initialize AWS clients for SageMaker."""
        try:
            self.sagemaker_runtime = boto3.client('sagemaker-runtime')
            self.s3_client = boto3.client('s3')
            self.sagemaker_endpoint = config.services.sagemaker.get('endpoint_name')
            logger.info(f"AWS SageMaker client initialized with endpoint {self.sagemaker_endpoint}")
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {str(e)}")
            self.use_aws_sagemaker = False
    
    def _load_local_models(self):
        """Load local AI models for offline operation."""
        try:
            # This is a placeholder for local model loading logic
            # In a real implementation, you would load ML models here
            logger.info("Local models loaded")
        except Exception as e:
            logger.error(f"Failed to load local models: {str(e)}")
            self.use_local_models = False
    
    @handle_errors()
    async def generate_tattoo_image(self, description: str, style: str = "traditional", 
                               size: str = "medium", color: bool = True) -> Tuple[bytes, dict]:
        """
        Generate a tattoo image based on the provided description using available AI services.
        
        Args:
            description (str): The description of the tattoo to generate
            style (str): The tattoo style (traditional, realistic, etc.)
            size (str): Size category (small, medium, large)
            color (bool): Whether to generate a color tattoo
            
        Returns:
            Tuple[bytes, dict]: The generated image in bytes and metadata
            
        Raises:
            ModelError: If image generation fails on all available services
        """
        # Check cache first if enabled
        if self.use_cache:
            cache_key = f"{description}_{style}_{size}_{color}"
            if cache_key in self.cache:
                logger.info(f"Cache hit for: {description[:30]}...")
                return self.cache[cache_key]
        
        # Format prompt for better results
        formatted_prompt = self._format_generation_prompt(description, style, size, color)
        
        # Try different services in order of preference
        services = [
            self._generate_with_openai,
            self._generate_with_stability_ai,
            self._generate_with_sagemaker,
            self._generate_with_local_model
        ]
        
        # Track errors for better error reporting
        errors = []
        start_time = time.time()
        
        for service_func in services:
            try:
                # Only try SageMaker if configured
                if service_func == self._generate_with_sagemaker and not self.use_aws_sagemaker:
                    continue
                    
                # Only try local model if configured
                if service_func == self._generate_with_local_model and not self.use_local_models:
                    continue
                
                logger.info(f"Attempting to generate with {service_func.__name__}")
                image_bytes, metadata = await service_func(formatted_prompt, style, size, color)
                
                if image_bytes:
                    # Update metadata with generation info
                    metadata.update({
                        "generation_time": time.time() - start_time,
                        "service": service_func.__name__.replace("_generate_with_", "")
                    })
                    
                    # Cache result if enabled
                    if self.use_cache:
                        self.cache[cache_key] = (image_bytes, metadata)
                        
                    return image_bytes, metadata
            
            except Exception as e:
                error_msg = f"Error in {service_func.__name__}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                continue
        
        # If we get here, all services failed
        error_details = "\n".join(errors)
        raise ModelError(f"Failed to generate tattoo image using all available services.\nDetails: {error_details}")
    
    def _format_generation_prompt(self, description: str, style: str, size: str, color: bool) -> str:
        """
        Format the generation prompt for better results.
        
        Args:
            description (str): User's tattoo description
            style (str): Tattoo style
            size (str): Size category
            color (bool): Whether color is desired
            
        Returns:
            str: Formatted prompt for AI model
        """
        # Style-specific keywords
        style_keywords = {
            "traditional": "bold black outlines, limited color palette, American traditional tattoo style",
            "realistic": "photorealistic, detailed shading, depth and dimension, realistic tattoo style",
            "blackwork": "solid black ink, intricate patterns, geometric elements, blackwork tattoo style",
            "watercolor": "soft color blending, paint splatter effects, artistic, watercolor tattoo style",
            "japanese": "irezumi style, bold outlines, Japanese iconography, traditional Japanese tattoo",
            "minimalist": "simple clean lines, minimal detail, elegant, minimalist tattoo style",
            "tribal": "bold black tribal patterns, symmetrical design, Polynesian influenced tattoo style",
            "geometric": "precise geometric shapes, clean lines, mathematical patterns, geometric tattoo style",
            "new_school": "cartoonish, exaggerated, vibrant colors, bold outlines, new school tattoo style"
        }
        
        # Size adjustments
        size_desc = {
            "small": "small-sized tattoo, compact design, 2-3 inches",
            "medium": "medium-sized tattoo, 4-6 inches",
            "large": "large-sized tattoo, detailed artwork, 7+ inches"
        }
        
        # Color modifiers
        color_desc = "full color tattoo design" if color else "black and gray tattoo design, no color"
        
        # Build the prompt
        style_desc = style_keywords.get(style.lower(), style)
        prompt = (
            f"A high-quality tattoo design of {description}. "
            f"{style_desc}, {size_desc.get(size.lower(), 'medium-sized')}, {color_desc}. "
            f"Clean background, high contrast, suitable for tattooing on skin."
        )
        
        return prompt
        
    async def _generate_with_openai(self, prompt: str, style: str, 
                               size: str, color: bool) -> Tuple[bytes, dict]:
        """
        Generate image using OpenAI's DALL-E API.
        
        Args:
            prompt (str): The formatted prompt
            style (str): Tattoo style
            size (str): Size category
            color (bool): Whether color is desired
            
        Returns:
            Tuple[bytes, dict]: Generated image and metadata
            
        Raises:
            Exception: If the image generation fails
        """
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        headers = {
            'Authorization': f'Bearer {self.openai_api_key}',
            'Content-Type': 'application/json',
        }
        
        # Determine image quality and size based on parameters
        quality = "hd" if style in ["realistic", "watercolor"] else "standard"
        image_size = "1024x1024"  # Standard size for tattoo designs
        
        data = {
            'model': 'dall-e-3',
            'prompt': prompt,
            'n': 1,
            'quality': quality,
            'size': image_size,
            'style': 'vivid' if color else 'natural'
        }
        
        response = await asyncio.to_thread(
            requests.post, 
            'https://api.openai.com/v1/images/generations', 
            headers=headers, 
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            image_url = result['data'][0]['url']
            # Fetch the image bytes
            image_response = await asyncio.to_thread(requests.get, image_url)
            
            if image_response.status_code == 200:
                metadata = {
                    'model': 'dall-e-3',
                    'prompt': prompt,
                    'revised_prompt': result['data'][0].get('revised_prompt', prompt),
                    'size': image_size,
                    'quality': quality
                }
                return image_response.content, metadata
            else:
                raise Exception(f"Failed to fetch generated image: {image_response.text}")
        else:
            raise Exception(f"OpenAI image generation failed: {response.text}")
    
    async def _generate_with_stability_ai(self, prompt: str, style: str, 
                                     size: str, color: bool) -> Tuple[bytes, dict]:
        """
        Generate image using Stability AI's API.
        
        Args:
            prompt (str): The formatted prompt
            style (str): Tattoo style
            size (str): Size category
            color (bool): Whether color is desired
            
        Returns:
            Tuple[bytes, dict]: Generated image and metadata
            
        Raises:
            Exception: If the image generation fails
        """
        if not self.stability_api_key:
            raise ValueError("Stability AI API key not found in environment variables")
        
        headers = {
            'Authorization': f'Bearer {self.stability_api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        # Configure style parameters based on tattoo style
        style_presets = {
            "traditional": "analog-film",
            "realistic": "photographic",
            "watercolor": "watercolor",
            "japanese": "anime",
            "minimalist": "line-art"
        }
        
        preset = style_presets.get(style.lower(), "tattoo")
        
        data = {
            'text_prompts': [
                {'text': prompt, 'weight': 1.0},
                {'text': 'blurry, low quality, distorted, watermark', 'weight': -1.0}
            ],
            'cfg_scale': 7.0,
            'height': 1024,
            'width': 1024,
            'samples': 1,
            'steps': 50,
            'style_preset': preset
        }
        
        response = await asyncio.to_thread(
            requests.post, 
            'https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image', 
            headers=headers, 
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            # Get base64 image
            image_data = result['artifacts'][0]['base64']
            image_bytes = base64.b64decode(image_data)
            
            metadata = {
                'model': 'stable-diffusion-xl',
                'prompt': prompt,
                'width': 1024,
                'height': 1024,
                'style_preset': preset
            }
            return image_bytes, metadata
        else:
            raise Exception(f"Stability AI image generation failed: {response.text}")
    
    async def _generate_with_sagemaker(self, prompt: str, style: str, 
                                  size: str, color: bool) -> Tuple[bytes, dict]:
        """
        Generate image using AWS SageMaker endpoint.
        
        Args:
            prompt (str): The formatted prompt
            style (str): Tattoo style
            size (str): Size category
            color (bool): Whether color is desired
            
        Returns:
            Tuple[bytes, dict]: Generated image and metadata
            
        Raises:
            Exception: If the image generation fails
        """
        if not self.use_aws_sagemaker or not self.sagemaker_endpoint:
            raise ValueError("SageMaker is not configured")
        
        # Prepare input payload for the SageMaker endpoint
        payload = {
            'prompt': prompt,
            'style': style,
            'size': size,
            'color': color
        }
        
        # Convert payload to JSON
        payload_bytes = json.dumps(payload).encode('utf-8')
        
        # Invoke the SageMaker endpoint
        response = await asyncio.to_thread(
            self.sagemaker_runtime.invoke_endpoint,
            EndpointName=self.sagemaker_endpoint,
            ContentType='application/json',
            Body=payload_bytes
        )
        
        # Process the response
        response_body = response['Body'].read()
        result = json.loads(response_body)
        
        if 'image' in result:
            # Decode base64 image
            image_bytes = base64.b64decode(result['image'])
            metadata = {
                'model': 'sagemaker-custom',
                'prompt': prompt,
                'endpoint': self.sagemaker_endpoint
            }
            return image_bytes, metadata
        else:
            raise Exception(f"SageMaker image generation failed: {result.get('error', 'Unknown error')}")
    
    async def _generate_with_local_model(self, prompt: str, style: str, 
                                    size: str, color: bool) -> Tuple[bytes, dict]:
        """
        Generate image using local ML model.
        
        Args:
            prompt (str): The formatted prompt
            style (str): Tattoo style
            size (str): Size category
            color (bool): Whether color is desired
            
        Returns:
            Tuple[bytes, dict]: Generated image and metadata
            
        Raises:
            Exception: If the image generation fails
        """
        if not self.use_local_models:
            raise ValueError("Local models are not configured")
        
        # This is a placeholder for local model inference
        # In a real implementation, you would run inference on a local ML model
        
        # For this example, we'll just load a placeholder image
        try:
            placeholder_path = self.models_path / "placeholders" / f"{style.lower()}_sample.png"
            
            if not placeholder_path.exists():
                # If style-specific placeholder doesn't exist, use default
                placeholder_path = self.models_path / "placeholders" / "default_sample.png"
            
            if not placeholder_path.exists():
                raise FileNotFoundError(f"No placeholder image found at {placeholder_path}")
            
            with open(placeholder_path, "rb") as f:
                image_bytes = f.read()
            
            metadata = {
                'model': 'local-placeholder',
                'prompt': prompt,
                'path': str(placeholder_path)
            }
            return image_bytes, metadata
            
        except Exception as e:
            raise Exception(f"Local model generation failed: {str(e)}")
    
    @handle_errors()
    async def modify_tattoo_image(self, image_bytes: bytes, modification: str,
                             intensity: float = 0.5) -> Tuple[bytes, dict]:
        """
        Apply modifications to an existing tattoo design.
        
        Args:
            image_bytes (bytes): The original image bytes
            modification (str): Type of modification (rotate, resize, recolor, etc.)
            intensity (float): The intensity of the modification (0.0 to 1.0)
            
        Returns:
            Tuple[bytes, dict]: Modified image and metadata
            
        Raises:
            ModelError: If the modification fails
        """
        try:
            image = Image.open(BytesIO(image_bytes))
            metadata = {'original_size': image.size, 'modification': modification, 'intensity': intensity}
            
            # Apply different modifications based on the requested type
            if modification == "rotate":
                angle = intensity * 360  # Convert intensity to angle (0-360)
                image = image.rotate(angle, expand=True)
                metadata['angle'] = angle
                
            elif modification == "resize":
                scale = 0.5 + intensity  # Scale between 0.5x and 1.5x
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size, Image.LANCZOS)
                metadata['scale'] = scale
                metadata['new_size'] = new_size
                
            elif modification == "recolor":
                # Simple recolor by adjusting hue
                from PIL import ImageEnhance, ImageOps
                
                # Adjust contrast
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.0 + intensity)
                
                # Adjust color balance
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.0 + intensity)
                
                metadata['color_adjustment'] = 1.0 + intensity
            
            else:
                raise ValueError(f"Unsupported modification: {modification}")
            
            # Convert back to bytes
            output = BytesIO()
            image.save(output, format="PNG")
            output.seek(0)
            
            return output.getvalue(), metadata
            
        except Exception as e:
            raise ModelError(f"Failed to modify tattoo image: {str(e)}")
    
    @handle_errors()
    async def mix_tattoo_styles(self, description: str, style1: str, style2: str,
                           blend_ratio: float = 0.5) -> Tuple[bytes, dict]:
        """
        Generate a tattoo that blends two different styles.
        
        Args:
            description (str): The tattoo description
            style1 (str): First tattoo style
            style2 (str): Second tattoo style
            blend_ratio (float): Ratio of style1 to style2 (0.0 to 1.0)
            
        Returns:
            Tuple[bytes, dict]: Generated image and metadata
            
        Raises:
            ModelError: If generation fails
        """
        # Create a blended prompt that emphasizes both styles
        blended_prompt = (
            f"A tattoo design of {description} that combines {style1} and {style2} tattoo styles. "
            f"The design should be {int(blend_ratio * 100)}% {style1} style and "
            f"{int((1 - blend_ratio) * 100)}% {style2} style."
        )
        
        # Use OpenAI for style blending as it handles complex prompts better
        try:
            image_bytes, metadata = await self._generate_with_openai(blended_prompt, "mixed", "medium", True)
            
            # Add style blend information to metadata
            metadata.update({
                'style1': style1,
                'style2': style2,
                'blend_ratio': blend_ratio,
                'blended_prompt': blended_prompt
            })
            
            return image_bytes, metadata
            
        except Exception as e:
            # Try another service if OpenAI fails
            try:
                image_bytes, metadata = await self._generate_with_stability_ai(blended_prompt, "mixed", "medium", True)
                
                # Add style blend information to metadata
                metadata.update({
                    'style1': style1,
                    'style2': style2,
                    'blend_ratio': blend_ratio,
                    'blended_prompt': blended_prompt
                })
                
                return image_bytes, metadata
                
            except Exception as e2:
                raise ModelError(
                    f"Failed to blend tattoo styles: Primary error: {str(e)}, Fallback error: {str(e2)}"
                )
    
    def clear_cache(self):
        """Clear the internal model cache."""
        cache_size = len(self.cache)
        self.cache = {}
        logger.info(f"Cleared model cache ({cache_size} items)")
        return cache_size


# Create a singleton instance
model_handler = ModelHandler()

def get_model_handler() -> ModelHandler:
    """
    Get the singleton model handler instance.
    
    Returns:
        ModelHandler: The model handler instance
    """
    return model_handler


async def generate_tattoo_image(description: str, **kwargs) -> bytes:
    """
    Convenience function to generate a tattoo image.
    
    Args:
        description (str): The tattoo description
        **kwargs: Additional parameters for generation
        
    Returns:
        bytes: The generated image
        
    Raises:
        Exception: If generation fails
    """
    handler = get_model_handler()
    image_bytes, _ = await handler.generate_tattoo_image(description, **kwargs)
    return image_bytes