"""
Tattoo generation module for the Th.ink AR application.

This module provides functionality to generate custom tattoo designs using 
AI models and traditional image manipulation techniques, with support for 
different tattoo styles and customization options.
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple, List, Any, Union
import cv2
import asyncio
import time
import logging
from pathlib import Path
import uuid
import os
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

from .model_handler import get_model_handler
from ..config.model_config import get_config
from ..utils.performance_monitor import measure_time
from ..errors.error_handler import handle_errors, ModelError


# Configure logger
logger = logging.getLogger(__name__)


class TattooStyle:
    """Predefined tattoo styles with their parameters."""
    
    TRADITIONAL = "traditional"
    REALISTIC = "realistic"
    BLACKWORK = "blackwork"
    WATERCOLOR = "watercolor"
    TRIBAL = "tribal"
    JAPANESE = "japanese"
    NEW_SCHOOL = "new_school"
    MINIMALIST = "minimalist"
    GEOMETRIC = "geometric"
    DOTWORK = "dotwork"

    @staticmethod
    def get_all_styles() -> List[str]:
        """Get all available styles."""
        return [
            TattooStyle.TRADITIONAL, 
            TattooStyle.REALISTIC, 
            TattooStyle.BLACKWORK,
            TattooStyle.WATERCOLOR, 
            TattooStyle.TRIBAL, 
            TattooStyle.JAPANESE,
            TattooStyle.NEW_SCHOOL, 
            TattooStyle.MINIMALIST,
            TattooStyle.GEOMETRIC,
            TattooStyle.DOTWORK
        ]

    @staticmethod
    def get_style_description(style: str) -> str:
        """Get description for a specific style."""
        descriptions = {
            TattooStyle.TRADITIONAL: "Bold black outlines with limited color palette",
            TattooStyle.REALISTIC: "Photorealistic detail with shading and depth",
            TattooStyle.BLACKWORK: "Solid black ink with intricate patterns",
            TattooStyle.WATERCOLOR: "Soft color blending with paint splatter effects",
            TattooStyle.TRIBAL: "Bold black tribal patterns with cultural influences",
            TattooStyle.JAPANESE: "Traditional Japanese iconography and techniques",
            TattooStyle.NEW_SCHOOL: "Cartoonish style with exaggerated proportions and vibrant colors",
            TattooStyle.MINIMALIST: "Clean, simple lines with minimal detail",
            TattooStyle.GEOMETRIC: "Precise geometric shapes and patterns",
            TattooStyle.DOTWORK: "Intricate patterns of tiny dots creating detailed shading"
        }
        return descriptions.get(style, "Custom tattoo style")


class TattooGenerator:
    """
    AI-powered tattoo design generator with style customization.
    
    This class provides methods to generate and customize tattoo designs
    using various AI models and image processing techniques.
    """
    
    def __init__(self, model_handler=None, config=None):
        """
        Initialize the tattoo generator.
        
        Args:
            model_handler: Optional model handler, or None to use default
            config: Optional configuration, or None to use default
        """
        self.model_handler = model_handler or get_model_handler()
        self.config = config or get_config()
        self.output_dir = Path(self.config.ai.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories by style
        for style in TattooStyle.get_all_styles():
            style_dir = self.output_dir / style
            style_dir.mkdir(exist_ok=True)
            
        # Generation statistics
        self.generation_count = 0
        self.total_generation_time = 0
        
        logger.info("Tattoo generator initialized")
    
    @measure_time("tattoo_generation")
    @handle_errors()
    async def generate_tattoo(
        self,
        prompt: str,
        style: str = TattooStyle.TRADITIONAL,
        size: Tuple[int, int] = (512, 512),
        color: bool = True,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate a tattoo design based on the provided prompt and style.
        
        Args:
            prompt: Text description of the desired tattoo
            style: Tattoo style (traditional, realistic, etc.)
            size: Output image size (width, height)
            color: Whether to generate a color tattoo
            options: Additional generation options
            
        Returns:
            Tuple of (generated image as numpy array, metadata)
            
        Raises:
            ModelError: If tattoo generation fails
        """
        start_time = time.time()
        options = options or {}
        
        try:
            # Load the diffusion model if not already loaded
            model_name = "tattoo_generator"
            if model_name not in self.model_handler.models:
                logger.info(f"Loading {model_name} model")
                await self.model_handler.load_model(model_name)
            
            # Enhanced prompt with style-specific terminology
            enhanced_prompt = self._enhance_prompt(prompt, style, color)
            
            # Prepare model inputs
            inputs = self._prepare_model_inputs(enhanced_prompt, style, size, color, options)
            
            # Generate the base design
            raw_image = await self._generate_base_design(inputs, model_name)
            
            # Post-process the image according to style
            processed_image = self._post_process_by_style(raw_image, style, color, options)
            
            # Apply any additional effects requested in options
            final_image = self._apply_effects(processed_image, options)
            
            # Save the generated image if configured
            output_path = self._save_generated_image(final_image, style, prompt)
            
            # Update statistics
            self.generation_count += 1
            generation_time = time.time() - start_time
            self.total_generation_time += generation_time
            
            # Prepare metadata
            metadata = {
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "style": style,
                "size": size,
                "color": color,
                "generation_time": generation_time,
                "output_path": str(output_path) if output_path else None,
                "model": model_name,
                "generation_id": str(uuid.uuid4())
            }
            
            return final_image, metadata
            
        except Exception as e:
            error_msg = f"Tattoo generation failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg)
    
    def _enhance_prompt(self, prompt: str, style: str, color: bool) -> str:
        """
        Enhance the user prompt with style-specific terminology.
        
        Args:
            prompt: User's original prompt
            style: Tattoo style
            color: Whether generating a color tattoo
            
        Returns:
            Enhanced prompt for better generation results
        """
        style_modifiers = {
            TattooStyle.TRADITIONAL: (
                "traditional American tattoo with bold black outlines, "
                "limited color palette, vibrant, clean"
            ),
            TattooStyle.REALISTIC: (
                "photorealistic tattoo with detailed shading, "
                "depth and dimension, skin texture"
            ),
            TattooStyle.BLACKWORK: (
                "blackwork tattoo with solid black ink, intricate patterns, "
                "high contrast, bold"
            ),
            TattooStyle.WATERCOLOR: (
                "watercolor tattoo with soft color blending, splatter effects, "
                "flowing, artistic, no outlines"
            ),
            TattooStyle.TRIBAL: (
                "tribal tattoo with bold black patterns, "
                "symmetrical, cultural, symbolic"
            ),
            TattooStyle.JAPANESE: (
                "traditional Japanese irezumi tattoo with bold outlines, "
                "waves, clouds, cultural iconography"
            ),
            TattooStyle.NEW_SCHOOL: (
                "new school tattoo with exaggerated proportions, "
                "cartoonish, vibrant colors, bold outlines"
            ),
            TattooStyle.MINIMALIST: (
                "minimalist tattoo with simple clean lines, "
                "minimal detail, elegant, subtle"
            ),
            TattooStyle.GEOMETRIC: (
                "geometric tattoo with precise shapes, "
                "mathematical patterns, symmetrical, structured"
            ),
            TattooStyle.DOTWORK: (
                "dotwork tattoo composed of tiny dots, "
                "stippling technique, intricate, textured"
            )
        }
        
        # Get style-specific modifiers
        style_prompt = style_modifiers.get(style, f"{style} style tattoo")
        
        # Color modifiers
        color_prompt = "with vibrant colors" if color else "in black and gray only, monochrome"
        
        # Additional tattoo-specific terms
        tattoo_terms = "high contrast, suitable for tattooing on skin, clean background"
        
        # Combine everything
        return f"A tattoo design of {prompt}. {style_prompt}, {color_prompt}. {tattoo_terms}"
    
    def _prepare_model_inputs(
        self, 
        prompt: str, 
        style: str, 
        size: Tuple[int, int],
        color: bool,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare inputs for the AI model.
        
        Args:
            prompt: Enhanced prompt
            style: Tattoo style
            size: Output size
            color: Whether color is enabled
            options: Additional options
            
        Returns:
            Dictionary of model inputs
        """
        # Get guidance scale based on style
        guidance_scale = options.get('guidance_scale', 7.5)
        if style == TattooStyle.WATERCOLOR:
            guidance_scale = 6.5  # Less guidance for more creative styles
        elif style == TattooStyle.REALISTIC:
            guidance_scale = 8.5  # More guidance for realistic styles
        
        # Get number of inference steps
        steps = options.get('steps', 50)
        
        # Negative prompt to avoid common issues
        negative_prompt = (
            "blurry, low quality, distorted, unrealistic, pixelated, "
            "text, watermark, signature, frame, border, cropped, deformed"
        )
        
        # Prepare seed for reproducibility
        seed = options.get('seed', np.random.randint(0, 2**32 - 1))
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": steps,
            "height": size[1],
            "width": size[0],
            "seed": seed,
            "style": style,
            "color": color
        }
    
    async def _generate_base_design(self, inputs: Dict[str, Any], model_name: str) -> np.ndarray:
        """
        Generate the base tattoo design using the AI model.
        
        Args:
            inputs: Model inputs
            model_name: Name of the model to use
            
        Returns:
            Raw generated image as numpy array
            
        Raises:
            ModelError: If generation fails
        """
        try:
            # Create cache key
            cache_key = f"{inputs['prompt']}_{inputs['style']}_{inputs['seed']}"
            
            # Get model prediction
            output = await self.model_handler.predict(model_name, inputs, cache_key)
            
            if output is None:
                raise ModelError("Model returned None output")
            
            # For diffusion models, the output might be a tensor or a PIL Image
            if isinstance(output, torch.Tensor):
                # Convert tensor to numpy array
                image = output.cpu().permute(0, 2, 3, 1).numpy()[0]
                
                # Normalize to 0-255 range
                image = ((image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                
            elif isinstance(output, Image.Image):
                # Convert PIL Image to numpy array
                image = np.array(output)
                
            else:
                # Handle other types (e.g., raw numpy array)
                image = output
                
            return image
            
        except Exception as e:
            raise ModelError(f"Base design generation failed: {str(e)}")
    
    def _post_process_by_style(
        self, 
        image: np.ndarray, 
        style: str, 
        color: bool,
        options: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply style-specific post-processing to the generated image.
        
        Args:
            image: Raw generated image
            style: Tattoo style
            color: Whether color is enabled
            options: Additional options
            
        Returns:
            Post-processed image
        """
        # Convert to PIL for easier processing
        pil_image = Image.fromarray(image)
        
        # Apply style-specific enhancements
        if style == TattooStyle.TRADITIONAL:
            # Traditional: bold outlines, limited colors
            pil_image = self._enhance_traditional(pil_image, color)
            
        elif style == TattooStyle.BLACKWORK:
            # Blackwork: high contrast black
            pil_image = self._enhance_blackwork(pil_image)
            
        elif style == TattooStyle.WATERCOLOR:
            # Watercolor: soft edges, paint splatter
            pil_image = self._enhance_watercolor(pil_image, color)
            
        elif style == TattooStyle.MINIMALIST:
            # Minimalist: clean lines, reduce detail
            pil_image = self._enhance_minimalist(pil_image)
        
        elif style == TattooStyle.DOTWORK:
            # Dotwork: convert to dot patterns
            pil_image = self._enhance_dotwork(pil_image)
            
        # Process the color based on the color flag
        if not color and style != TattooStyle.BLACKWORK:
            # Convert to grayscale for non-color designs
            pil_image = ImageOps.grayscale(pil_image)
            # Convert back to RGB for consistency
            pil_image = Image.merge('RGB', [pil_image, pil_image, pil_image])
            
        # Apply universal enhancements
        pil_image = self._apply_universal_enhancements(pil_image, options)
            
        # Convert back to numpy
        return np.array(pil_image)
    
    def _enhance_traditional(self, image: Image.Image, color: bool) -> Image.Image:
        """
        Enhance image for traditional tattoo style.
        
        Args:
            image: Input PIL image
            color: Whether color is enabled
            
        Returns:
            Enhanced image
        """
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Sharpen
        image = image.filter(ImageFilter.SHARPEN)
        
        # Enhance edges to simulate bold outlines
        edges = image.filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.invert(edges)
        
        # Merge with original
        r, g, b = image.split()
        edges_gray = edges.convert('L')
        edges_gray = ImageOps.invert(edges_gray)
        
        # Threshold edges to get bold outlines
        from PIL import ImageOps
        edges_gray = edges_gray.point(lambda x: 0 if x > 128 else 255)
        
        # Apply bold outlines
        result = Image.merge('RGB', [
            Image.blend(r, edges_gray.convert('L'), 0.3),
            Image.blend(g, edges_gray.convert('L'), 0.3),
            Image.blend(b, edges_gray.convert('L'), 0.3)
        ])
        
        if color:
            # Enhance colors for traditional look
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(1.5)
            
            # Reduce color palette for traditional look
            result = result.quantize(colors=8).convert('RGB')
        
        return result
    
    def _enhance_blackwork(self, image: Image.Image) -> Image.Image:
        """
        Enhance image for blackwork tattoo style.
        
        Args:
            image: Input PIL image
            
        Returns:
            Enhanced image
        """
        # Convert to grayscale
        image = ImageOps.grayscale(image)
        
        # Increase contrast dramatically
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Further increase contrast with curves
        def curve(x):
            # Increase black and white contrast
            return 0 if x < 128 else 255
        
        # Apply curve for high contrast black
        image = image.point(curve)
        
        # Convert back to RGB
        return Image.merge('RGB', [image, image, image])
    
    def _enhance_watercolor(self, image: Image.Image, color: bool) -> Image.Image:
        """
        Enhance image for watercolor tattoo style.
        
        Args:
            image: Input PIL image
            color: Whether color is enabled
            
        Returns:
            Enhanced image
        """
        # Soft blur for watercolor effect
        image = image.filter(ImageFilter.GaussianBlur(radius=1))
        
        if color:
            # Enhance colors for watercolor look
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.3)
            
            # Add slight color bleeding
            r, g, b = image.split()
            r = r.filter(ImageFilter.GaussianBlur(radius=0.7))
            g = g.filter(ImageFilter.GaussianBlur(radius=0.8))
            b = b.filter(ImageFilter.GaussianBlur(radius=0.9))
            
            image = Image.merge('RGB', [r, g, b])
        
        # Reduce sharpness for soft edges
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(0.7)
        
        return image
    
    def _enhance_minimalist(self, image: Image.Image) -> Image.Image:
        """
        Enhance image for minimalist tattoo style.
        
        Args:
            image: Input PIL image
            
        Returns:
            Enhanced image
        """
        # Convert to grayscale
        image = ImageOps.grayscale(image)
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.8)
        
        # Find edges for line art
        edges = image.filter(ImageFilter.FIND_EDGES)
        
        # Threshold to create clean lines
        def threshold(x):
            return 255 if x > 100 else 0
        
        edges = edges.point(threshold)
        
        # Invert to get black lines on white background
        edges = ImageOps.invert(edges)
        
        # Convert back to RGB
        return Image.merge('RGB', [edges, edges, edges])
    
    def _enhance_dotwork(self, image: Image.Image) -> Image.Image:
        """
        Enhance image for dotwork tattoo style.
        
        Args:
            image: Input PIL image
            
        Returns:
            Enhanced image with dot pattern
        """
        # Convert to grayscale
        gray = ImageOps.grayscale(image)
        
        # Create new blank image
        dotwork = Image.new('L', gray.size, 255)
        
        # Create dot pattern
        width, height = gray.size
        gray_data = gray.load()
        dotwork_data = dotwork.load()
        
        # Dot spacing depends on brightness
        dot_size = 3
        spacing = 5
        
        for y in range(0, height, spacing):
            for x in range(0, width, spacing):
                if x < width and y < height:
                    # Get brightness value
                    brightness = gray_data[x, y]
                    
                    # Calculate dot size based on brightness
                    # Darker areas get larger dots
                    actual_dot_size = max(1, int(dot_size * (255 - brightness) / 255))
                    
                    # Draw dot
                    for dy in range(-actual_dot_size, actual_dot_size + 1):
                        for dx in range(-actual_dot_size, actual_dot_size + 1):
                            dot_x, dot_y = x + dx, y + dy
                            if (0 <= dot_x < width and 0 <= dot_y < height and
                                dx*dx + dy*dy <= actual_dot_size*actual_dot_size):
                                dotwork_data[dot_x, dot_y] = 0
        
        # Convert back to RGB
        return Image.merge('RGB', [dotwork, dotwork, dotwork])
    
    def _apply_universal_enhancements(self, image: Image.Image, options: Dict[str, Any]) -> Image.Image:
        """
        Apply enhancements that work well for all tattoo styles.
        
        Args:
            image: Input PIL image
            options: Additional options
            
        Returns:
            Enhanced image
        """
        # Adjust contrast
        contrast = options.get('contrast', 1.2)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        
        # Sharpen slightly
        if options.get('sharpen', True):
            image = image.filter(ImageFilter.SHARPEN)
        
        # Adjust brightness
        brightness = options.get('brightness', 1.1)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
        
        # Remove background if requested and not already transparent
        if options.get('transparent_background', False) and image.mode != 'RGBA':
            # Convert to RGBA
            image = image.convert('RGBA')
            
            # Create a white background mask
            gray = ImageOps.grayscale(image)
            threshold = 240
            mask = gray.point(lambda x: 0 if x > threshold else 255, 'L')
            
            # Apply the mask
            image.putalpha(mask)
        
        return image
    
    def _apply_effects(self, image: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Apply additional special effects to the image.
        
        Args:
            image: Input image as numpy array
            options: Effect options
            
        Returns:
            Processed image
        """
        # Handle special effects
        effects = options.get('effects', [])
        
        if not effects:
            return image
        
        # Convert to PIL for easier processing
        pil_image = Image.fromarray(image)
        
        for effect in effects:
            effect_type = effect.get('type', '')
            intensity = effect.get('intensity', 1.0)
            
            if effect_type == 'aged':
                # Add aged paper texture effect
                pil_image = self._apply_aged_effect(pil_image, intensity)
                
            elif effect_type == 'sketch':
                # Make it look hand-sketched
                pil_image = self._apply_sketch_effect(pil_image, intensity)
                
            elif effect_type == 'distressed':
                # Add distressed/worn look
                pil_image = self._apply_distressed_effect(pil_image, intensity)
        
        # Convert back to numpy
        return np.array(pil_image)
    
    def _apply_aged_effect(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply aged paper texture effect."""
        # Sepia tone
        sepia = Image.new('RGB', image.size, (255, 240, 192))
        image = Image.blend(image, sepia, intensity * 0.3)
        
        # Add noise
        noise = Image.effect_noise(image.size, 10)
        noise = noise.convert('RGB')
        image = Image.blend(image, noise, intensity * 0.1)
        
        return image
    
    def _apply_sketch_effect(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply hand-sketched effect."""
        # Convert to grayscale
        gray = ImageOps.grayscale(image)
        
        # Edge enhance
        edges = gray.filter(ImageFilter.EDGE_ENHANCE_MORE)
        edges = edges.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        # Adjust contrast
        enhancer = ImageEnhance.Contrast(edges)
        sketch = enhancer.enhance(1.5)
        
        # Blend with original based on intensity
        result = Image.blend(image, sketch.convert('RGB'), intensity)
        
        return result
    
    def _apply_distressed_effect(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply distressed/worn look."""
        # Add scratches
        width, height = image.size
        scratches = Image.new('RGB', image.size, (255, 255, 255))
        
        # Draw random scratches
        from PIL import ImageDraw
        draw = ImageDraw.Draw(scratches)
        
        num_scratches = int(30 * intensity)
        for _ in range(num_scratches):
            # Random scratch line
            x1 = np.random.randint(0, width)
            y1 = np.random.randint(0, height)
            x2 = x1 + np.random.randint(-100, 100)
            y2 = y1 + np.random.randint(-100, 100)
            
            draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=1)
        
        # Blend with original
        result = Image.blend(image, scratches, intensity * 0.3)
        
        return result
    
    def _save_generated_image(self, image: np.ndarray, style: str, prompt: str) -> Optional[Path]:
        """
        Save the generated image to disk if configured.
        
        Args:
            image: Generated image
            style: Tattoo style
            prompt: Original prompt
            
        Returns:
            Path where the image was saved, or None if not saved
        """
        if not self.config.ai.save_generations:
            return None
        
        # Create a safe filename from the prompt
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)
        safe_prompt = safe_prompt[:30]  # Limit length
        
        # Create a unique filename
        timestamp = int(time.time())
        filename = f"{safe_prompt}_{timestamp}.png"
        
        # Save to style-specific directory
        style_dir = self.output_dir / style
        output_path = style_dir / filename
        
        # Convert numpy array to PIL and save
        Image.fromarray(image).save(output_path)
        
        logger.info(f"Saved generated image to {output_path}")
        return output_path
    
    @handle_errors()
    async def adjust_design(
        self, 
        design: np.ndarray,
        target_size: Tuple[int, int],
        adjustments: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Adjust an existing tattoo design with various transformations.
        
        Args:
            design: Original tattoo design
            target_size: Desired output size
            adjustments: Adjustment parameters
            
        Returns:
            Adjusted design
            
        Raises:
            ModelError: If adjustment fails
        """
        if design is None:
            raise ModelError("Design is None")
        
        adjustments = adjustments or {}
        
        try:
            # Convert to RGBA if needed
            if design.shape[2] == 3:
                rgba = np.zeros((design.shape[0], design.shape[1], 4), dtype=np.uint8)
                rgba[:,:,:3] = design
                rgba[:,:,3] = 255
                design = rgba
            
            # Apply adjustments
            design = self._apply_adjustments(design, adjustments)
            
            # Resize to target size
            return cv2.resize(design, target_size, interpolation=cv2.INTER_LANCZOS4)
            
        except Exception as e:
            error_msg = f"Design adjustment failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg)
    
    def _apply_adjustments(self, design: np.ndarray, adjustments: Dict[str, Any]) -> np.ndarray:
        """
        Apply various adjustments to the design.
        
        Args:
            design: Original design
            adjustments: Adjustment parameters
            
        Returns:
            Adjusted design
        """
        # Convert to PIL for easier processing
        design_pil = Image.fromarray(design)
        
        # Apply rotation
        if 'rotation' in adjustments:
            angle = adjustments['rotation']
            design_pil = design_pil.rotate(angle, expand=True, resample=Image.BICUBIC)
        
        # Apply scaling
        if 'scale' in adjustments:
            scale = adjustments['scale']
            width, height = design_pil.size
            new_size = (int(width * scale), int(height * scale))
            design_pil = design_pil.resize(new_size, Image.LANCZOS)
        
        # Apply color adjustments
        if 'color_adjustments' in adjustments:
            color_adj = adjustments['color_adjustments']
            
            # Brightness
            if 'brightness' in color_adj:
                enhancer = ImageEnhance.Brightness(design_pil)
                design_pil = enhancer.enhance(color_adj['brightness'])
            
            # Contrast
            if 'contrast' in color_adj:
                enhancer = ImageEnhance.Contrast(design_pil)
                design_pil = enhancer.enhance(color_adj['contrast'])
            
            # Saturation
            if 'saturation' in color_adj:
                enhancer = ImageEnhance.Color(design_pil)
                design_pil = enhancer.enhance(color_adj['saturation'])
            
            # Hue shift (requires more complex processing)
            if 'hue_shift' in color_adj:
                # Convert to HSV, adjust hue, convert back
                design_np = np.array(design_pil)
                hsv = cv2.cvtColor(design_np, cv2.COLOR_RGB2HSV)
                hsv[:,:,0] = (hsv[:,:,0] + color_adj['hue_shift']) % 180
                design_np = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                design_pil = Image.fromarray(design_np)
        
        # Apply filters
        if 'filter' in adjustments:
            filter_type = adjustments['filter']
            
            if filter_type == 'sharpen':
                design_pil = design_pil.filter(ImageFilter.SHARPEN)
            elif filter_type == 'blur':
                design_pil = design_pil.filter(ImageFilter.GaussianBlur(radius=1))
            elif filter_type == 'edge_enhance':
                design_pil = design_pil.filter(ImageFilter.EDGE_ENHANCE)
            elif filter_type == 'edge_enhance':
                design_pil = design_pil.filter(ImageFilter.EDGE_ENHANCE)
            elif filter_type == 'contour':
                design_pil = design_pil.filter(ImageFilter.CONTOUR)
            elif filter_type == 'emboss':
                design_pil = design_pil.filter(ImageFilter.EMBOSS)
        
        # Apply transparency adjustments
        if 'transparency' in adjustments:
            transparency = adjustments['transparency']
            
            # Convert to RGBA if not already
            if design_pil.mode != 'RGBA':
                design_pil = design_pil.convert('RGBA')
                
            # Get the alpha channel
            r, g, b, a = design_pil.split()
            
            # Adjust alpha based on transparency value (0-1)
            a = a.point(lambda x: int(x * (1 - transparency)))
            
            # Merge channels back
            design_pil = Image.merge('RGBA', (r, g, b, a))
        
        # Convert back to numpy array
        return np.array(design_pil)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get generation statistics.
        
        Returns:
            Dictionary of generation statistics
        """
        return {
            "total_generations": self.generation_count,
            "total_generation_time": self.total_generation_time,
            "average_generation_time": self.total_generation_time / max(1, self.generation_count),
            "output_directory": str(self.output_dir)
        }
    
    @handle_errors()
    async def blend_designs(
        self, 
        design1: np.ndarray, 
        design2: np.ndarray, 
        blend_ratio: float = 0.5,
        blend_mode: str = 'normal'
    ) -> np.ndarray:
        """
        Blend two tattoo designs.
        
        Args:
            design1: First design
            design2: Second design
            blend_ratio: Ratio of first design to second design (0.0 to 1.0)
            blend_mode: Blending mode (normal, multiply, screen, overlay)
            
        Returns:
            Blended design
            
        Raises:
            ModelError: If blending fails
        """
        try:
            # Convert to PIL for blending
            pil1 = Image.fromarray(design1)
            pil2 = Image.fromarray(design2)
            
            # Resize second image to match first
            pil2 = pil2.resize(pil1.size, Image.LANCZOS)
            
            # Convert both to RGBA
            if pil1.mode != 'RGBA':
                pil1 = pil1.convert('RGBA')
            if pil2.mode != 'RGBA':
                pil2 = pil2.convert('RGBA')
            
            # Apply different blending modes
            if blend_mode == 'normal':
                result = Image.blend(pil1, pil2, 1 - blend_ratio)
            elif blend_mode == 'multiply':
                # Simulate multiply blend mode
                np1 = np.array(pil1).astype(float) / 255
                np2 = np.array(pil2).astype(float) / 255
                
                # Blend using the formula: result = img1 * img2
                blend = np1 * np2
                
                # Mix with original based on blend ratio
                result_np = blend_ratio * np1 + (1 - blend_ratio) * blend
                
                # Convert back to 8-bit
                result_np = (result_np * 255).clip(0, 255).astype(np.uint8)
                result = Image.fromarray(result_np)
            elif blend_mode == 'screen':
                # Simulate screen blend mode
                np1 = np.array(pil1).astype(float) / 255
                np2 = np.array(pil2).astype(float) / 255
                
                # Blend using the formula: result = 1 - (1 - img1) * (1 - img2)
                blend = 1 - (1 - np1) * (1 - np2)
                
                # Mix with original based on blend ratio
                result_np = blend_ratio * np1 + (1 - blend_ratio) * blend
                
                # Convert back to 8-bit
                result_np = (result_np * 255).clip(0, 255).astype(np.uint8)
                result = Image.fromarray(result_np)
            elif blend_mode == 'overlay':
                # Simulate overlay blend mode
                np1 = np.array(pil1).astype(float) / 255
                np2 = np.array(pil2).astype(float) / 255
                
                # Overlay formula: result = (img1 < 0.5) ? (2 * img1 * img2) : (1 - 2 * (1 - img1) * (1 - img2))
                mask = np1 < 0.5
                blend = np.zeros_like(np1)
                blend[mask] = 2 * np1[mask] * np2[mask]
                blend[~mask] = 1 - 2 * (1 - np1[~mask]) * (1 - np2[~mask])
                
                # Mix with original based on blend ratio
                result_np = blend_ratio * np1 + (1 - blend_ratio) * blend
                
                # Convert back to 8-bit
                result_np = (result_np * 255).clip(0, 255).astype(np.uint8)
                result = Image.fromarray(result_np)
            else:
                # Default to normal blend
                result = Image.blend(pil1, pil2, 1 - blend_ratio)
            
            # Return as numpy array
            return np.array(result)
            
        except Exception as e:
            error_msg = f"Design blending failed: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg)


# Create a singleton instance
_tattoo_generator = None

def get_tattoo_generator() -> TattooGenerator:
    """
    Get the singleton tattoo generator instance.
    
    Returns:
        TattooGenerator: The tattoo generator instance
    """
    global _tattoo_generator
    if _tattoo_generator is None:
        _tattoo_generator = TattooGenerator()
    return _tattoo_generator