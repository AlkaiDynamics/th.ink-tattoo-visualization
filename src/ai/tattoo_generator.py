import torch
from diffusers import StableDiffusionPipeline
from typing import Optional, Dict, Tuple
import numpy as np
from .model_handler import ModelHandler
from ..config.model_config import ModelConfig

class TattooGenerator:
    def __init__(self, model_handler: ModelHandler):
        self.model_handler = model_handler
        self.pipeline = None
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1"
            ).to(self.model_handler.device)
            
        except Exception as e:
            self.model_handler.logger.error(f"Failed to initialize pipeline: {str(e)}")
            
    async def generate_tattoo(
        self, 
        prompt: str, 
        style: str,
        size: tuple = (512, 512)
    ) -> Optional[torch.Tensor]:
        try:
            full_prompt = f"tattoo design, {style} style, {prompt}"
            image = self.pipeline(
                full_prompt,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]
            
            return image
            
        except Exception as e:
            self.model_handler.logger.error(f"Tattoo generation failed: {str(e)}")
            return None
    def adjust_design(self, design: np.ndarray, 
                     target_size: Tuple[int, int]) -> np.ndarray:
        """Adjust design size and format"""
        if design is None:
            return None
            
        # Convert to RGBA if needed
        if design.shape[2] == 3:
            rgba = np.zeros((design.shape[0], design.shape[1], 4), dtype=np.uint8)
            rgba[:,:,:3] = design
            rgba[:,:,3] = 255
            design = rgba
            
        # Resize to target size
        return cv2.resize(design, target_size, interpolation=cv2.INTER_LANCZOS4)