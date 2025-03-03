import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from ..config.model_config import ModelConfig

class ModelHandler:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.models: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
    async def load_model(self, model_name: str) -> bool:
        try:
            model_path = self.config.MODEL_PATH / f"{model_name}.pth"
            if not model_path.exists():
                self.logger.error(f"Model {model_name} not found at {model_path}")
                return False
                
            model = torch.jit.load(model_path)
            model.to(self.device)
            
            if self.config.USE_QUANTIZATION:
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                
            self.models[model_name] = model
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            return False
            
    async def unload_model(self, model_name: str) -> bool:
        if model_name in self.models:
            del self.models[model_name]
            torch.cuda.empty_cache()
            return True
        return False
        
    async def predict(self, model_name: str, input_data: torch.Tensor) -> Optional[torch.Tensor]:
        if model_name not in self.models:
            self.logger.error(f"Model {model_name} not loaded")
            return None
            
        try:
            with torch.no_grad():
                output = self.models[model_name](input_data.to(self.device))
            return output
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return None