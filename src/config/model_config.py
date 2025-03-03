from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class ModelConfig:
    # AI Model Settings
    MODEL_PATH: Path = Path("c:/devdrive/thInk/models")
    DEVICE: str = "cuda"  # or "cpu"
    BATCH_SIZE: int = 1
    
    # AR Settings
    MIN_CONFIDENCE: float = 0.7
    TRACKING_FREQUENCY: int = 30
    RENDER_RESOLUTION: tuple = (1920, 1080)
    
    # Privacy Settings
    DATA_RETENTION_DAYS: int = 30
    ANONYMIZATION_ENABLED: bool = True
    
    # Performance Settings
    USE_QUANTIZATION: bool = True
    USE_MOBILE_OPTIMIZED: bool = True
    MAX_MEMORY_MB: int = 2048

@dataclass
class PrivacyConfig:
    RETENTION_POLICIES: Dict[str, int] = None
    ALLOWED_DATA_USES: List[str] = None
    OPT_OUT_FEATURES: List[str] = None
    
    def __post_init__(self):
        self.RETENTION_POLICIES = {
            "user_data": 365,  # days
            "generated_tattoos": 90,
            "analytics": 30,
            "session_data": 7
        }
        
        self.ALLOWED_DATA_USES = [
            "service_improvement",
            "model_training",
            "analytics"
        ]
        
        self.OPT_OUT_FEATURES = [
            "model_training",
            "analytics",
            "marketing"
        ]

@dataclass
class SubscriptionConfig:
    TIERS: Dict[str, Dict] = None
    
    def __post_init__(self):
        self.TIERS = {
            "free": {
                "generations_per_day": 3,
                "resolution": "standard",
                "features": ["basic_ar", "basic_generation"]
            },
            "premium": {
                "generations_per_day": 20,
                "resolution": "high",
                "features": ["advanced_ar", "priority_generation", "style_transfer"]
            },
            "professional": {
                "generations_per_day": -1,  # unlimited
                "resolution": "ultra",
                "features": ["all"]
            }
        }