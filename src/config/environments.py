from enum import Enum
from typing import Dict, Any
import json
import os

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class EnvironmentConfig:
    def __init__(self, env: Environment):
        self.env = env
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(
            "c:/devdrive/thInk/config",
            f"{self.env.value}.json"
        )
        with open(config_path, "r") as f:
            return json.load(f)
    
    @property
    def debug_mode(self) -> bool:
        return self.env != Environment.PRODUCTION
    
    @property
    def api_url(self) -> str:
        return self.config.get("api_url")
    
    @property
    def model_path(self) -> str:
        return self.config.get("model_path")