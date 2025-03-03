import os
from typing import Any, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

class EnvManager:
    def __init__(self):
        self.env_file = Path("c:/devdrive/thInk/.env")
        self._load_env()
        
    def _load_env(self):
        """Load environment variables from .env file"""
        if self.env_file.exists():
            load_dotenv(self.env_file)
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get environment variable value"""
        return os.getenv(key, default)
        
    def set(self, key: str, value: str):
        """Set environment variable"""
        os.environ[key] = value
        self._update_env_file(key, value)
        
    def _update_env_file(self, key: str, value: str):
        """Update .env file with new value"""
        current_content = {}
        if self.env_file.exists():
            with open(self.env_file) as f:
                for line in f:
                    if '=' in line:
                        k, v = line.strip().split('=', 1)
                        current_content[k] = v
                        
        current_content[key] = value
        
        with open(self.env_file, 'w') as f:
            for k, v in current_content.items():
                f.write(f"{k}={v}\n")
                
    def get_all(self) -> Dict[str, str]:
        """Get all environment variables for the application"""
        return {
            key: value for key, value in os.environ.items()
            if key.startswith('THINK_')
        }