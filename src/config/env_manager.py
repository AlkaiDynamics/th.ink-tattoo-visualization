"""
Environment variable management for the Th.ink AR application.

This module provides utilities for loading, setting, and accessing environment variables
in a secure and consistent manner, with support for both .env files and OS environment variables.
"""

import os
import re
from typing import Any, Dict, Optional, Set, Union, List, cast
from pathlib import Path
import logging
from dotenv import load_dotenv, find_dotenv

# Configure logger
logger = logging.getLogger("think.config")


class EnvManager:
    """
    Environment variable manager for secure configuration.
    
    This class provides methods to load environment variables from various sources,
    manage their values, and resolve placeholders in configuration values.
    """
    
    def __init__(self, env_file: Optional[Union[str, Path]] = None, auto_load: bool = True):
        """
        Initialize the environment manager.
        
        Args:
            env_file: Path to .env file, or None to search automatically
            auto_load: Whether to automatically load environment variables on init
        """
        if env_file:
            self.env_file = Path(env_file)
        else:
            # Try to find .env file in parent directories
            env_path = find_dotenv(usecwd=True)
            self.env_file = Path(env_path) if env_path else None
        
        self.loaded = False
        if auto_load:
            self.load_env()
    
    def load_env(self) -> bool:
        """
        Load environment variables from .env file and OS environment.
        
        Returns:
            True if environment was loaded successfully, False otherwise
        """
        if self.env_file and self.env_file.exists():
            logger.debug(f"Loading environment from {self.env_file}")
            load_dotenv(self.env_file)
            self.loaded = True
            return True
        else:
            logger.warning("No .env file found, using OS environment variables only")
            self.loaded = True
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get environment variable value.
        
        Args:
            key: Environment variable key
            default: Default value if key is not found
            
        Returns:
            Environment variable value or default
        """
        if not self.loaded:
            self.load_env()
        
        return os.environ.get(key, default)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """
        Get environment variable as integer.
        
        Args:
            key: Environment variable key
            default: Default value if key is not found or not an integer
            
        Returns:
            Environment variable value as integer
        """
        value = self.get(key)
        if value is None:
            return default
        
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Environment variable {key} is not an integer: {value}")
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """
        Get environment variable as float.
        
        Args:
            key: Environment variable key
            default: Default value if key is not found or not a float
            
        Returns:
            Environment variable value as float
        """
        value = self.get(key)
        if value is None:
            return default
        
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Environment variable {key} is not a float: {value}")
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """
        Get environment variable as boolean.
        
        Args:
            key: Environment variable key
            default: Default value if key is not found
            
        Returns:
            Environment variable value as boolean
        """
        value = self.get(key)
        if value is None:
            return default
        
        return value.lower() in ('true', '1', 't', 'yes', 'y')
    
    def get_list(self, key: str, default: Optional[List[str]] = None, 
                 delimiter: str = ',') -> List[str]:
        """
        Get environment variable as list of strings.
        
        Args:
            key: Environment variable key
            default: Default value if key is not found
            delimiter: Delimiter for splitting the string
            
        Returns:
            Environment variable value as list
        """
        value = self.get(key)
        if value is None:
            return default if default is not None else []
        
        return [item.strip() for item in value.split(delimiter)]
    
    def set(self, key: str, value: str, update_env_file: bool = True) -> None:
        """
        Set environment variable.
        
        Args:
            key: Environment variable key
            value: Environment variable value
            update_env_file: Whether to update the .env file
        """
        os.environ[key] = value
        
        if update_env_file and self.env_file:
            self._update_env_file(key, value)
    
    def _update_env_file(self, key: str, value: str) -> None:
        """
        Update .env file with new value.
        
        Args:
            key: Environment variable key
            value: Environment variable value
        """
        if not self.env_file:
            logger.warning("Cannot update .env file: no file path specified")
            return
        
        # Create .env file if it doesn't exist
        if not self.env_file.exists():
            self.env_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.env_file, 'w'):
                pass
        
        # Read current content
        current_content = {}
        try:
            with open(self.env_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if '=' in line:
                        k, v = line.split('=', 1)
                        current_content[k.strip()] = v.strip()
        except Exception as e:
            logger.error(f"Failed to read .env file: {e}")
            return
        
        # Update value
        current_content[key] = value
        
        # Write back to file
        try:
            with open(self.env_file, 'w') as f:
                for k, v in current_content.items():
                    f.write(f"{k}={v}\n")
            logger.debug(f"Updated {key} in {self.env_file}")
        except Exception as e:
            logger.error(f"Failed to update .env file: {e}")
    
    def get_all(self, prefix: Optional[str] = None) -> Dict[str, str]:
        """
        Get all environment variables, optionally filtered by prefix.
        
        Args:
            prefix: Optional prefix to filter environment variables
            
        Returns:
            Dictionary of environment variables
        """
        if not self.loaded:
            self.load_env()
        
        if prefix:
            return {
                key: value for key, value in os.environ.items()
                if key.startswith(prefix)
            }
        else:
            return dict(os.environ)
    
    def resolve_placeholders(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve environment variable placeholders in configuration.
        
        This function recursively walks through a configuration dictionary and
        replaces placeholders like ${ENV_VAR} with their actual values.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with resolved placeholders
        """
        if not self.loaded:
            self.load_env()
        
        return self._resolve_placeholders_recursive(config)
    
    def _resolve_placeholders_recursive(self, value: Any) -> Any:
        """
        Recursively resolve placeholders in configuration values.
        
        Args:
            value: Configuration value to process
            
        Returns:
            Processed value with resolved placeholders
        """
        if isinstance(value, str):
            return self._resolve_placeholders_in_string(value)
        elif isinstance(value, dict):
            return {k: self._resolve_placeholders_recursive(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_placeholders_recursive(item) for item in value]
        else:
            return value
    
    def _resolve_placeholders_in_string(self, value: str) -> str:
        """
        Resolve placeholders in a string.
        
        Args:
            value: String value to process
            
        Returns:
            String with resolved placeholders
        """
        # Find all ${ENV_VAR} patterns
        pattern = r'\${([A-Za-z0-9_]+)}'
        matches = re.findall(pattern, value)
        
        result = value
        for env_var in matches:
            env_value = self.get(env_var)
            if env_value is not None:
                result = result.replace(f"${{{env_var}}}", env_value)
            else:
                logger.warning(f"Environment variable {env_var} not found for placeholder in: {value}")
        
        return result
    
    def validate_required(self, required_vars: List[str]) -> List[str]:
        """
        Validate that required environment variables are set.
        
        Args:
            required_vars: List of required environment variable keys
            
        Returns:
            List of missing environment variables
        """
        if not self.loaded:
            self.load_env()
        
        missing = []
        for var in required_vars:
            if self.get(var) is None:
                missing.append(var)
                logger.error(f"Required environment variable {var} is not set")
        
        return missing


# Create a singleton instance for easy access
env_manager = EnvManager()


def get_env_manager() -> EnvManager:
    """
    Get the singleton environment manager instance.
    
    Returns:
        Environment manager instance
    """
    return env_manager