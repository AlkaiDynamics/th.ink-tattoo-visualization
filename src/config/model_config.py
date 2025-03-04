"""
Configuration management for the Th.ink AR application.

This module provides the core configuration classes used across the application,
with support for different environments, configuration sources, and validation.
"""

import os
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set, Tuple, cast
from dataclasses import dataclass, field, asdict
import time

from .env_manager import get_env_manager
from .validator import get_config_validator

# Configure logger
logger = logging.getLogger("think.config")


class Environment(Enum):
    """Application environment types."""
    
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"
    
    @classmethod
    def from_string(cls, value: str) -> 'Environment':
        """
        Convert string to Environment enum.
        
        Args:
            value: String value to convert
            
        Returns:
            Environment enum value
            
        Raises:
            ValueError: If value is not a valid environment
        """
        try:
            return cls(value.lower())
        except ValueError:
            valid_values = [e.value for e in cls]
            raise ValueError(f"Invalid environment: {value}. Must be one of: {valid_values}")


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    
    url: str = "sqlite:///./think.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    connect_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityConfig:
    """Security and authentication configuration settings."""
    
    secret_key: str = "default_secret_key_please_change_in_production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    password_min_length: int = 8
    bcrypt_rounds: int = 12
    # JWT token settings
    token_url: str = "/auth/token"
    # CORS settings
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    allow_methods: List[str] = field(default_factory=lambda: ["*"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class AIConfig:
    """AI model configuration settings."""
    
    model_path: str = "./models"
    device: str = "cuda"  # or "cpu"
    batch_size: int = 1
    min_confidence: float = 0.7
    allow_gpu: bool = True
    quantization: bool = True
    mobile_optimized: bool = True
    max_memory_mb: int = 2048
    # Model file names
    skin_detection_model: str = "skin_detection.pth"
    tattoo_generator_model: str = "tattoo_generator.pth"
    motion_tracking_model: str = "motion_tracking.pth"


@dataclass
class ARConfig:
    """Augmented reality configuration settings."""
    
    camera_resolution: Tuple[int, int] = (1920, 1080)
    min_fps: int = 45
    max_fps: int = 60
    shadow_quality: str = "high"
    tracking_precision: float = 0.95
    skin_detection_threshold: float = 0.85
    tattoo_overlay_opacity: float = 0.9
    surface_mapping_quality: str = "high"
    depth_sensing_enabled: bool = True
    # NeRF Metahuman Avatar settings
    nerf_resolution: int = 512
    nerf_samples_per_ray: int = 64
    nerf_chunk_size: int = 32768


@dataclass
class PrivacyConfig:
    """Data privacy and retention configuration settings."""
    
    data_retention_days: int = 90
    anonymization_enabled: bool = True
    allowed_data_uses: List[str] = field(default_factory=lambda: ["service_improvement", "model_training", "analytics"])
    opt_out_features: List[str] = field(default_factory=lambda: ["model_training", "analytics", "marketing"])


@dataclass
class SubscriptionConfig:
    """Subscription tier configuration settings."""
    
    tiers: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
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
    })


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    
    level: str = "INFO"
    file_path: str = "./logs"
    rotation: str = "1 day"
    retention: str = "30 days"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    json_format: bool = True


@dataclass
class ApiConfig:
    """API configuration settings."""
    
    url: str = "http://localhost:8000"
    prefix: str = "/api/v1"
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"])
    allowed_hosts: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])


@dataclass
class ServicesConfig:
    """External services configuration settings."""
    
    # Redis
    redis: Dict[str, Any] = field(default_factory=lambda: {
        "host": "localhost",
        "port": 6379,
        "password": None,
        "db": 0
    })
    
    # AWS SageMaker
    sagemaker: Dict[str, Any] = field(default_factory=lambda: {
        "region": "us-west-2",
        "model_name": "think-ai-model",
        "endpoint_name": "think-ai-endpoint"
    })
    
    # Stripe
    stripe: Dict[str, Any] = field(default_factory=lambda: {
        "public_key": "",
        "webhook_secret": ""
    })


@dataclass
class PerformanceConfig:
    """Performance optimization configuration settings."""
    
    caching: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "ttl_seconds": 3600,
        "max_size_mb": 500
    })
    
    throttling: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "rate_limit_per_minute": 60
    })


@dataclass
class MonitoringConfig:
    """Application monitoring configuration settings."""
    
    prometheus: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "port": 9090
    })
    
    health_check: Dict[str, Any] = field(default_factory=lambda: {
        "interval_seconds": 30,
        "timeout_seconds": 5
    })


@dataclass
class ModelConfig:
    """
    Main configuration class for the Th.ink AR application.
    
    This class combines all configuration settings and provides methods
    for loading, validating, and accessing configuration values.
    """
    
    env: Environment = Environment.DEVELOPMENT
    debug: bool = True
    api: ApiConfig = field(default_factory=ApiConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    ar: ARConfig = field(default_factory=ARConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    services: ServicesConfig = field(default_factory=ServicesConfig)
    subscription: SubscriptionConfig = field(default_factory=SubscriptionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    def __post_init__(self):
        """Initialize configuration with values from environment or config files."""
        self._env_manager = get_env_manager()
        self._validator = get_config_validator()
        self._load_time = int(time.time())
        
        # Load configuration from environment or config files
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from environment and config files."""
        # First, try to determine environment
        env_value = self._env_manager.get("THINK_ENV", "development")
        try:
            self.env = Environment.from_string(env_value)
        except ValueError as e:
            logger.warning(str(e))
            # Default to development if invalid
            self.env = Environment.DEVELOPMENT
        
        # Load configuration from JSON file for environment
        config_file = Path(f"config/{self.env.value}.json")
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config_data = json.load(f)
                self._update_from_dict(config_data)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load configuration from {config_file}: {e}")
        else:
            logger.warning(f"Configuration file {config_file} not found, using defaults")
        
        # Override with environment variables
        self._load_from_env()
        
        # Validate configuration
        self._validate_config()
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Process all environment variables with THINK_ prefix
        env_vars = self._env_manager.get_all(prefix="THINK_")
        for key, value in env_vars.items():
            self._set_from_env(key, value)
    
    def _set_from_env(self, key: str, value: str) -> None:
        """
        Set configuration value from environment variable.
        
        Args:
            key: Environment variable key
            value: Environment variable value
        """
        # Skip if empty
        if not value:
            return
        
        # Remove prefix
        if key.startswith("THINK_"):
            key = key[6:]
        
        # Convert to nested dict path (e.g., DATABASE_URL -> database.url)
        parts = key.lower().split("_")
        
        if len(parts) < 2:
            # Skip short keys
            return
        
        # First part is the section
        section = parts[0]
        
        if section == "env" and len(parts) == 1:
            # Special case for environment
            try:
                self.env = Environment.from_string(value)
            except ValueError as e:
                logger.warning(str(e))
            return
        
        # Try to find the corresponding dataclass field
        if hasattr(self, section):
            field_value = getattr(self, section)
            
            # For simple sections with direct values
            if len(parts) == 2 and not isinstance(field_value, (dict, list, tuple)):
                setattr(self, section, self._convert_value(value, type(field_value)))
                return
            
            # For dataclass sections
            if len(parts) >= 2 and isinstance(field_value, object) and hasattr(field_value, "__dataclass_fields__"):
                # Join remaining parts as attribute name
                attr_name = "_".join(parts[1:])
                
                # Check if attribute exists in dataclass
                if hasattr(field_value, attr_name):
                    try:
                        # Get current value to determine type
                        current_value = getattr(field_value, attr_name)
                        # Convert value to appropriate type
                        converted_value = self._convert_value(value, type(current_value))
                        # Set attribute
                        setattr(field_value, attr_name, converted_value)
                        logger.debug(f"Set {section}.{attr_name} = {converted_value}")
                    except Exception as e:
                        logger.warning(f"Failed to set {section}.{attr_name} from environment: {e}")
    
    def _convert_value(self, value: str, target_type: type) -> Any:
        """
        Convert string value to target type.
        
        Args:
            value: String value to convert
            target_type: Target type
            
        Returns:
            Converted value
        """
        if target_type == str:
            return value
        elif target_type == bool:
            return value.lower() in ("true", "1", "t", "yes", "y")
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == list:
            return [item.strip() for item in value.split(",")]
        elif target_type == tuple:
            items = [item.strip() for item in value.split(",")]
            return tuple(items)
        elif target_type == dict:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON value: {value}")
                return {}
        else:
            # Try to use the type's constructor
            return target_type(value)
    
    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.
        
        Args:
            config_data: Configuration dictionary
        """
        # Process top-level keys
        for key, value in config_data.items():
            if key == "env":
                try:
                    self.env = Environment.from_string(value)
                except ValueError as e:
                    logger.warning(str(e))
                continue
            
            if hasattr(self, key):
                current_value = getattr(self, key)
                
                # For simple values
                if not isinstance(current_value, (dict, list, tuple)) and not hasattr(current_value, "__dataclass_fields__"):
                    setattr(self, key, value)
                    continue
                
                # For dataclass fields
                if hasattr(current_value, "__dataclass_fields__"):
                    if isinstance(value, dict):
                        self._update_dataclass(current_value, value)
                    continue
                
                # For dict fields
                if isinstance(current_value, dict) and isinstance(value, dict):
                    current_value.update(value)
                    continue
                
                # For list fields
                if isinstance(current_value, list) and isinstance(value, list):
                    setattr(self, key, value)
                    continue
    
    def _update_dataclass(self, obj: Any, data: Dict[str, Any]) -> None:
        """
        Update dataclass instance from dictionary.
        
        Args:
            obj: Dataclass instance
            data: Data dictionary
        """
        for key, value in data.items():
            if hasattr(obj, key):
                current_value = getattr(obj, key)
                
                # For nested dataclasses
                if hasattr(current_value, "__dataclass_fields__") and isinstance(value, dict):
                    self._update_dataclass(current_value, value)
                else:
                    setattr(obj, key, value)
    
    def _validate_config(self) -> None:
        """Validate configuration settings."""
        # Convert to dict for validation
        config_dict = self.to_dict()
        
        # Flatten dict for validation
        flat_config = self._flatten_dict(config_dict)
        
        # Validate
        issues = []
        for key, value in flat_config.items():
            env_key = f"THINK_{key.upper()}"
            if env_key in self._validator.validation_rules:
                key_issues = self._validator.validate(env_key, value)
                issues.extend(key_issues)
        
        # Log validation issues
        for issue in issues:
            level = issue.get("level", "warning")
            if level.lower() == "error":
                logger.error(issue.get("message"))
            elif level.lower() == "warning":
                logger.warning(issue.get("message"))
            else:
                logger.info(issue.get("message"))
    
    def _flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """
        Flatten nested dictionary with underscore-separated keys.
        
        Args:
            d: Dictionary to flatten
            prefix: Key prefix
            
        Returns:
            Flattened dictionary
        """
        result = {}
        
        for key, value in d.items():
            new_key = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively flatten nested dicts
                nested = self._flatten_dict(value, new_key)
                result.update(nested)
            else:
                result[new_key] = value
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        result = {
            "env": self.env.value,
            "debug": self.debug
        }
        
        # Convert dataclass fields to dictionaries
        for field_name in self.__dataclass_fields__:  # type: ignore
            if field_name in ("env", "debug"):
                continue
            
            value = getattr(self, field_name)
            if hasattr(value, "__dataclass_fields__"):
                result[field_name] = asdict(value)
            else:
                result[field_name] = value
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert configuration to JSON string.
        
        Args:
            indent: JSON indentation
            
        Returns:
            Configuration as JSON string
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    def reload(self) -> None:
        """Reload configuration from sources."""
        self._load_config()
        logger.info("Configuration reloaded")
    
    @property
    def is_production(self) -> bool:
        """Check if environment is production."""
        return self.env == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if environment is development."""
        return self.env == Environment.DEVELOPMENT
    
    @property
    def is_test(self) -> bool:
        """Check if environment is test."""
        return self.env == Environment.TEST


# Create a singleton instance
_config_instance = None


def get_config() -> ModelConfig:
    """
    Get the singleton configuration instance.
    
    Returns:
        Configuration instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ModelConfig()
    return _config_instance