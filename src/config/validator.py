"""
Configuration validation system for the Th.ink AR application.

This module provides utilities for validating configuration values,
ensuring they meet the required format, type, and constraints for
the application to function correctly.
"""

import os
import re
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Set, Callable
from pathlib import Path
import ipaddress

from .env_manager import get_env_manager

# Configure logger
logger = logging.getLogger("think.config")


class ValidationLevel(Enum):
    """Validation issue severity levels."""
    
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationRule:
    """Rule for validating a configuration value."""
    
    def __init__(
        self,
        required: bool = False,
        default: Any = None,
        level: ValidationLevel = ValidationLevel.ERROR,
        values: Optional[List[Any]] = None,
        pattern: Optional[str] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        exists: bool = False,
        custom_validator: Optional[Callable[[Any], bool]] = None,
        description: Optional[str] = None,
        sensitive: bool = False,
        type_: Optional[type] = None
    ):
        """
        Initialize validation rule.
        
        Args:
            required: Whether the value is required
            default: Default value if not provided
            level: Severity level of validation failure
            values: List of allowed values
            pattern: Regex pattern the value must match
            min_length: Minimum length for string values
            max_length: Maximum length for string values
            min_value: Minimum value for numeric values
            max_value: Maximum value for numeric values
            exists: Whether a file or directory path must exist
            custom_validator: Custom validation function
            description: Description of the configuration value
            sensitive: Whether the value contains sensitive data
            type_: Expected type of the value
        """
        self.required = required
        self.default = default
        self.level = level
        self.values = values
        self.pattern = pattern
        self.min_length = min_length
        self.max_length = max_length
        self.min_value = min_value
        self.max_value = max_value
        self.exists = exists
        self.custom_validator = custom_validator
        self.description = description
        self.sensitive = sensitive
        self.type = type_


class ConfigValidator:
    """
    Configuration validator for the Th.ink AR application.
    
    This class provides methods to define validation rules for configuration values
    and validate them against those rules.
    """
    
    def __init__(self):
        """Initialize the config validator with default validation rules."""
        self.validation_rules: Dict[str, ValidationRule] = {}
        self.env_manager = get_env_manager()
        
        # Define default validation rules
        self._define_default_rules()
    
    def _define_default_rules(self) -> None:
        """Define default validation rules for common configuration values."""
        # Environment settings
        self.add_rule(
            'THINK_ENV',
            ValidationRule(
                required=True,
                values=['development', 'staging', 'production', 'test'],
                level=ValidationLevel.ERROR,
                description="Application environment"
            )
        )
        
        # Security settings
        self.add_rule(
            'THINK_SECRET_KEY',
            ValidationRule(
                required=True,
                min_length=32,
                pattern=r'^[A-Za-z0-9_\-]+$',
                level=ValidationLevel.ERROR,
                description="Secret key for cryptographic operations",
                sensitive=True
            )
        )
        
        # API settings
        self.add_rule(
            'THINK_API_URL',
            ValidationRule(
                required=True,
                pattern=r'^https?://[\w\-\.]+(:\d+)?(/[\w\-\.]*)*$',
                level=ValidationLevel.ERROR,
                description="Base URL for the API"
            )
        )
        
        # Database settings
        self.add_rule(
            'THINK_DB_CONNECTION',
            ValidationRule(
                required=True,
                pattern=r'^(sqlite:///.*|postgresql://.*|mysql://.*|oracle://.*|mssql://.*|cockroachdb://.*|redshift://.*|firebird://.*|sybase://.*|db2://.*|informix://.*|sqlserver://.*|access://.*|firebird://.*|teradata://.*|presto://.*|snowflake://.*|bigquery://.*|clickhouse://.*|awsathena://.*|awsglue://.*|databricks://.*|jdbc://.*|odbc://.*|vertica://.*|aster://.*|exasol://.*|impala://.*|monetdb://.*|netezza://.*|singlestore://.*|vectorwise://.*|vertica://.*|interbase://.*|maxdb://.*|sapdb://.*|saphana://.*|spark://.*|sqlanywhere://.*|tsunamidb://.*|sqlserver://.*|db2://.*|oracle://.*|mysql://.*|postgresql://.*|sqlite:///.*|mssql://.*|sqlite:///.*)$',
                level=ValidationLevel.ERROR,
                description="Database connection string",
                sensitive=True
            )
        )
        
        # Model path
        self.add_rule(
            'THINK_MODEL_PATH',
            ValidationRule(
                required=True,
                exists=True,
                level=ValidationLevel.ERROR,
                description="Path to AI model files"
            )
        )
        
        # Logging settings
        self.add_rule(
            'THINK_LOG_LEVEL',
            ValidationRule(
                required=False,
                values=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                default='INFO',
                level=ValidationLevel.WARNING,
                description="Logging level"
            )
        )
        
        # Performance settings
        self.add_rule(
            'THINK_GPU_ENABLED',
            ValidationRule(
                required=False,
                values=['true', 'false', '1', '0', 'yes', 'no'],
                default='false',
                level=ValidationLevel.INFO,
                description="Whether to use GPU acceleration if available"
            )
        )
        
        self.add_rule(
            'THINK_MAX_WORKERS',
            ValidationRule(
                required=False,
                pattern=r'^\d+$',
                min_value=1,
                max_value=32,
                default='4',
                level=ValidationLevel.WARNING,
                description="Maximum number of worker threads/processes"
            )
        )
        
        # External service settings
        self.add_rule(
            'THINK_STRIPE_API_KEY',
            ValidationRule(
                required=False,
                pattern=r'^(pk|sk|rk)_\w+$',
                level=ValidationLevel.ERROR,
                description="Stripe API key",
                sensitive=True
            )
        )
        
        self.add_rule(
            'THINK_STRIPE_WEBHOOK_SECRET',
            ValidationRule(
                required=False,
                pattern=r'^whsec_\w+$',
                level=ValidationLevel.ERROR,
                description="Stripe webhook secret",
                sensitive=True
            )
        )
        
        self.add_rule(
            'THINK_OPENAI_API_KEY',
            ValidationRule(
                required=False,
                pattern=r'^sk-\w+$',
                level=ValidationLevel.ERROR,
                description="OpenAI API key",
                sensitive=True
            )
        )
        
        # Redis settings
        self.add_rule(
            'THINK_REDIS_URL',
            ValidationRule(
                required=False,
                pattern=r'^redis://(?:\w+:\w+@)?[\w\-\.]+:\d+(?:/\d+)?$',
                level=ValidationLevel.WARNING,
                description="Redis connection URL",
                sensitive=True
            )
        )
        
        # AWS settings
        self.add_rule(
            'THINK_AWS_ACCESS_KEY_ID',
            ValidationRule(
                required=False,
                pattern=r'^[A-Z0-9]+$',
                level=ValidationLevel.ERROR,
                description="AWS access key ID",
                sensitive=True
            )
        )
        
        self.add_rule(
            'THINK_AWS_SECRET_ACCESS_KEY',
            ValidationRule(
                required=False,
                pattern=r'^[A-Za-z0-9\+/]+$',
                level=ValidationLevel.ERROR,
                description="AWS secret access key",
                sensitive=True
            )
        )
        
        self.add_rule(
            'THINK_AWS_REGION',
            ValidationRule(
                required=False,
                pattern=r'^[a-z]{2}-[a-z]+-\d+,
                level=ValidationLevel.ERROR,
                description="AWS region"
            )
        )
    
    def add_rule(self, key: str, rule: ValidationRule) -> None:
        """
        Add a validation rule for a configuration value.
        
        Args:
            key: Configuration key
            rule: Validation rule
        """
        self.validation_rules[key] = rule
    
    def validate(self, key: str, value: Any) -> List[Dict[str, Any]]:
        """
        Validate a single configuration value against its rule.
        
        Args:
            key: Configuration key
            value: Configuration value
            
        Returns:
            List of validation issues
        """
        if key not in self.validation_rules:
            logger.warning(f"No validation rule defined for {key}")
            return []
        
        rule = self.validation_rules[key]
        issues = []
        
        # Check if required
        if rule.required and value is None:
            issues.append({
                'variable': key,
                'message': f"Required configuration value {key} is not set",
                'level': rule.level.value
            })
            return issues
        
        # Skip further validation if value is None
        if value is None:
            return issues
        
        # Check type if specified
        if rule.type is not None and not isinstance(value, rule.type):
            issues.append({
                'variable': key,
                'message': f"Invalid type for {key}. Expected {rule.type.__name__}, got {type(value).__name__}",
                'level': rule.level.value
            })
        
        # Check allowed values
        if rule.values is not None and value not in rule.values:
            issues.append({
                'variable': key,
                'message': f"Invalid value for {key}. Must be one of: {rule.values}",
                'level': rule.level.value
            })
        
        # Check pattern
        if rule.pattern is not None and isinstance(value, str) and not re.match(rule.pattern, value):
            issues.append({
                'variable': key,
                'message': f"Invalid format for {key}",
                'level': rule.level.value
            })
        
        # Check length constraints
        if isinstance(value, str):
            if rule.min_length is not None and len(value) < rule.min_length:
                issues.append({
                    'variable': key,
                    'message': f"{key} must be at least {rule.min_length} characters long",
                    'level': rule.level.value
                })
            
            if rule.max_length is not None and len(value) > rule.max_length:
                issues.append({
                    'variable': key,
                    'message': f"{key} must be at most {rule.max_length} characters long",
                    'level': rule.level.value
                })
        
        # Check numeric constraints
        if isinstance(value, (int, float)):
            if rule.min_value is not None and value < rule.min_value:
                issues.append({
                    'variable': key,
                    'message': f"{key} must be at least {rule.min_value}",
                    'level': rule.level.value
                })
            
            if rule.max_value is not None and value > rule.max_value:
                issues.append({
                    'variable': key,
                    'message': f"{key} must be at most {rule.max_value}",
                    'level': rule.level.value
                })
        
        # Check path existence
        if rule.exists and isinstance(value, str):
            path = Path(value)
            if not path.exists():
                issues.append({
                    'variable': key,
                    'message': f"Path {value} for {key} does not exist",
                    'level': rule.level.value
                })
        
        # Run custom validator
        if rule.custom_validator is not None:
            try:
                if not rule.custom_validator(value):
                    issues.append({
                        'variable': key,
                        'message': f"Custom validation failed for {key}",
                        'level': rule.level.value
                    })
            except Exception as e:
                issues.append({
                    'variable': key,
                    'message': f"Custom validation error for {key}: {str(e)}",
                    'level': rule.level.value
                })
        
        return issues
    
    def validate_all(self) -> List[Dict[str, Any]]:
        """
        Validate all configuration values against their rules.
        
        Returns:
            List of validation issues
        """
        issues = []
        
        for key, rule in self.validation_rules.items():
            value = self.env_manager.get(key)
            
            # If value is not set but has a default, use default
            if value is None and rule.default is not None:
                value = rule.default
                
                # Log info about using default
                logger.debug(f"Using default value for {key}: {rule.default if not rule.sensitive else '******'}")
            
            # Validate the value
            key_issues = self.validate(key, value)
            issues.extend(key_issues)
        
        return issues
    
    def validate_config(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of validation issues
        """
        issues = []
        
        for key, value in config.items():
            if key in self.validation_rules:
                key_issues = self.validate(key, value)
                issues.extend(key_issues)
        
        return issues
    
    def apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply default values to a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with defaults applied
        """
        result = config.copy()
        
        for key, rule in self.validation_rules.items():
            if key not in result and rule.default is not None:
                result[key] = rule.default
        
        return result
    
    def get_rule(self, key: str) -> Optional[ValidationRule]:
        """
        Get the validation rule for a configuration key.
        
        Args:
            key: Configuration key
            
        Returns:
            Validation rule or None if not found
        """
        return self.validation_rules.get(key)
    
    def get_required_keys(self) -> List[str]:
        """
        Get list of required configuration keys.
        
        Returns:
            List of required configuration keys
        """
        return [
            key for key, rule in self.validation_rules.items()
            if rule.required
        ]
    
    def describe_rules(self, include_sensitive: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Get a description of all validation rules.
        
        Args:
            include_sensitive: Whether to include sensitive configuration keys
            
        Returns:
            Dictionary of validation rule descriptions
        """
        result = {}
        
        for key, rule in self.validation_rules.items():
            if rule.sensitive and not include_sensitive:
                continue
            
            result[key] = {
                'required': rule.required,
                'default': rule.default if not rule.sensitive else '******',
                'description': rule.description,
                'sensitive': rule.sensitive
            }
            
            if rule.values:
                result[key]['allowed_values'] = rule.values
            
            if rule.pattern:
                result[key]['pattern'] = rule.pattern
            
            if rule.min_length is not None:
                result[key]['min_length'] = rule.min_length
            
            if rule.max_length is not None:
                result[key]['max_length'] = rule.max_length
            
            if rule.min_value is not None:
                result[key]['min_value'] = rule.min_value
            
            if rule.max_value is not None:
                result[key]['max_value'] = rule.max_value
            
            if rule.exists:
                result[key]['path_must_exist'] = True
        
        return result


# Create a singleton instance for easy access
config_validator = ConfigValidator()


def get_config_validator() -> ConfigValidator:
    """
    Get the singleton config validator instance.
    
    Returns:
        Config validator instance
    """
    return config_validator