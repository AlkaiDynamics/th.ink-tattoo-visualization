from typing import Dict, List, Optional
from enum import Enum
import re
import os

class ValidationLevel(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class ConfigValidator:
    def __init__(self):
        self.validation_rules = {
            'THINK_ENV': {
                'required': True,
                'values': ['development', 'staging', 'production'],
                'level': ValidationLevel.ERROR
            },
            'THINK_SECRET_KEY': {
                'required': True,
                'min_length': 32,
                'pattern': r'^[A-Za-z0-9_-]+$',
                'level': ValidationLevel.ERROR
            },
            'THINK_API_URL': {
                'required': True,
                'pattern': r'^https?://[\w\-\.]+(:\d+)?(/[\w\-\.]*)*$',
                'level': ValidationLevel.ERROR
            },
            'THINK_DB_CONNECTION': {
                'required': True,
                'pattern': r'^sqlite:///.*|postgresql://.*$',
                'level': ValidationLevel.ERROR
            },
            'THINK_MODEL_PATH': {
                'required': True,
                'exists': True,
                'level': ValidationLevel.ERROR
            },
            'THINK_LOG_LEVEL': {
                'required': False,
                'values': ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                'default': 'INFO',
                'level': ValidationLevel.WARNING
            },
            'THINK_GPU_ENABLED': {
                'required': False,
                'values': ['true', 'false'],
                'default': 'false',
                'level': ValidationLevel.INFO
            },
            'THINK_MAX_WORKERS': {
                'required': False,
                'pattern': r'^\d+$',
                'min_value': 1,
                'max_value': 16,
                'default': '4',
                'level': ValidationLevel.WARNING
            }
        }
        
    def validate_all(self) -> List[Dict]:
        """Validate all configuration settings"""
        issues = []
        
        for var_name, rules in self.validation_rules.items():
            value = os.getenv(var_name)
            
            if rules.get('required', False) and not value:
                issues.append({
                    'variable': var_name,
                    'message': f"Required environment variable {var_name} is not set",
                    'level': rules['level'].value
                })
                continue
                
            if value:
                issues.extend(self._validate_value(var_name, value, rules))
                
        return issues
    
    def _validate_value(self, var_name: str, value: str, rules: Dict) -> List[Dict]:
        issues = []
        
        if 'values' in rules and value not in rules['values']:
            issues.append({
                'variable': var_name,
                'message': f"Invalid value for {var_name}. Must be one of: {rules['values']}",
                'level': rules['level'].value
            })
            
        if 'pattern' in rules and not re.match(rules['pattern'], value):
            issues.append({
                'variable': var_name,
                'message': f"Invalid format for {var_name}",
                'level': rules['level'].value
            })
            
        if 'min_length' in rules and len(value) < rules['min_length']:
            issues.append({
                'variable': var_name,
                'message': f"{var_name} must be at least {rules['min_length']} characters long",
                'level': rules['level'].value
            })
            
        if 'exists' in rules and not os.path.exists(value):
            issues.append({
                'variable': var_name,
                'message': f"Path {value} for {var_name} does not exist",
                'level': rules['level'].value
            })
            
        return issues