import pytest
from src.config.validator import ConfigValidator, ValidationLevel
import os
from typing import Dict, List

class TestReleaseValidation:
    @pytest.fixture
    def validator(self):
        return ConfigValidator()
        
    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        test_vars = {
            'THINK_ENV': 'production',
            'THINK_SECRET_KEY': 'test-secret-key-12345678901234567890',
            'THINK_API_URL': 'https://api.think-ar.dev',
            'THINK_DB_CONNECTION': 'sqlite:///c:/devdrive/thInk/data/prod.db',
            'THINK_MODEL_PATH': 'c:/devdrive/thInk/models',
            'THINK_LOG_LEVEL': 'INFO',
            'THINK_GPU_ENABLED': 'true',
            'THINK_MAX_WORKERS': '4'
        }
        for key, value in test_vars.items():
            monkeypatch.setenv(key, value)
        return test_vars

    def test_production_config_validation(self, validator, mock_env_vars):
        issues = validator.validate_all()
        assert not any(issue['level'] == ValidationLevel.ERROR.value for issue in issues)

    def test_missing_required_vars(self, validator, monkeypatch):
        monkeypatch.delenv('THINK_SECRET_KEY', raising=False)
        issues = validator.validate_all()
        assert any(
            issue['variable'] == 'THINK_SECRET_KEY' and 
            issue['level'] == ValidationLevel.ERROR.value 
            for issue in issues
        )

    def test_invalid_url_format(self, validator, mock_env_vars, monkeypatch):
        monkeypatch.setenv('THINK_API_URL', 'invalid-url')
        issues = validator.validate_all()
        assert any(
            issue['variable'] == 'THINK_API_URL' and 
            issue['level'] == ValidationLevel.ERROR.value 
            for issue in issues
        )

    def test_path_existence(self, validator, mock_env_vars, monkeypatch):
        monkeypatch.setenv('THINK_MODEL_PATH', 'c:/nonexistent/path')
        issues = validator.validate_all()
        assert any(
            issue['variable'] == 'THINK_MODEL_PATH' and 
            issue['level'] == ValidationLevel.ERROR.value 
            for issue in issues
        )

    @pytest.mark.parametrize("worker_count,should_warn", [
        ('0', True),
        ('4', False),
        ('17', True),
        ('abc', True)
    ])
    def test_worker_count_validation(self, validator, mock_env_vars, monkeypatch, worker_count, should_warn):
        monkeypatch.setenv('THINK_MAX_WORKERS', worker_count)
        issues = validator.validate_all()
        has_warning = any(
            issue['variable'] == 'THINK_MAX_WORKERS' and 
            issue['level'] == ValidationLevel.WARNING.value 
            for issue in issues
        )
        assert has_warning == should_warn