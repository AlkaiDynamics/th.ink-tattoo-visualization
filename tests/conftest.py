import pytest
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables and configurations"""
    os.environ['THINK_ENV'] = 'test'
    os.environ['THINK_TEST_MODE'] = 'true'
    yield
    os.environ.pop('THINK_ENV', None)
    os.environ.pop('THINK_TEST_MODE', None)

import pytest
import numpy as np
from src.ar.ar_config import ARConfig

@pytest.fixture(scope="session")
def test_frame():
    """Provide a standard test frame for AR processing"""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)

@pytest.fixture(scope="session")
def test_config():
    """Provide a test configuration"""
    config = ARConfig()
    config.MIN_FPS = 30  # Lower for testing
    config.MAX_FPS = 45
    config.SKIN_DETECTION_THRESHOLD = 0.7  # More permissive for testing
    return config

@pytest.fixture(scope="session")
def mock_camera_feed():
    """Provide a sequence of test frames"""
    def frame_generator():
        for _ in range(10):
            yield np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    return frame_generator

@pytest.fixture(scope="session")
def performance_config():
    """Configuration for performance testing"""
    config = ARConfig()
    config.PERFORMANCE_MONITORING = True
    config.PROFILING_ENABLED = True
    config.MIN_FPS = 30
    config.MAX_FPS = 60
    config.MEMORY_LIMIT_MB = 512
    return config

@pytest.fixture(scope="session")
def benchmark_frames():
    """Generate benchmark frames of different sizes"""
    resolutions = [(720, 1280), (1080, 1920), (1440, 2560)]
    return {
        f"{height}p": np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        for height, width in resolutions
    }
@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available for testing"""
    return torch.cuda.is_available()

@pytest.fixture(scope="session")
def gpu_info(gpu_available):
    """Provide GPU information for tests"""
    if not gpu_available:
        pytest.skip("GPU not available")
    return {
        'device_name': torch.cuda.get_device_name(0),
        'compute_capability': torch.cuda.get_device_capability(0),
        'memory_total': torch.cuda.get_device_properties(0).total_memory
    }