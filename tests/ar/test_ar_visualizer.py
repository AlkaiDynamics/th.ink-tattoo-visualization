import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.ar.ar_visualizer import ARVisualizer
from src.ar.ar_config import ARConfig

@pytest.fixture
def ar_config():
    return ARConfig()

@pytest.fixture
def ar_visualizer(ar_config):
    return ARVisualizer(ar_config)

class TestARVisualizer:
    def test_initialization(self, ar_visualizer):
        assert ar_visualizer.current_fps == ar_visualizer.config.MAX_FPS
        assert not ar_visualizer.is_tracking
        assert ar_visualizer.current_overlay is None
        
    @patch('cv2.VideoCapture')
    def test_camera_initialization(self, mock_camera, ar_visualizer):
        mock_camera.return_value.isOpened.return_value = True
        assert ar_visualizer.initialize_camera()
        
    @pytest.mark.asyncio
    async def test_generate_tattoo_without_auth(self, ar_visualizer):
        result = await ar_visualizer.generate_and_set_tattoo("test prompt")
        assert not result
        
    def test_performance_settings(self, ar_visualizer):
        ar_visualizer.update_performance_settings(15.0, 45.0)
        assert ar_visualizer.current_fps == ar_visualizer.config.MIN_FPS
        
    def test_process_frame_without_tracking(self, ar_visualizer):
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame, metrics = ar_visualizer.process_frame(test_frame)
        assert frame.shape == test_frame.shape
        assert 'battery_level' in metrics