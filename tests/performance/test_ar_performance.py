import pytest
import time
import numpy as np
from src.ar.ar_visualizer import ARVisualizer
from src.ar.ar_config import ARConfig

@pytest.mark.performance
class TestARPerformance:
    @pytest.fixture(scope="class")
    def ar_system(self, test_config):
        visualizer = ARVisualizer(test_config)
        visualizer.initialize()
        yield visualizer
        visualizer.cleanup()

    def test_frame_processing_speed(self, ar_system, test_frame):
        processing_times = []
        
        # Warm-up phase
        for _ in range(5):
            ar_system.process_frame(test_frame)
            
        # Measurement phase
        for _ in range(100):
            start_time = time.perf_counter()
            ar_system.process_frame(test_frame)
            processing_times.append(time.perf_counter() - start_time)
            
        avg_time = np.mean(processing_times)
        fps = 1.0 / avg_time
        
        assert fps >= ar_system.config.MIN_FPS
        assert np.std(processing_times) < 0.016  # Max 16ms variance

    @pytest.mark.asyncio
    async def test_tattoo_generation_performance(self, ar_system):
        generation_times = []
        
        for _ in range(5):
            start_time = time.perf_counter()
            await ar_system.generate_and_set_tattoo("test tattoo", "minimalist")
            generation_times.append(time.perf_counter() - start_time)
            
        avg_generation_time = np.mean(generation_times)
        assert avg_generation_time < 2.0  # Max 2 seconds for generation

    def test_memory_usage(self, ar_system, mock_camera_feed):
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple frames
        for frame in mock_camera_feed():
            ar_system.process_frame(frame)
            
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 100  # Max 100MB increase