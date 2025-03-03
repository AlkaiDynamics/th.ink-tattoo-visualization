import pytest
import torch
import numpy as np
from src.ar.ar_visualizer import ARVisualizer

@pytest.mark.gpu
class TestGPUAcceleration:
    @pytest.fixture(scope="class")
    def gpu_config(self, performance_config):
        performance_config.GPU_ENABLED = True
        performance_config.CUDA_DEVICE = 0 if torch.cuda.is_available() else None
        return performance_config

    @pytest.fixture(scope="class")
    def gpu_ar_system(self, gpu_config):
        visualizer = ARVisualizer(gpu_config)
        visualizer.initialize()
        yield visualizer
        visualizer.cleanup()

    def test_gpu_availability(self, gpu_ar_system):
        assert torch.cuda.is_available(), "CUDA not available"
        assert gpu_ar_system.config.GPU_ENABLED
        assert gpu_ar_system.config.CUDA_DEVICE is not None

    def test_gpu_frame_processing(self, gpu_ar_system, benchmark_frames):
        frame = benchmark_frames['1080p']
        
        # Process frame with GPU acceleration
        processed_frame, metrics = gpu_ar_system.process_frame(frame)
        
        assert metrics.get('gpu_utilization') is not None
        assert metrics.get('processing_device') == 'gpu'
        assert metrics.get('fps') > gpu_ar_system.config.MIN_FPS

    @pytest.mark.asyncio
    async def test_gpu_tattoo_generation(self, gpu_ar_system):
        start_memory = torch.cuda.memory_allocated()
        
        await gpu_ar_system.generate_and_set_tattoo(
            "test tattoo",
            style="realistic"
        )
        
        peak_memory = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        
        assert peak_memory - start_memory < 1024 * 1024 * 1024  # Max 1GB GPU memory usage