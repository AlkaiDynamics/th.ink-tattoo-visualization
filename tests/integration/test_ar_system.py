import pytest
import numpy as np
from src.ar.ar_visualizer import ARVisualizer
from src.ar.ar_config import ARConfig

@pytest.mark.integration
class TestARSystemIntegration:
    @pytest.fixture(scope="class")
    def ar_system(self):
        config = ARConfig()
        visualizer = ARVisualizer(config)
        visualizer.initialize()
        yield visualizer
        visualizer.cleanup()

    @pytest.mark.asyncio
    async def test_full_tattoo_generation_flow(self, ar_system):
        # Setup test user
        test_token = "test_token_123"
        ar_system.set_user(test_token)

        # Generate and apply tattoo
        success = await ar_system.generate_and_set_tattoo(
            "simple flower tattoo",
            style="minimalist"
        )
        assert success

        # Process frame with generated tattoo
        test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        processed_frame, metrics = ar_system.process_frame(test_frame)
        
        assert processed_frame is not None
        assert metrics.get('fps') is not None
        assert 'error' not in metrics

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, ar_system):
        test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Process multiple frames to gather performance data
        for _ in range(10):
            _, metrics = ar_system.process_frame(test_frame)
            assert metrics.get('battery_level') is not None
            assert metrics.get('temperature') is not None

        # Verify performance logs
        assert len(ar_system.performance_monitor.get_metrics_history()) > 0

    @pytest.mark.asyncio
    async def test_subscription_integration(self, ar_system):
        test_token = "premium_test_token"
        ar_system.set_user(test_token)

        # Test subscription limits
        for _ in range(5):
            success = await ar_system.generate_and_set_tattoo(
                "test tattoo",
                style="traditional"
            )
            assert success

        # Verify usage tracking
        usage = ar_system.subscription_manager.get_usage(ar_system.user_id)
        assert usage.get('daily_generations', 0) == 5