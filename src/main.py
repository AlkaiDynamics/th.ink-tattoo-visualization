"""
Main application entry point for the Th.ink AR Tattoo Visualizer.

This module initializes and launches the AR application, sets up components,
and provides the main interaction loop for the user interface.
"""

import asyncio
import logging
import sys
import os
import signal
import time
from pathlib import Path
from datetime import datetime
import traceback
import argparse
from typing import Dict, List, Optional, Any, Tuple

# Import core modules
from src.ar.ar_visualizer import get_ar_visualizer
from src.config.model_config import get_config, ModelConfig
from src.utils.performance_monitor import get_performance_monitor
from src.logging.log_manager import get_logger

# Import necessary services
from src.session.session_manager import get_session_manager
from src.privacy.privacy_manager import get_privacy_manager
from src.subscription.subscription_manager import get_subscription_manager
from src.auth.auth_manager import get_auth_manager

# Import error handling
from src.errors.error_handler import handle_errors, ARError, ErrorLevel

# For metrics
from prometheus_client import start_http_server, Summary, Counter, Gauge

# Configure logging
logger = get_logger()


class ThinkApplication:
    """
    Main application class for the Th.ink AR Tattoo Visualizer.
    
    This class manages the lifecycle of the application, initializes
    components, and coordinates interaction between different modules.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the application.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        if config_path:
            # Would load from specified path in a real implementation
            self.config = get_config()
        else:
            self.config = get_config()
        
        # Initialize components
        self.ar_visualizer = get_ar_visualizer()
        self.performance_monitor = get_performance_monitor()
        self.session_manager = get_session_manager()
        self.privacy_manager = get_privacy_manager()
        self.subscription_manager = get_subscription_manager()
        self.auth_manager = get_auth_manager()
        
        # Application state
        self.running = False
        self.startup_time = datetime.now()
        
        # Metrics
        self.metrics = {
            "ar_frames_processed": Counter("ar_frames_processed_total", "Total AR frames processed"),
            "ar_fps": Gauge("ar_fps", "Current AR frame rate"),
            "memory_usage": Gauge("memory_usage_bytes", "Memory usage in bytes"),
            "active_sessions": Gauge("active_sessions", "Number of active user sessions"),
            "designs_generated": Counter("designs_generated_total", "Total designs generated"),
            "api_requests": Counter("api_requests_total", "Total API requests", ["endpoint", "method"]),
            "processing_time": Summary("processing_time_seconds", "Time spent processing frames")
        }
        
        logger.info("Th.ink application initialized")
    
    @handle_errors()
    async def initialize(self) -> bool:
        """
        Initialize all application components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing Th.ink application")
            
            # Start metrics server
            if self.config.monitoring.prometheus.get("enabled", False):
                port = self.config.monitoring.prometheus.get("port", 9090)
                logger.info(f"Starting metrics server on port {port}")
                start_http_server(port)
            
            # Initialize AR system
            ar_initialized = await self.ar_visualizer.initialize()
            if not ar_initialized:
                logger.error("Failed to initialize AR system")
                return False
            
            # Create required directories
            self._create_directories()
            
            # Set signal handlers
            self._setup_signal_handlers()
            
            # Set running state
            self.running = True
            
            logger.info("Th.ink application initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}", exc_info=True)
            return False
    
    def _create_directories(self) -> None:
        """Create required directories for the application."""
        directories = [
            "data",
            "data/users",
            "data/cache",
            "logs",
            "temp",
            "screenshots"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_signal_handlers(self) -> None:
        """Set up handlers for system signals."""
        # Set up graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame) -> None:
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {sig}, shutting down")
        self.running = False
    
    @handle_errors()
    async def run(self) -> None:
        """
        Run the main application loop.
        
        This is the main entry point for the application execution.
        """
        try:
            # Initialize components
            initialized = await self.initialize()
            if not initialized:
                logger.error("Failed to initialize application, exiting")
                return
            
            logger.info("Starting main application loop")
            
            # Main application loop
            while self.running:
                # This would be the main application loop in a real implementation
                # In a GUI application, this would typically be managed by the UI framework
                
                # Process any background tasks
                await self._process_background_tasks()
                
                # Update metrics
                self._update_metrics()
                
                # Sleep to prevent busy waiting
                await asyncio.sleep(1)
            
            logger.info("Main application loop ended")
            
        except Exception as e:
            logger.error(f"Error in main application loop: {str(e)}", exc_info=True)
        finally:
            # Clean up resources
            await self.cleanup()
    
    async def _process_background_tasks(self) -> None:
        """Process any background tasks for the application."""
        # Cleanup expired sessions
        self.session_manager.cleanup_expired_sessions()
        
        # Run garbage collection if needed
        # This would be implemented in a real application
    
    def _update_metrics(self) -> None:
        """Update application metrics."""
        # Get AR state
        ar_state = self.ar_visualizer.get_state()
        
        # Update metrics
        self.metrics["ar_fps"].set(ar_state.get("fps", 0))
        self.metrics["ar_frames_processed"].inc(1)
        
        # Update memory usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            self.metrics["memory_usage"].set(memory_info.rss)
        except ImportError:
            # psutil not available
            pass
        
        # Update session count
        active_sessions = len(self.session_manager.sessions)
        self.metrics["active_sessions"].set(active_sessions)
    
    @handle_errors()
    async def cleanup(self) -> None:
        """Clean up resources before application exit."""
        logger.info("Cleaning up application resources")
        
        # Clean up AR visualizer
        await self.ar_visualizer.cleanup()
        
        # Clean up other resources
        # This would be implemented in a real application
        
        logger.info("Application cleanup complete")
    
    @handle_errors()
    async def process_command(self, command: str, args: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a command from the user interface.
        
        Args:
            command: Command to process
            args: Command arguments
            
        Returns:
            Command result
        """
        args = args or {}
        
        try:
            # Process different commands
            if command == "get_state":
                return await self.ar_visualizer.get_state()
            
            elif command == "set_overlay":
                image_path = args.get("image_path")
                target_part = args.get("target_part")
                
                if not image_path:
                    raise ARError("Image path required", ErrorLevel.ERROR)
                
                success = await self.ar_visualizer.set_tattoo_overlay(image_path, target_part=target_part)
                return {"success": success}
            
            elif command == "generate_tattoo":
                prompt = args.get("prompt")
                style = args.get("style", "traditional")
                target_part = args.get("target_part")
                
                if not prompt:
                    raise ARError("Prompt required", ErrorLevel.ERROR)
                
                success, design_id = await self.ar_visualizer.generate_and_set_tattoo(
                    prompt=prompt,
                    style=style,
                    target_part=target_part
                )
                
                return {"success": success, "design_id": design_id}
            
            elif command == "adjust_overlay":
                scale = args.get("scale")
                rotation = args.get("rotation")
                position = args.get("position")
                
                success = await self.ar_visualizer.adjust_overlay(scale, rotation, position)
                return {"success": success}
            
            elif command == "clear_overlay":
                success = await self.ar_visualizer.clear_overlay()
                return {"success": success}
            
            elif command == "start_tracking":
                success = await self.ar_visualizer.start_tracking()
                return {"success": success}
            
            elif command == "stop_tracking":
                success = await self.ar_visualizer.stop_tracking()
                return {"success": success}
            
            elif command == "toggle_camera":
                success = await self.ar_visualizer.toggle_camera()
                return {"success": success}
            
            elif command == "set_preview_mode":
                enabled = args.get("enabled", True)
                success = await self.ar_visualizer.set_preview_mode(enabled)
                return {"success": success}
            
            elif command == "save_screenshot":
                filename = args.get("filename")
                path = await self.ar_visualizer.save_screenshot(filename)
                return {"success": path is not None, "path": path}
            
            elif command == "save_design":
                name = args.get("name")
                description = args.get("description")
                
                if not name:
                    raise ARError("Design name required", ErrorLevel.ERROR)
                
                design_id = await self.ar_visualizer.save_design(name, description)
                return {"success": design_id is not None, "design_id": design_id}
            
            elif command == "load_design":
                design_id = args.get("design_id")
                
                if not design_id:
                    raise ARError("Design ID required", ErrorLevel.ERROR)
                
                success = await self.ar_visualizer.load_design(design_id)
                return {"success": success}
            
            elif command == "authenticate":
                user_id = args.get("user_id")
                token = args.get("token")
                
                if not user_id or not token:
                    raise ARError("User ID and token required", ErrorLevel.ERROR)
                
                success = await self.ar_visualizer.set_user(user_id, token)
                return {"success": success}
            
            elif command == "generate_nerf_avatar":
                success = await self.ar_visualizer.generate_nerf_avatar()
                return {"success": success}
            
            else:
                logger.warning(f"Unknown command: {command}")
                return {"success": False, "error": f"Unknown command: {command}"}
            
        except ARError as e:
            logger.error(f"Command error: {str(e)}")
            return {"success": False, "error": str(e), "code": e.code}
        
        except Exception as e:
            logger.error(f"Command error: {str(e)}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Th.ink AR Tattoo Visualizer")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-camera", action="store_true", help="Disable camera usage")
    parser.add_argument("--metrics-port", type=int, default=9090, help="Port for metrics server")
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/think.log")
        ]
    )
    
    # Create application instance
    app = ThinkApplication(config_path=args.config)
    
    # Set debug mode if specified
    if args.debug:
        app.config.debug = True
    
    # Set camera usage if specified
    if args.no_camera:
        # This would disable camera usage in a real implementation
        pass
    
    # Set metrics port if specified
    if args.metrics_port != 9090:
        if hasattr(app.config, "monitoring") and hasattr(app.config.monitoring, "prometheus"):
            app.config.monitoring.prometheus["port"] = args.metrics_port
    
    # Run the application
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application crashed: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    # Set current directory to the script directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Run the application
    sys.exit(main())