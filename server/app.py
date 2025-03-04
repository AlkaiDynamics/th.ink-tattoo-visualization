"""
Main server application for the Th.ink AR Tattoo Visualizer system.

This module initializes and configures the FastAPI application, sets up
middleware, database connections, and includes all API routers for the
various services of the Th.ink platform.
"""

import os
import logging
import time
from pathlib import Path
from fastapi import FastAPI, Request, Response, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, start_http_server
import threading
import uvicorn
from typing import Dict, List, Optional, Any

# Import database modules
from server.database import models, engine, dependencies, connect_to_db

# Import configuration
from server.config.model_config import get_config

# Import router modules
from server.auth.auth_manager import auth_router
from server.payment.payment_manager import payment_router
from server.marketplace.marketplace_manager import marketplace_router
from server.ai.tattoo_generator import router as ai_router
from server.subscription.subscription_manager import router as subscription_router

# Import error handling
from server.errors.error_handler import handle_errors, ARError, ModelError, ValidationError

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/server.log")
    ]
)
logger = logging.getLogger("think.server")

# Get configuration
config = get_config()

# Create FastAPI application
app = FastAPI(
    title="Th.ink AR Tattoo Visualization System",
    description="Backend services for AI-powered tattoo visualization, user management, and marketplace functionality.",
    version="1.0.0",
    docs_url=None if config.is_production else "/docs",
    redoc_url=None if config.is_production else "/redoc",
    openapi_url=None if config.is_production else "/openapi.json"
)

# Set up middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=config.api.allowed_hosts
)

# Set up metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP Request Latency", ["method", "endpoint"])

@app.middleware("http")
async def metrics_middleware(request: Request, call_next) -> Response:
    """Middleware to collect request metrics."""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record request count and latency
    endpoint = request.url.path
    method = request.method
    status = response.status_code
    
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(time.time() - start_time)
    
    return response

@app.middleware("http")
async def log_requests_middleware(request: Request, call_next) -> Response:
    """Middleware to log all requests."""
    start_time = time.time()
    
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Log request
    logger.info(f"Request {request.method} {request.url.path} from {client_ip}")
    
    # Process request
    response = await call_next(request)
    
    # Log response time
    process_time = time.time() - start_time
    logger.info(f"Response {response.status_code} for {request.method} {request.url.path} - Took {process_time:.4f}s")
    
    return response

@app.middleware("http")
async def add_headers_middleware(request: Request, call_next) -> Response:
    """Middleware to add security headers."""
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Server"] = "Th.ink Server"
    
    return response

# Error handlers
@app.exception_handler(ARError)
async def ar_error_handler(request: Request, exc: ARError):
    """Handle application-specific errors."""
    logger.error(f"AR Error: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": exc.message,
            "code": exc.code,
            "level": exc.level.value,
            "category": exc.category.value
        }
    )

@app.exception_handler(ModelError)
async def model_error_handler(request: Request, exc: ModelError):
    """Handle model-related errors."""
    logger.error(f"Model Error: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": exc.message,
            "code": exc.code,
            "level": exc.level.value,
            "category": exc.category.value
        }
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation Error: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": exc.message,
            "code": exc.code,
            "level": exc.level.value,
            "category": exc.category.value
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred",
            "message": str(exc) if config.debug else None
        }
    )

# Create tables if they don't exist
models.Base.metadata.create_all(bind=engine)

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(payment_router, prefix="/payments", tags=["Payments"])
app.include_router(marketplace_router, prefix="/marketplace", tags=["Marketplace"])
app.include_router(ai_router, prefix="/ai", tags=["AI"])
app.include_router(subscription_router, prefix="/subscriptions", tags=["Subscriptions"])

# Mount static files
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint
@app.get("/")
def read_root():
    """Root endpoint returning system information."""
    return {
        "message": "Welcome to the Th.ink AR Tattoo Visualization System API",
        "version": "1.0.0",
        "status": "online",
        "environment": config.env.value,
        "docs": "/docs" if not config.is_production else None
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check database connection
    db_status = "healthy"
    try:
        # Get database session to check connection
        db = next(dependencies.get_db())
        db.execute("SELECT 1")
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
        "timestamp": time.time()
    }

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom schema components
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    # Add security requirement to all routes
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["security"] = [{"bearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Start metrics server in separate thread
def start_metrics_server():
    """Start Prometheus metrics server."""
    if config.monitoring.prometheus.get("enabled", False):
        port = config.monitoring.prometheus.get("port", 9090)
        logger.info(f"Starting metrics server on port {port}")
        start_http_server(port)

# Main entry point
def main():
    """Main entry point for the server application."""
    logger.info(f"Starting Th.ink AR server in {config.env.value} environment")
    
    # Connect to database
    connect_to_db()
    
    # Start metrics server in background thread
    if config.monitoring.prometheus.get("enabled", False):
        threading.Thread(target=start_metrics_server, daemon=True).start()
    
    # Create directories if they don't exist
    directories = ["static", "static/tattoos", "static/uploads", "static/temp", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # Start the server
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        reload=config.is_development,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()