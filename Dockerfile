# Dockerfile for Th.ink AR Tattoo Visualizer
# Multi-stage build for optimized production image


# ===== BUILDER STAGE =====
FROM python:3.11-slim AS builder

# Set build arguments and environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* || exit 1

# Set working directory
WORKDIR /build

# Copy only requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary source files
COPY setup.py .
COPY README.md .
COPY src/ ./src/
COPY server/ ./server/
COPY config/ ./config/

# Build the package
RUN pip install --no-cache-dir -e .

# ===== RUNTIME STAGE =====
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    THINK_ENV=production \
    PYTHONPATH=/app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* || exit 1

# Set working directory
WORKDIR /app

# Copy built package and dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /build/src /app/src
COPY --from=builder /build/server /app/server
COPY --from=builder /build/config /app/config

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data /app/models /app/static/tattoos && \
    chmod -R 755 /app

# Expose port for the FastAPI server
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint and default command
ENTRYPOINT ["python"]
CMD ["-m", "server.app"]

# Label metadata
LABEL maintainer="thInk Team <contact@think-ar.dev>" \
      version="1.0.0" \
      description="Th.ink AR Tattoo Visualizer"