# Use the official Python image as the base
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install dependencies
COPY server/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY server/ ./server/

# Expose port
EXPOSE 8000

# Define environment variables for secrets
ENV STRIPE_SECRET_KEY=your-stripe-secret-key
ENV STRIPE_WEBHOOK_SECRET=your-webhook-secret

# Run the application
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]