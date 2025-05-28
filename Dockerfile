FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
COPY requirements-fixed.txt .

# Install Python dependencies with exact versions
RUN pip install --no-cache-dir -r requirements-fixed.txt
RUN pip install --no-cache-dir trl==0.7.0 transformers==4.31.0 tokenizers==0.13.3

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models

# Set environment variables
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Run the application
CMD ["python", "gradio_app.py"]
