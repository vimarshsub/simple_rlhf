FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p data models/reward_model models/rlhf_model logs

# Expose port for the web interface
EXPOSE 8080

# Set environment variables
ENV PORT=8080

# Start the application
CMD ["python", "gradio_app.py"]
