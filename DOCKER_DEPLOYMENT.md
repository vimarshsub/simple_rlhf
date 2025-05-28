# Docker Deployment Instructions for Simple RLHF

This document provides instructions for deploying the Simple RLHF application using Docker, which resolves dependency and build issues.

## Prerequisites

- Docker installed on your local machine or deployment environment
- Git to clone/pull the repository

## Building the Docker Image

1. Clone the repository (if you haven't already):
   ```bash
   git clone https://github.com/vimarshsub/simple_rlhf.git
   cd simple_rlhf
   ```

2. Build the Docker image:
   ```bash
   docker build -t simple_rlhf:latest .
   ```

3. Run the container locally to test (optional):
   ```bash
   docker run -p 8080:8080 simple_rlhf:latest
   ```
   Then access the application at http://localhost:8080

## Deploying to Koyeb with Docker

1. Push your Docker image to a container registry (Docker Hub, GitHub Container Registry, etc.)
   ```bash
   # For Docker Hub
   docker tag simple_rlhf:latest yourusername/simple_rlhf:latest
   docker push yourusername/simple_rlhf:latest
   ```

2. Deploy on Koyeb using the Docker image:
   - Log in to your Koyeb account
   - Create a new service
   - Select "Docker Registry" as the deployment method
   - Enter your Docker image URL (e.g., yourusername/simple_rlhf:latest)
   - Set the port to 8080
   - Deploy the service

## Alternative Deployment Options

### Deploy to Heroku

```bash
# Install Heroku CLI if needed
heroku container:login
heroku create your-app-name
heroku container:push web -a your-app-name
heroku container:release web -a your-app-name
```

### Deploy to Google Cloud Run

```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build and push the image to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/simple-rlhf

# Deploy to Cloud Run
gcloud run deploy simple-rlhf --image gcr.io/YOUR_PROJECT_ID/simple-rlhf --platform managed
```

## Troubleshooting

- If you encounter memory issues during training, you may need to increase the container's memory limit
- For persistent storage of models and data, consider mounting volumes to the Docker container

## Why Docker?

This Docker-based approach resolves several issues:
1. Ensures the correct Python version (3.10) is used
2. Pins all dependencies to compatible versions (trl==0.7.0, transformers==4.31.0)
3. Avoids Rust compiler requirements by using pre-built wheels
4. Creates a reproducible environment that works consistently across different platforms
