# Deploying Simple RLHF System on Koyeb

This guide provides step-by-step instructions for deploying the Simple RLHF System on Koyeb, allowing you to experiment with Reinforcement Learning from Human Feedback using the Mistral model.

## Overview

This deployment will set up a web interface where you can:
1. Generate content using the Mistral model
2. Upload CSV files with feedback ratings
3. Train an RLHF model based on your feedback
4. Generate new content with the RLHF-trained model

## Prerequisites

- A Koyeb account (sign up at [koyeb.com](https://www.koyeb.com) if you don't have one)
- Git installed on your local machine (for repository setup)

## Step 1: Prepare Your Repository

First, create a Git repository with all the necessary files:

1. Create a new directory for your project:
   ```bash
   mkdir simple-rlhf
   cd simple-rlhf
   git init
   ```

2. Copy all the provided files into this directory:
   - `requirements.txt`
   - `feedback_db.py`
   - `rlhf_trainer.py`
   - `gradio_app.py`

3. Create a `Dockerfile` with the following content:
   ```dockerfile
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
   ```

4. Create a `.dockerignore` file:
   ```
   __pycache__/
   *.py[cod]
   *$py.class
   .git/
   .env
   venv/
   data/*.csv
   data/*.json
   models/
   logs/
   ```

5. Commit your files:
   ```bash
   git add .
   git commit -m "Initial commit of Simple RLHF System"
   ```

## Step 2: Create a GitHub Repository

1. Create a new repository on GitHub (or your preferred Git hosting service)

2. Push your local repository:
   ```bash
   git remote add origin https://github.com/yourusername/simple-rlhf.git
   git branch -M main
   git push -u origin main
   ```

## Step 3: Deploy on Koyeb

### Option 1: Using the Koyeb Web Interface

1. Log in to your Koyeb account at [app.koyeb.com](https://app.koyeb.com)

2. Click on "Create App"

3. Select "GitHub" as the deployment method

4. Connect your GitHub account if not already connected

5. Select your repository and branch (main)

6. Configure the deployment:
   - **Name**: `simple-rlhf` (or your preferred name)
   - **Region**: Choose the region closest to you
   - **Instance Type**: Select "L4 GPU" (as you mentioned having an NVIDIA L4)
   - **Environment Variables**: None required for basic setup
   - **Port**: 8080

7. Click "Deploy"

### Option 2: Using the Koyeb CLI

1. Install the Koyeb CLI if you haven't already:
   ```bash
   curl -fsSL https://cli.koyeb.com/install.sh | sh
   ```

2. Log in to your Koyeb account:
   ```bash
   koyeb login
   ```

3. Deploy your application:
   ```bash
   koyeb app create simple-rlhf \
     --git github.com/yourusername/simple-rlhf \
     --git-branch main \
     --ports 8080:http \
     --routes /:8080 \
     --instance-type l4
   ```

## Step 4: Access Your Application

1. Once deployed, Koyeb will provide you with a URL to access your application (e.g., `https://simple-rlhf-yourusername.koyeb.app`)

2. Open this URL in your browser to access the Simple RLHF System

## Step 5: Using the RLHF System

### Generate Content

1. Go to the "Generate Content" tab
2. Enter a prompt (e.g., "Write a troubleshooting runbook for when a web server returns 503 errors")
3. Click "Generate Content"
4. The model will generate a response based on your prompt

### Provide Feedback via CSV

1. Go to the "Feedback Management" tab
2. Click "Download Example CSV" to get a template
3. Edit the CSV file with your own prompts, outputs, and ratings (1-5)
4. Upload your CSV file and click "Upload and Process"

### Train the RLHF Model

1. Go to the "Train RLHF Model" tab
2. Set the minimum rating threshold (e.g., 3)
3. Click "Start Training"
4. Wait for the training to complete (this may take some time)

### Generate with RLHF Model

1. Return to the "Generate Content" tab
2. Check the "Use RLHF Model" option
3. Enter a prompt and click "Generate Content"
4. The RLHF-trained model will generate a response

## Advanced Configuration

### Persistent Storage

By default, Koyeb apps don't have persistent storage. To preserve your data between deployments:

1. In the Koyeb dashboard, go to your app settings
2. Add a volume mount:
   - Path: `/app/data`
   - Size: 1GB (or as needed)

### Custom Domain

To use a custom domain:

1. In the Koyeb dashboard, go to your app settings
2. Under "Domains", add your custom domain
3. Follow the instructions to configure DNS settings

## Troubleshooting

### Memory Issues

If you encounter memory errors:

1. Edit `rlhf_trainer.py` to reduce batch sizes or use more aggressive quantization
2. Commit and push changes to trigger a redeployment

### Training Takes Too Long

For faster training:

1. Reduce the number of epochs in `rlhf_trainer.py`
2. Use a smaller subset of your feedback data

### Application Crashes

If the application crashes:

1. Check the logs in the Koyeb dashboard
2. Common issues include:
   - Out of memory errors (reduce batch size)
   - Missing directories (ensure all required directories exist)
   - GPU issues (verify L4 GPU is properly allocated)

## Understanding the Code

- `feedback_db.py`: Handles CSV import/export and feedback storage
- `rlhf_trainer.py`: Implements the RLHF training pipeline using TRL
- `gradio_app.py`: Creates the web interface for interaction

## Next Steps

After getting familiar with the basic RLHF workflow:

1. Experiment with different types of prompts and feedback
2. Adjust training parameters to see their effect on the model
3. Compare outputs from the base model and the RLHF-trained model
4. Create more structured feedback datasets for specific tasks

## Resources

- [Koyeb Documentation](https://www.koyeb.com/docs)
- [TRL Library Documentation](https://huggingface.co/docs/trl/)
- [Mistral Model Card](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [RLHF Paper](https://arxiv.org/abs/2203.02155)
