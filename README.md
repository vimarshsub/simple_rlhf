# Simple RLHF System

A simple Reinforcement Learning from Human Feedback (RLHF) system for Mistral LLM, designed for deployment on Koyeb.

## Overview

This system allows you to experiment with RLHF using the Mistral model. It provides:

- Content generation with Mistral
- CSV-based feedback collection
- RLHF training with TRL
- Web interface for all operations

## Features

- Simple CSV format for feedback
- Example CSV generation
- Transparent training process
- Educational design for learning RLHF
- Optimized for NVIDIA L4 GPU on Koyeb

## Files

- `feedback_db.py` - Database module for CSV import/export and feedback storage
- `rlhf_trainer.py` - RLHF implementation using TRL
- `gradio_app.py` - Web interface for generation, feedback, and training
- `Dockerfile` - Ready for deployment on Koyeb
- `requirements.txt` - All necessary dependencies
- `deployment_instructions.md` - Step-by-step guide for deploying on Koyeb

## Deployment

See `deployment_instructions.md` for detailed deployment steps.

## How It Works

1. **Generate Content**: Use the Mistral model to generate content
2. **Provide Feedback**: Rate outputs in a CSV file and upload it
3. **Train RLHF Model**: Train a reward model and fine-tune Mistral
4. **Use RLHF Model**: Generate new content with your fine-tuned model

## Requirements

- Koyeb account
- NVIDIA L4 GPU (as specified in your requirements)

## License

This project is provided for educational purposes.
