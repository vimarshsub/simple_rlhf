"""
Gradio interface for simple RLHF system
Provides a web interface for generating content, uploading feedback, and training
"""

print("\n\n")
print("*" * 80)
print("*" * 80)
print("*** STARTING GRADIO APP - LATEST CODE VERSION WITH DEBUG PRINTS ***")
print("*** DEPLOYMENT TIMESTAMP:", __import__('datetime').datetime.now(), "***")
print("*" * 80)
print("*" * 80)
print("\n\n")

import sys
import os
import importlib.metadata
import tempfile

# Create necessary directories for Gradio
try:
    # Create /tmp directory if it doesn't exist
    os.makedirs("/tmp", exist_ok=True)
    
    # Create /tmp/gradio directory for Gradio file uploads
    os.makedirs("/tmp/gradio", exist_ok=True)
    
    # Set permissions to ensure it's writable
    os.chmod("/tmp/gradio", 0o777)
    
    print(f"Created and verified /tmp/gradio directory for file uploads")
except Exception as e:
    print(f"Warning: Could not create /tmp/gradio directory: {str(e)}")
    # Try alternative locations
    try:
        # Use the current directory as fallback
        alt_dir = os.path.join(os.getcwd(), "gradio_tmp")
        os.makedirs(alt_dir, exist_ok=True)
        # Set environment variable to tell Gradio to use this directory
        os.environ["GRADIO_TEMP_DIR"] = alt_dir
        print(f"Using alternative temp directory: {alt_dir}")
    except Exception as alt_e:
        print(f"Warning: Could not create alternative temp directory: {str(alt_e)}")

# Check versions before importing anything else
try:
    trl_version = importlib.metadata.version('trl')
    transformers_version = importlib.metadata.version('transformers')
    
    print("\n\n")
    print("!" * 80)
    print(f"!!! DETECTED VERSIONS: trl=={trl_version}, transformers=={transformers_version} !!!")
    print("!!! REQUIRED VERSIONS: trl==0.7.0, transformers==4.31.0 !!!")
    print("!" * 80)
    print("\n\n")
    
    # Version warning only - no reinstall or restart
    if trl_version != "0.7.0" or transformers_version != "4.31.0":
        print(f"WARNING: Detected non-optimal versions: trl=={trl_version}, transformers=={transformers_version}")
        print("Recommended: trl==0.7.0, transformers==4.31.0")
        print("Continuing with current versions...")
except Exception as e:
    print(f"Error checking versions: {str(e)}")

import gradio as gr
import pandas as pd
import torch
import logging
from datetime import datetime

from feedback_db import FeedbackDatabase
from rlhf_trainer import RLHFTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize database and trainer
db = FeedbackDatabase()
trainer = RLHFTrainer()

def generate_content(prompt, use_rlhf_model=False, max_length=512, temperature=0.7):
    """Generate content using the model"""
    try:
        # Determine which model to use
        model_path = None
        if use_rlhf_model:
            rlhf_path = os.path.join("models", "rlhf_model")
            if os.path.exists(rlhf_path):
                model_path = rlhf_path
            else:
                logger.warning("RLHF model not found, falling back to base model")
                # Continue with model_path as None, which will use the base model
        
        # Generate text
        output = trainer.generate_text(
            prompt, 
            model_path=model_path,
            max_new_tokens=max_length,
            temperature=temperature
        )
        
        if output is None:
            return "Failed to generate content. Please try again with different parameters."
        
        # Only store in database if generation was successful
        if not output.startswith("Error generating text:"):
            try:
                db.add_prompt_output(prompt, output)
            except Exception as db_error:
                logger.error(f"Error storing in database: {str(db_error)}")
                # Continue even if database storage fails
        
        return output
    
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        return f"Error generating content: {str(e)}"

def upload_feedback_csv(csv_file):
    """Upload and process feedback CSV file"""
    try:
        if csv_file is None:
            return "No file uploaded."
        
        # Save uploaded file
        os.makedirs("data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"data/feedback_{timestamp}.csv"
        
        # Read and validate CSV
        df = pd.read_csv(csv_file.name)
        required_columns = ['prompt', 'output', 'rating']
        
        for col in required_columns:
            if col not in df.columns:
                return f"CSV file missing required column: {col}"
        
        # Save a copy
        df.to_csv(csv_path, index=False)
        
        # Import to database
        count = db.import_from_csv(csv_path)
        
        return f"Successfully imported {count} feedback records from CSV."
    
    except Exception as e:
        logger.error(f"Error uploading feedback CSV: {str(e)}")
        return f"Error uploading feedback CSV: {str(e)}"

def export_example_csv():
    """Generate and export an example CSV file"""
    try:
        csv_path = "data/example_feedback.csv"
        os.makedirs("data", exist_ok=True)
        
        db.generate_example_csv(csv_path)
        
        return csv_path
    
    except Exception as e:
        logger.error(f"Error generating example CSV: {str(e)}")
        return None

def train_rlhf_model(min_rating, progress=gr.Progress()):
    """Train the RLHF model using feedback data"""
    try:
        progress(0, desc="Starting training...")
        
        # Check if we have feedback data
        data = db.get_training_data(min_rating=min_rating)
        if not data or len(data) == 0:
            return "No training data available. Please upload feedback CSV first."
        
        progress(0.1, desc="Preparing dataset...")
        
        # Export data for training
        json_path = "data/training_data.json"
        db.export_training_data_json(json_path, min_rating=min_rating)
        
        progress(0.2, desc="Training reward model...")
        
        # Prepare dataset
        dataset = trainer.prepare_dataset_from_db(db_path=db.db_path, min_rating=min_rating)
        if dataset is None:
            return "Failed to prepare dataset."
        
        # Train reward model
        progress(0.3, desc="Training reward model...")
        reward_success = trainer.train_reward_model(dataset)
        if not reward_success:
            return "Reward model training failed."
        
        progress(0.6, desc="Training RLHF model...")
        
        # Train with RLHF
        rlhf_success = trainer.train_with_rlhf(dataset)
        if not rlhf_success:
            return "RLHF training failed."
        
        progress(1.0, desc="Training complete!")
        
        return "RLHF training completed successfully! You can now generate content using the trained model."
    
    except Exception as e:
        logger.error(f"Error training RLHF model: {str(e)}")
        return f"Error training RLHF model: {str(e)}"

def view_feedback_data(min_rating=None):
    """View feedback data in the database"""
    try:
        # Get data from database
        data = db.get_all_prompt_outputs(with_feedback_only=(min_rating is not None))
        
        if min_rating is not None:
            data = [item for item in data if item['rating'] is not None and item['rating'] >= min_rating]
        
        if not data:
            return "No feedback data available."
        
        # Format as markdown table
        table = "| Prompt | Output | Rating | Comments |\n"
        table += "| ------ | ------ | ------ | -------- |\n"
        
        for item in data[:10]:  # Limit to 10 items for display
            prompt = item['prompt'][:50] + "..." if len(item['prompt']) > 50 else item['prompt']
            output = item['output'][:50] + "..." if len(item['output']) > 50 else item['output']
            rating = item['rating'] if item['rating'] is not None else "N/A"
            comments = item['comments'] if item['comments'] else ""
            
            table += f"| {prompt} | {output} | {rating} | {comments} |\n"
        
        if len(data) > 10:
            table += f"\n... and {len(data) - 10} more items."
        
        return table
    
    except Exception as e:
        logger.error(f"Error viewing feedback data: {str(e)}")
        return f"Error viewing feedback data: {str(e)}"

def create_interface():
    """Create the Gradio interface"""
    # Configure Gradio cache directory
    if "GRADIO_TEMP_DIR" in os.environ:
        gr.paths.set_temp_dir(os.environ["GRADIO_TEMP_DIR"])
    
    with gr.Blocks(title="Simple RLHF System") as interface:
        gr.Markdown("# Simple RLHF System for Mistral")
        gr.Markdown("This system allows you to experiment with Reinforcement Learning from Human Feedback (RLHF) using the Mistral model.")
        
        with gr.Tab("Generate Content"):
            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(
                        label="Input Prompt",
                        placeholder="Write a troubleshooting runbook for when a web server returns 503 errors",
                        lines=5
                    )
                    
                    with gr.Row():
                        use_rlhf = gr.Checkbox(label="Use RLHF Model (if available)", value=False)
                        max_length = gr.Slider(
                            minimum=64, maximum=1024, value=512, step=64,
                            label="Maximum Length"
                        )
                        temperature = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.7, step=0.1,
                            label="Temperature"
                        )
                    
                    generate_btn = gr.Button("Generate Content")
                
                with gr.Column():
                    response_output = gr.Textbox(
                        label="Generated Content",
                        lines=20
                    )
            
            generate_btn.click(
                fn=generate_content,
                inputs=[prompt_input, use_rlhf, max_length, temperature],
                outputs=response_output
            )
        
        with gr.Tab("Feedback Management"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Upload Feedback CSV")
                    gr.Markdown("Upload a CSV file with feedback data. The CSV should have columns: `prompt`, `output`, `rating`, and optionally `comments`.")
                    
                    csv_upload = gr.File(
                        label="Upload Feedback CSV",
                        file_types=[".csv"]
                    )
                    
                    upload_btn = gr.Button("Upload and Process")
                    upload_result = gr.Textbox(label="Upload Result")
                    
                    gr.Markdown("## Example CSV")
                    gr.Markdown("Download an example CSV file to see the required format.")
                    example_btn = gr.Button("Download Example CSV")
                    example_output = gr.File(label="Example CSV")
                
                with gr.Column():
                    gr.Markdown("## View Feedback Data")
                    min_rating_filter = gr.Slider(
                        minimum=1, maximum=5, value=None, step=1,
                        label="Minimum Rating (leave at 0 to show all)"
                    )
                    view_btn = gr.Button("View Feedback Data")
                    feedback_display = gr.Markdown(label="Feedback Data")
            
            upload_btn.click(
                fn=upload_feedback_csv,
                inputs=csv_upload,
                outputs=upload_result
            )
            
            example_btn.click(
                fn=export_example_csv,
                inputs=None,
                outputs=example_output
            )
            
            view_btn.click(
                fn=view_feedback_data,
                inputs=min_rating_filter,
                outputs=feedback_display
            )
        
        with gr.Tab("Train RLHF Model"):
            gr.Markdown("## Train RLHF Model")
            gr.Markdown("Train the RLHF model using the feedback data you've uploaded.")
            
            min_rating_train = gr.Slider(
                minimum=1, maximum=5, value=3, step=1,
                label="Minimum Rating for Training"
            )
            
            train_btn = gr.Button("Start Training")
            train_output = gr.Textbox(label="Training Status")
            
            train_btn.click(
                fn=train_rlhf_model,
                inputs=min_rating_train,
                outputs=train_output
            )
        
        with gr.Tab("About RLHF"):
            gr.Markdown("""
            ## About Reinforcement Learning from Human Feedback (RLHF)
            
            RLHF is a technique for fine-tuning language models using human feedback. The process involves:
            
            1. **Initial Generation**: The base model generates responses to prompts
            2. **Human Feedback**: Humans rate the quality of these responses
            3. **Reward Model**: A model is trained to predict human ratings
            4. **Policy Optimization**: The base model is fine-tuned to maximize the predicted reward
            
            ### How This System Works
            
            1. **Generate Content**: Use the Mistral model to generate content
            2. **Provide Feedback**: Upload a CSV file with ratings for generated content
            3. **Train RLHF Model**: Train a reward model and fine-tune the base model
            4. **Use RLHF Model**: Generate new content with the fine-tuned model
            
            ### CSV Format
            
            The feedback CSV should have the following columns:
            - `prompt`: The input prompt
            - `output`: The model's output
            - `rating`: A numerical rating (1-5)
            - `comments`: (Optional) Additional feedback
            
            ### References
            
            - [TRL Library Documentation](https://huggingface.co/docs/trl/)
            - [Mistral Model](https://huggingface.co/mistralai/Mistral-7B-v0.1)
            - [RLHF Paper](https://arxiv.org/abs/2203.02155)
            """)
    
    return interface

if __name__ == "__main__":
    # Print versions again right before launching
    try:
        trl_version = importlib.metadata.version('trl')
        transformers_version = importlib.metadata.version('transformers')
        print("\n\n")
        print("#" * 80)
        print(f"### FINAL VERSIONS BEFORE LAUNCH: trl=={trl_version}, transformers=={transformers_version} ###")
        print("#" * 80)
        print("\n\n")
    except Exception as e:
        print(f"Error checking final versions: {str(e)}")
        
    # Create and launch the interface
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))
