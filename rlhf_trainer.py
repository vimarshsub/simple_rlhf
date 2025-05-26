"""
TRL integration module for simple RLHF system
Handles reinforcement learning from human feedback using TRL library
"""

import os
import torch
import json
import logging
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl import RewardTrainer, RewardConfig
from datasets import Dataset

from feedback_db import FeedbackDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default model configuration
MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # Base model
REWARD_MODEL_NAME = "reward_model"  # Name for saved reward model
RLHF_MODEL_NAME = "rlhf_model"  # Name for saved RLHF model

class RLHFTrainer:
    """
    Trainer for Reinforcement Learning from Human Feedback (RLHF).
    Uses TRL library to implement RLHF with Mistral model.
    """
    
    def __init__(self, model_name=MODEL_NAME, output_dir="models"):
        """
        Initialize the RLHF trainer.
        
        Args:
            model_name (str): Name or path of the base model
            output_dir (str): Directory to save trained models
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.reward_model_path = os.path.join(output_dir, REWARD_MODEL_NAME)
        self.rlhf_model_path = os.path.join(output_dir, RLHF_MODEL_NAME)
        
        # Create output directories
        os.makedirs(self.reward_model_path, exist_ok=True)
        os.makedirs(self.rlhf_model_path, exist_ok=True)
        
        logger.info(f"Initialized RLHF trainer with model {model_name}")
    
    def setup_tokenizer(self):
        """
        Set up the tokenizer for the model.
        
        Returns:
            The configured tokenizer
        """
        # Get token from environment variable
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def setup_model(self, quantize=True):
        """
        Set up the model with quantization for memory efficiency.
        
        Args:
            quantize (bool): Whether to use quantization
            
        Returns:
            The configured model and tokenizer
        """
        tokenizer = self.setup_tokenizer()
        
        # Get token from environment variable
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        
        # Configure quantization for memory efficiency on L4 GPU
        if quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                token=hf_token
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token
            )
        
        return model, tokenizer
    
    def prepare_dataset_from_db(self, db_path="data/feedback.db", min_rating=None):
        """
        Prepare a dataset from the feedback database.
        
        Args:
            db_path (str): Path to the feedback database
            min_rating (int, optional): Minimum rating threshold
            
        Returns:
            Dataset: HuggingFace dataset for training
        """
        # Connect to database
        db = FeedbackDatabase(db_path)
        
        # Get training data
        training_data = db.get_training_data(min_rating)
        db.close()
        
        if not training_data:
            logger.warning("No training data available")
            return None
        
        # Convert to intermediate dataset format
        df = pd.DataFrame({
            "prompt": [item["prompt"] for item in training_data],
            "output": [item["completion"] for item in training_data],
            "rating": [item["rating"] for item in training_data]
        })
        
        # Transform to preference pairs format using the same logic as CSV preparation
        return self._transform_to_preference_pairs(df)
    
    def _transform_to_preference_pairs(self, df):
        """
        Transform a dataframe with 'prompt', 'output', 'rating' columns
        into a dataset with 'chosen' and 'rejected' columns for RewardTrainer.
        
        Args:
            df (pd.DataFrame): DataFrame with prompt, output, rating columns
            
        Returns:
            Dataset: HuggingFace dataset with chosen and rejected columns
        """
        try:
            # Sort by prompt and rating to group similar prompts
            df = df.sort_values(by=['prompt', 'rating'], ascending=[True, False])
            
            chosen_list = []
            rejected_list = []
            
            # Group by prompt to create preference pairs
            for prompt, group in df.groupby('prompt'):
                if len(group) >= 2:
                    # If we have multiple responses for the same prompt, use the highest rated as chosen
                    # and the lowest rated as rejected
                    chosen = group.iloc[0]['output']  # Highest rated (due to sorting)
                    rejected = group.iloc[-1]['output']  # Lowest rated
                    chosen_list.append(chosen)
                    rejected_list.append(rejected)
                else:
                    # If we only have one response, create a synthetic rejected response
                    # by slightly modifying the original response
                    chosen = group.iloc[0]['output']
                    
                    # Create a synthetic rejected response by truncating or modifying the chosen one
                    if len(chosen) > 50:
                        # Truncate to create a less complete answer
                        rejected = chosen[:len(chosen)//2] + "..."
                    else:
                        # For short responses, add a disclaimer that makes it less helpful
                        rejected = "I'm not sure, but maybe: " + chosen
                    
                    chosen_list.append(chosen)
                    rejected_list.append(rejected)
            
            # Convert to HuggingFace dataset with the required 'chosen' and 'rejected' columns
            dataset_dict = {
                "chosen": chosen_list,
                "rejected": rejected_list
            }
            
            dataset = Dataset.from_dict(dataset_dict)
            logger.info(f"Prepared preference dataset with {len(dataset)} examples for reward modeling")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error transforming to preference pairs: {str(e)}")
            return None
    
    def prepare_dataset_from_csv(self, csv_path):
        """
        Prepare a dataset from a CSV file.
        
        Args:
            csv_path (str): Path to the CSV file
            
        Returns:
            Dataset: HuggingFace dataset for training
        """
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            required_columns = ['prompt', 'output', 'rating']
            
            # Validate CSV format
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"CSV file missing required column: {col}")
                    return None
            
            # Transform to preference pairs format
            return self._transform_to_preference_pairs(df)
        
        except Exception as e:
            logger.error(f"Error preparing dataset from CSV: {str(e)}")
            return None
    
    def train_reward_model(self, dataset, output_dir=None):
        """
        Train a reward model based on human feedback.
        
        Args:
            dataset: HuggingFace dataset with prompts, completions, and ratings
            output_dir (str, optional): Directory to save the model
            
        Returns:
            Success status
        """
        if output_dir is None:
            output_dir = self.reward_model_path
        
        logger.info("Setting up reward model training...")
        
        try:
            # Set up model and tokenizer
            model, tokenizer = self.setup_model()
            
            # Apply LoRA for parameter-efficient fine-tuning
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            
            # Prepare model for training
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)
            
            # Configure reward training
            training_args = RewardConfig(
                output_dir=output_dir,
                num_train_epochs=3,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                gradient_checkpointing=True,
                learning_rate=1e-5,
                report_to="none",
                remove_unused_columns=False,
                optim="paged_adamw_32bit",
                lr_scheduler_type="cosine",
                warmup_ratio=0.1,
            )
            
            # Initialize reward trainer with processing_class parameter
            trainer = RewardTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                processing_class=tokenizer,
            )
            
            # Train the model
            trainer.train()
            
            # Save the trained model
            trainer.save_model(output_dir)
            logger.info(f"Reward model trained and saved to {output_dir}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error training reward model: {str(e)}")
            return False
    
    def train_with_rlhf(self, dataset, reward_model_path=None, output_dir=None):
        """
        Train a model with RLHF using PPO.
        
        Args:
            dataset: HuggingFace dataset with prompts
            reward_model_path (str, optional): Path to the reward model
            output_dir (str, optional): Directory to save the model
            
        Returns:
            Success status
        """
        if reward_model_path is None:
            reward_model_path = self.reward_model_path
        
        if output_dir is None:
            output_dir = self.rlhf_model_path
        
        logger.info("Setting up RLHF training...")
        
        try:
            # Get token from environment variable
            hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
            
            # Set up tokenizer
            tokenizer = self.setup_tokenizer()
            
            # Load base model with value head for PPO
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token
            )
            
            # Configure LoRA for parameter-efficient fine-tuning
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            
            # Prepare model for training
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)
            
            # Load reward model
            reward_model = AutoModelForCausalLM.from_pretrained(
                reward_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token
            )
            
            # Create reward function
            def reward_function(samples):
                inputs = tokenizer(samples, return_tensors="pt", padding=True).to(reward_model.device)
                with torch.no_grad():
                    outputs = reward_model(**inputs)
                rewards = outputs.logits[:, -1].cpu().tolist()
                return rewards
            
            # Configure PPO training
            ppo_config = PPOConfig(
                learning_rate=1.5e-5,
                batch_size=8,
                mini_batch_size=1,
                gradient_accumulation_steps=4,
                optimize_cuda_cache=True,
                early_stopping=True,
                target_kl=0.1,
                ppo_epochs=4,
                seed=42,
                init_kl_coef=0.2,
                adap_kl_ctrl=True,
                model_name=output_dir
            )
            
            # Initialize PPO trainer
            ppo_trainer = PPOTrainer(
                config=ppo_config,
                model=model,
                tokenizer=tokenizer,
                reward_function=reward_function
            )
            
            # Extract prompts
            prompts = dataset["prompt"]
            
            # Training loop
            for epoch in range(3):  # 3 epochs
                logger.info(f"Starting epoch {epoch+1}/3")
                
                for i, prompt in enumerate(prompts):
                    # Tokenize prompt
                    encoded_prompt = tokenizer(prompt, return_tensors="pt").to(ppo_trainer.model.device)
                    
                    # Generate response
                    response = ppo_trainer.generate(encoded_prompt.input_ids, max_new_tokens=256)
                    response_decoded = tokenizer.decode(response[0])
                    
                    # Compute reward
                    reward = ppo_trainer.reward_function([response_decoded])[0]
                    
                    # Run PPO step
                    ppo_trainer.step([prompt], [response_decoded], [reward])
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i+1}/{len(prompts)} prompts in epoch {epoch+1}")
                
                # Save checkpoint after each epoch
                ppo_trainer.save_pretrained(f"{output_dir}/checkpoint-epoch-{epoch+1}")
            
            # Save final model
            ppo_trainer.save_pretrained(output_dir)
            logger.info(f"RLHF training completed and model saved to {output_dir}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error during RLHF training: {str(e)}")
            return False
    
    def generate_text(self, prompt, model_path=None, max_new_tokens=512, temperature=0.7):
        """
        Generate text using a trained model.
        
        Args:
            prompt (str): Input prompt
            model_path (str, optional): Path to the model
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            
        Returns:
            str: Generated text
        """
        try:
            # Get token from environment variable
            hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
            
            # Determine which model to use
            if model_path is None:
                # Always use the base Hugging Face model if no specific model is provided
                model_path = self.model_name
            
            logger.info(f"Generating text using model: {model_path}")
            
            # Load model and tokenizer
            try:
                # Try to load the model with authentication
                tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
                tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    token=hf_token
                )
            except Exception as model_error:
                # If loading fails, log the error and re-raise
                logger.error(f"Failed to load model from {model_path}: {str(model_error)}")
                raise
            
            # Generate text
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=(temperature > 0)
                )
            
            # Decode and clean up output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from output if it's included
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
        
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return f"Error generating text: {str(e)}"
    
    def run_full_training_pipeline(self, data_source, min_rating=3):
        """
        Run the complete RLHF training pipeline.
        
        Args:
            data_source (str): Path to database or CSV file
            min_rating (int): Minimum rating threshold
            
        Returns:
            Success status
        """
        logger.info(f"Starting full RLHF training pipeline with data from {data_source}")
        
        # Prepare dataset
        if data_source.endswith('.csv'):
            dataset = self.prepare_dataset_from_csv(data_source)
        else:
            dataset = self.prepare_dataset_from_db(data_source, min_rating)
        
        if dataset is None or len(dataset) == 0:
            logger.error("No training data available")
            return False
        
        # Train reward model
        reward_success = self.train_reward_model(dataset)
        if not reward_success:
            logger.error("Reward model training failed")
            return False
        
        # Train with RLHF
        rlhf_success = self.train_with_rlhf(dataset)
        if not rlhf_success:
            logger.error("RLHF training failed")
            return False
        
        logger.info("Full RLHF training pipeline completed successfully")
        return True

# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = RLHFTrainer()
    
    # Option 1: Train from database
    # trainer.run_full_training_pipeline("data/feedback.db")
    
    # Option 2: Train from CSV
    # trainer.run_full_training_pipeline("data/feedback.csv")
    
    # Generate text with trained model
    prompt = "Write a troubleshooting runbook for when a web server returns 503 errors"
    output = trainer.generate_text(prompt)
    print(f"Generated text: {output}")
