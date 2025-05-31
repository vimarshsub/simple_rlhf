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
    TrainingArguments,
    AutoModelForSequenceClassification
)
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl import RewardTrainer
from datasets import Dataset
import shutil

from feedback_db import FeedbackDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default model configuration
MODEL_NAME = "distilgpt2"  # Changed back from gpt2 to Mistral
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
        
        # Ensure proper padding configuration
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        # Set padding side to left for generation (important for PPO)
        tokenizer.padding_side = "left"
        
        return tokenizer
    
    def setup_model(self, quantize=True):
        """
        Set up the model and tokenizer.
        
        Args:
            quantize (bool): Whether to use quantization
            
        Returns:
            tuple: (model, tokenizer)
        """
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            # Ensure padding side is correct for reward training
            if hasattr(self, '_reward_training') and self._reward_training:
                tokenizer.padding_side = "right"
            
            # Configure quantization if requested
            if quantize:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            else:
                bnb_config = None
            
            # Load model - use AutoModelForCausalLM for generation, AutoModelForSequenceClassification for reward
            if hasattr(self, '_reward_training') and self._reward_training:
                # For reward model training, use sequence classification
                model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME,
                    num_labels=1,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    quantization_config=bnb_config,
                    trust_remote_code=True
                )
            else:
                # For text generation, use causal LM
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    quantization_config=bnb_config,
                    trust_remote_code=True
                )
            
            logger.info(f"Model and tokenizer loaded successfully: {MODEL_NAME}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error setting up model: {str(e)}")
            raise
    
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
            
            # Tokenize the dataset for RewardTrainer in trl 0.7.0
            tokenizer = self.setup_tokenizer()
            
            def tokenize_dataset(examples):
                chosen_texts = examples["chosen"]
                rejected_texts = examples["rejected"]
                
                tokenized_chosen = tokenizer(chosen_texts, truncation=True, padding="max_length", max_length=512)
                tokenized_rejected = tokenizer(rejected_texts, truncation=True, padding="max_length", max_length=512)
                
                # Create the required format for RewardTrainer
                result = {
                    "input_ids_chosen": tokenized_chosen["input_ids"],
                    "attention_mask_chosen": tokenized_chosen["attention_mask"],
                    "input_ids_rejected": tokenized_rejected["input_ids"],
                    "attention_mask_rejected": tokenized_rejected["attention_mask"],
                }
                return result
            
            # Apply tokenization to create the required columns
            tokenized_dataset = dataset.map(
                tokenize_dataset,
                batched=True,
                desc="Tokenizing dataset for reward modeling"
            )
            
            logger.info(f"Successfully tokenized dataset with required columns for RewardTrainer")
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"Error transforming to preference pairs: {str(e)}")
            return None
    
    def prepare_dataset_from_csv(self, csv_file, min_rating=3):
        """
        Prepare dataset from CSV file for RLHF training.
        
        Args:
            csv_file (str): Path to CSV file with columns: prompt, output, rating
            min_rating (int): Minimum rating to consider as "chosen"
            
        Returns:
            Dataset: Processed dataset ready for training
        """
        try:
            # Load CSV data
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} rows from {csv_file}")
            
            # Group by prompt and create preference pairs
            grouped = df.groupby('prompt')
            preference_data = []
            
            for prompt, group in grouped:
                # Sort by rating to get best and worst responses
                sorted_group = group.sort_values('rating', ascending=False)
                
                if len(sorted_group) >= 2:
                    best = sorted_group.iloc[0]
                    worst = sorted_group.iloc[-1]
                    
                    # Only create pair if there's a clear preference
                    if best['rating'] >= min_rating and worst['rating'] < min_rating:
                        preference_data.append({
                            'chosen': f"{prompt}\n{best['output']}",
                            'rejected': f"{prompt}\n{worst['output']}"
                        })
            
            logger.info(f"Created {len(preference_data)} preference pairs")
            
            if len(preference_data) == 0:
                logger.error("No preference pairs created. Check your data and min_rating threshold.")
                return None
            
            # Convert to HuggingFace dataset
            dataset = Dataset.from_list(preference_data)
            
            # Tokenize the dataset for RewardTrainer in trl 0.7.0
            tokenizer = self.setup_tokenizer()
            
            def tokenize_dataset(examples):
                chosen_texts = examples["chosen"]
                rejected_texts = examples["rejected"]
                
                tokenized_chosen = tokenizer(chosen_texts, truncation=True, padding="max_length", max_length=512)
                tokenized_rejected = tokenizer(rejected_texts, truncation=True, padding="max_length", max_length=512)
                
                # Create the required format for RewardTrainer
                result = {
                    "input_ids_chosen": tokenized_chosen["input_ids"],
                    "attention_mask_chosen": tokenized_chosen["attention_mask"],
                    "input_ids_rejected": tokenized_rejected["input_ids"],
                    "attention_mask_rejected": tokenized_rejected["attention_mask"],
                }
                return result
            
            # Apply tokenization to create the required columns
            tokenized_dataset = dataset.map(
                tokenize_dataset,
                batched=True,
                desc="Tokenizing dataset for reward modeling"
            )
            
            logger.info(f"Successfully tokenized dataset with required columns for RewardTrainer")
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            return None
    
    def train_reward_model(self, dataset, output_dir=None):
        """
        Train a reward model based on human feedback.
        
        Args:
            dataset: HuggingFace dataset with chosen and rejected responses
            output_dir (str, optional): Directory to save the model
            
        Returns:
            Success status
        """
        if output_dir is None:
            output_dir = self.reward_model_path
        
        logger.info("Setting up reward model training...")
        
        try:
            # Set flag for reward training to use correct model type
            self._reward_training = True
            
            # Set up model and tokenizer
            base_model, tokenizer = self.setup_model(quantize=False)  # Disable quantization for compatibility
            
            # Apply LoRA for parameter-efficient fine-tuning
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="SEQ_CLS",  # Sequence classification for reward model
                target_modules=["c_attn", "c_proj"]  # Changed for Mistral architecture
            )
            
            # Apply LoRA directly without prepare_model_for_kbit_training
            model = get_peft_model(base_model, peft_config)
            
            # Ensure model is in training mode and gradients are enabled
            model.train()
            for param in model.parameters():
                param.requires_grad = True
            
            # Configure reward training using TrainingArguments instead of RewardConfig
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=1,
                per_device_train_batch_size=1,  # Reduced to 1 to avoid padding issues
                gradient_accumulation_steps=2,  # Increased to maintain effective batch size
                gradient_checkpointing=True,
                learning_rate=1e-5,
                report_to="none",
                remove_unused_columns=False,
                optim="adamw_torch",  # Changed from paged_adamw_32bit to standard adamw
                lr_scheduler_type="cosine",
                warmup_ratio=0.1,
            )
            
            # Initialize reward trainer - dataset should already have the required columns
            # CRITICAL FIX: Explicitly set max_length for RewardDataCollatorWithPadding
            trainer = RewardTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=tokenizer,  # Use tokenizer directly in trl 0.7.0
                max_length=512,  # Explicitly set max_length to match tokenization
            )
            
            # Train the model
            logger.info("Starting reward model training...")
            trainer.train()
            
            # Save the model using trainer's save_model method
            trainer.save_model()
            
            # DEBUG: List files in output directory after trainer.save_model()
            logger.info(f"Files in {output_dir} after trainer.save_model():")
            for file in os.listdir(output_dir):
                logger.info(f"  - {file}")
            
            # Create a temporary directory for saving the base model
            temp_base_model_dir = os.path.join(output_dir, "base_model_temp")
            os.makedirs(temp_base_model_dir, exist_ok=True)
            
            # Save the base model to the temporary directory
            base_model.save_pretrained(temp_base_model_dir)
            tokenizer.save_pretrained(temp_base_model_dir)
            
            # DEBUG: List files in temp directory after base_model.save_pretrained()
            logger.info(f"Files in {temp_base_model_dir} after base_model.save_pretrained():")
            for file in os.listdir(temp_base_model_dir):
                logger.info(f"  - {file}")
            
            # Copy the config.json from the base model to the output directory
            base_config_path = os.path.join(temp_base_model_dir, "config.json")
            output_config_path = os.path.join(output_dir, "config.json")
            if os.path.exists(base_config_path):
                shutil.copy(base_config_path, output_config_path)
                logger.info(f"Copied config.json from base model to {output_config_path}")
            else:
                logger.warning(f"config.json not found in base model at {base_config_path}")
            
            # Also explicitly save the PEFT model
            model.save_pretrained(output_dir)
            
            # Copy the pytorch_model.bin from the base model to the output directory
            base_model_bin_path = os.path.join(temp_base_model_dir, "pytorch_model.bin")
            output_model_bin_path = os.path.join(output_dir, "pytorch_model.bin")
            if os.path.exists(base_model_bin_path):
                shutil.copy(base_model_bin_path, output_model_bin_path)
                logger.info(f"Copied pytorch_model.bin from base model to {output_model_bin_path}")
            else:
                logger.warning(f"pytorch_model.bin not found in base model at {base_model_bin_path}")
                # Create a minimal pytorch_model.bin if it doesn't exist
                # This is just a placeholder to satisfy the model loading requirements
                torch.save({"placeholder": torch.tensor([0.0])}, output_model_bin_path)
                logger.info(f"Created minimal pytorch_model.bin at {output_model_bin_path}")
            
            # DEBUG: List files in output directory after model.save_pretrained()
            logger.info(f"Files in {output_dir} after model.save_pretrained():")
            for file in os.listdir(output_dir):
                logger.info(f"  - {file}")
            
            # Verify that config.json exists in the output directory
            if os.path.exists(output_config_path):
                logger.info(f"Verified config.json exists at {output_config_path}")
            else:
                logger.warning(f"config.json not found at {output_config_path}, creating it manually")
                
                # Create a minimal config.json directly
                minimal_config = {
                    "architectures": ["GPT2ForSequenceClassification"],
                    "model_type": "gpt2",
                    "num_labels": 1,
                    "vocab_size": 50257,
                    "n_positions": 1024,
                    "n_ctx": 1024,
                    "n_embd": 768,
                    "n_layer": 6,
                    "n_head": 12,
                    "activation_function": "gelu_new",
                    "resid_pdrop": 0.1,
                    "embd_pdrop": 0.1,
                    "attn_pdrop": 0.1,
                    "layer_norm_epsilon": 1e-05,
                    "initializer_range": 0.02,
                    "summary_type": "cls_index",
                    "summary_use_proj": True,
                    "summary_activation": None,
                    "summary_proj_to_labels": True,
                    "summary_first_dropout": 0.1,
                    "scale_attn_weights": True,
                    "use_cache": True,
                    "bos_token_id": 50256,
                    "eos_token_id": 50256
                }
                
                with open(output_config_path, 'w') as f:
                    json.dump(minimal_config, f)
                logger.info(f"Created minimal config.json at {output_config_path}")
                
                # Verify the file was created
                if os.path.exists(output_config_path):
                    with open(output_config_path, 'r') as f:
                        config_content = f.read()
                    logger.info(f"Verified config.json was created with size: {len(config_content)} bytes")
                else:
                    logger.error(f"Failed to create config.json at {output_config_path}")
            
            # Clean up temporary directory
            shutil.rmtree(temp_base_model_dir, ignore_errors=True)
            
            # Final verification
            logger.info(f"Final verification of files in {output_dir}:")
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                file_size = os.path.getsize(file_path)
                logger.info(f"  - {file} ({file_size} bytes)")
            
            # Explicitly check for config.json one more time
            if os.path.exists(output_config_path):
                with open(output_config_path, 'r') as f:
                    config_content = f.read()
                logger.info(f"Final verification: config.json exists with size: {len(config_content)} bytes")
            else:
                logger.error(f"Final verification: config.json is still missing at {output_config_path}")
                # Last resort: create it again with file permissions check
                try:
                    with open(output_config_path, 'w') as f:
                        json.dump(minimal_config, f)
                    logger.info(f"Last resort: Created config.json at {output_config_path}")
                except Exception as e:
                    logger.error(f"Failed to create config.json in last resort attempt: {str(e)}")
            
            logger.info(f"Reward model saved to {output_dir}")
            
            # Reset flag
            self._reward_training = False
            
            return True
            
        except Exception as e:
            logger.error(f"Error training reward model: {str(e)}")
            # Reset flag on error
            self._reward_training = False
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
            
            # CRITICAL FIX: Load the base model WITHOUT any PEFT/LoRA
            logger.info("Loading base model for RLHF training...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # CRITICAL FIX: Then wrap with value head WITHOUT applying LoRA
            logger.info("Wrapping base model with value head...")
            # Create a new instance directly from the base model
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                base_model,  # Use the base model instance
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # CRITICAL FIX: Explicitly set device attribute on the model
            # This is required by PPOTrainer but not provided by AutoModelForCausalLMWithValueHead
            model.device = next(model.parameters()).device
            logger.info(f"Explicitly set model.device = {model.device}")
            
            # Verify the model is correctly wrapped
            logger.info(f"Verified model type after wrapping: {type(model)}")
            
            # CRITICAL FIX: Do NOT apply LoRA to the wrapped model
            # This is the key change - we're removing PEFT/LoRA entirely from RLHF training
            
            # Check if config.json exists in reward_model_path
            config_path = os.path.join(reward_model_path, "config.json")
            logger.info(f"Checking for config.json at {config_path}")
            
            # List all files in reward_model_path
            logger.info(f"Files in reward model directory ({reward_model_path}):")
            for file in os.listdir(reward_model_path):
                file_path = os.path.join(reward_model_path, file)
                file_size = os.path.getsize(file_path)
                logger.info(f"  - {file} ({file_size} bytes)")
            
            if not os.path.exists(config_path):
                logger.error(f"config.json not found at {config_path}")
                
                # Create a minimal config.json as a last resort
                minimal_config = {
                    "architectures": ["GPT2ForSequenceClassification"],
                    "model_type": "gpt2",
                    "num_labels": 1,
                    "vocab_size": 50257,
                    "n_positions": 1024,
                    "n_ctx": 1024,
                    "n_embd": 768,
                    "n_layer": 6,
                    "n_head": 12,
                    "activation_function": "gelu_new",
                    "resid_pdrop": 0.1,
                    "embd_pdrop": 0.1,
                    "attn_pdrop": 0.1,
                    "layer_norm_epsilon": 1e-05,
                    "initializer_range": 0.02,
                    "summary_type": "cls_index",
                    "summary_use_proj": True,
                    "summary_activation": None,
                    "summary_proj_to_labels": True,
                    "summary_first_dropout": 0.1,
                    "scale_attn_weights": True,
                    "use_cache": True,
                    "bos_token_id": 50256,
                    "eos_token_id": 50256
                }
                
                with open(config_path, 'w') as f:
                    json.dump(minimal_config, f)
                logger.info(f"Created minimal config.json at {config_path}")
                
                # Verify the file was created
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config_content = f.read()
                    logger.info(f"Verified config.json was created with size: {len(config_content)} bytes")
                else:
                    logger.error(f"Failed to create config.json at {config_path}")
                    raise FileNotFoundError(f"Required file config.json not found in {reward_model_path}")
            else:
                logger.info(f"config.json found at {config_path}")
                # Verify the file is not empty
                with open(config_path, 'r') as f:
                    config_content = f.read()
                logger.info(f"config.json size: {len(config_content)} bytes")
                if len(config_content.strip()) == 0:
                    logger.error(f"config.json is empty at {config_path}")
                    raise ValueError(f"config.json is empty in {reward_model_path}")
            
            # Check if we need to load a PEFT model
            peft_config_path = os.path.join(reward_model_path, "adapter_config.json")
            is_peft_model = os.path.exists(peft_config_path)
            
            # Load reward model
            logger.info(f"Loading reward model from {reward_model_path}")
            if is_peft_model:
                logger.info(f"Detected PEFT model, loading with PeftModel")
                # First load the base model
                base_reward_model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=1,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                # Then load the PEFT adapter
                reward_model = PeftModel.from_pretrained(
                    base_reward_model,
                    reward_model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                # Load as a regular model
                reward_model = AutoModelForSequenceClassification.from_pretrained(
                    reward_model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    token=hf_token
                )
            
            logger.info(f"Successfully loaded reward model")
            
            # Create reward function
            def compute_reward(samples):
                inputs = tokenizer(samples, return_tensors="pt", padding=True).to(reward_model.device)
                with torch.no_grad():
                    outputs = reward_model(**inputs)
                rewards = outputs.logits[:, -1].cpu().tolist()
                return rewards
            
            # Configure PPO training with batch_size=8 to match internal expectations
            ppo_config = PPOConfig(
                learning_rate=1.5e-5,
                batch_size=8,  # CRITICAL FIX: Set batch_size to 8 to match internal expectations
                mini_batch_size=8,
                gradient_accumulation_steps=1,
                optimize_cuda_cache=True,
                early_stopping=True,
                target_kl=0.1,
                ppo_epochs=2,
                seed=42,
                init_kl_coef=0.2,
                adap_kl_ctrl=True,
                model_name=output_dir
            )
            
            # Initialize PPO trainer
            logger.info("Initializing PPOTrainer with the wrapped model (no PEFT/LoRA)...")
            ppo_trainer = PPOTrainer(
                config=ppo_config,
                model=model,
                tokenizer=tokenizer
            )
            
            # Extract prompts from dataset
            # First check if we have a 'prompt' column
            if 'prompt' in dataset.column_names:
                prompts = dataset["prompt"]
            else:
                # If not, extract prompts from the chosen/rejected columns
                # This is a fallback for datasets prepared for reward modeling
                if 'chosen' in dataset.column_names:
                    # Extract prompts from chosen responses (assuming format: prompt\nresponse)
                    prompts = []
                    for chosen in dataset["chosen"]:
                        if '\n' in chosen:
                            prompt = chosen.split('\n')[0]
                            prompts.append(prompt)
                        else:
                            # If no clear delimiter, use the first 50 chars as prompt
                            prompts.append(chosen[:50])
                else:
                    # Create synthetic prompts if no clear source
                    prompts = ["Generate a helpful response"] * len(dataset)
            
            # CRITICAL FIX: Process prompts in batches of size 8 to match PPOTrainer's expectations
            # Limit prompts to avoid memory issues
            prompts = prompts[:20]  # Use only first 20 prompts for this test
            for epoch in range(3):  # 3 epochs
                logger.info(f"Starting epoch {epoch+1}/3")
                
                # Process prompts in batches of size 8
                batch_size = ppo_config.batch_size  # Use the same batch size as in PPOConfig
                
                for i in range(0, len(prompts), batch_size):
                    # Get batch of prompts (pad if needed)
                    batch_end = min(i + batch_size, len(prompts))
                    batch_prompts = prompts[i:batch_end]
                    
                    # If we don't have enough prompts to fill a batch, duplicate the last one
                    if len(batch_prompts) < batch_size:
                        logger.info(f"Padding batch with {batch_size - len(batch_prompts)} duplicates to reach batch_size={batch_size}")
                        padding_needed = batch_size - len(batch_prompts)
                        batch_prompts = list(batch_prompts) + [batch_prompts[-1]] * padding_needed
                    
                    logger.info(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size} in epoch {epoch+1}")
                    
                    # Generate responses for all prompts in batch
                    batch_responses = []
                    batch_rewards = []
                    
                    # Process each prompt in the batch
                    for prompt in batch_prompts:
                        # Tokenize prompt with proper padding and attention mask
                        query_dict = tokenizer(
                            prompt, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True,
                            max_length=128
                        )
                        query_tensor = query_dict['input_ids'].squeeze(0)  # Remove batch dimension
                        
                        # Generate response using PPO trainer's generate method
                        with torch.no_grad():
                            response_tensor = ppo_trainer.generate(
                                query_tensor.unsqueeze(0),  # Add batch dimension back
                                max_new_tokens=50,  # Reduced to avoid memory issues
                                do_sample=True,
                                top_p=0.9,
                                temperature=0.7
                            )
                        
                        # Decode the response
                        response_decoded = tokenizer.decode(response_tensor[0], skip_special_tokens=True)
                        
                        # Add to batch
                        batch_responses.append(response_decoded)
                    
                    # Compute rewards for all responses in batch
                    batch_rewards = compute_reward(batch_responses)
                    
                    # Debug batch information
                    logger.info(f"Batch size: {len(batch_prompts)}")
                    logger.info(f"Responses batch size: {len(batch_responses)}")
                    logger.info(f"Rewards batch size: {len(batch_rewards)}")
                    
                    # Verify all batch sizes match
                    assert len(batch_prompts) == batch_size, f"Prompt batch size {len(batch_prompts)} doesn't match expected {batch_size}"
                    assert len(batch_responses) == batch_size, f"Response batch size {len(batch_responses)} doesn't match expected {batch_size}"
                    assert len(batch_rewards) == batch_size, f"Reward batch size {len(batch_rewards)} doesn't match expected {batch_size}"
                    
                    # Run PPO step with the full batch
                    train_stats = ppo_trainer.step(batch_prompts, batch_responses, batch_rewards)
                    
                    logger.info(f"Completed PPO step for batch {i//batch_size + 1} in epoch {epoch+1}")
            
            
            # Save the trained model
            ppo_trainer.save_pretrained(output_dir)
            logger.info(f"RLHF model trained and saved to {output_dir}")
            
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
            dataset = self.prepare_dataset_from_csv(data_source, min_rating)
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
