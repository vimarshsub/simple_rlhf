import os
import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments
from peft import PeftModel
import json
import shutil
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("reinforce_trainer")

class ReinforceTrainer:
    """
    Implementation of REINFORCE++ algorithm for RLHF.
    
    REINFORCE++ is a simpler and more stable algorithm than PPO, with the following key differences:
    1. Uses a simpler policy gradient update without clipping
    2. Incorporates KL penalty to prevent divergence from reference model
    3. Uses a simpler advantage calculation
    4. Doesn't require a separate value function (critic)
    """
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        reward_model_path: str = "models/reward_model",
        rlhf_model_path: str = "models/rlhf_model",
        learning_rate: float = 1e-5,
        kl_coef: float = 0.1,
        batch_size: int = 8,
        mini_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        max_length: int = 512,
        max_prompt_length: int = 128,
        max_new_tokens: int = 384,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
        seed: int = 42
    ):
        """
        Initialize the REINFORCE++ trainer.
        
        Args:
            model_name: Base model name or path
            reward_model_path: Path to the reward model
            rlhf_model_path: Path to save the RLHF model
            learning_rate: Learning rate for optimizer
            kl_coef: Coefficient for KL penalty
            batch_size: Batch size for training
            mini_batch_size: Mini batch size for training
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_length: Maximum sequence length
            max_prompt_length: Maximum prompt length
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for generation
            top_p: Top-p for nucleus sampling
            top_k: Top-k for sampling
            repetition_penalty: Penalty for repetition
            no_repeat_ngram_size: Size of n-grams to avoid repeating
            seed: Random seed
        """
        self.model_name = model_name
        self.reward_model_path = reward_model_path
        self.rlhf_model_path = rlhf_model_path
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.kl_coef = kl_coef
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Generation parameters
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        
        # Other parameters
        self.seed = seed
        
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Flags to track training state
        self._reward_training = False
        self._rlhf_training = False
    
    def setup_tokenizer(self):
        """Set up the tokenizer for the model."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Ensure the tokenizer has padding token, using eos_token as fallback
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            logger.error(f"Error setting up tokenizer: {str(e)}")
            raise
    
    def train_reward_model(self, dataset, output_dir=None):
        """
        Train a reward model using preference data.
        
        Args:
            dataset: HuggingFace dataset with chosen/rejected pairs
            output_dir (str, optional): Directory to save the model
            
        Returns:
            Success status
        """
        if output_dir is None:
            output_dir = self.reward_model_path
        
        logger.info("Setting up reward model training...")
        
        # Set flag to indicate reward model training is in progress
        self._reward_training = True
        
        try:
            # Set up tokenizer
            tokenizer = self.setup_tokenizer()
            
            # Load base model for reward modeling
            logger.info(f"Loading base model {self.model_name} for reward modeling...")
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=1,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Verify the model is correctly loaded
            logger.info(f"Model loaded successfully: {type(model)}")
            
            # Prepare dataset for reward modeling
            logger.info("Preparing dataset for reward modeling...")
            
            # Check if dataset has the required columns
            required_columns = ["chosen", "rejected"]
            if not all(col in dataset.column_names for col in required_columns):
                logger.error(f"Dataset missing required columns: {required_columns}")
                raise ValueError(f"Dataset must contain columns: {required_columns}")
            
            # Tokenize the dataset
            def tokenize_function(examples):
                chosen_inputs = tokenizer(examples["chosen"], truncation=True, max_length=self.max_length)
                rejected_inputs = tokenizer(examples["rejected"], truncation=True, max_length=self.max_length)
                
                # Create model inputs
                model_inputs = {
                    "input_ids_chosen": chosen_inputs["input_ids"],
                    "attention_mask_chosen": chosen_inputs["attention_mask"],
                    "input_ids_rejected": rejected_inputs["input_ids"],
                    "attention_mask_rejected": rejected_inputs["attention_mask"],
                }
                return model_inputs
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Configure training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=3,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2,
                learning_rate=self.learning_rate,
                weight_decay=0.01,
                fp16=True,
                logging_steps=10,
                save_strategy="epoch",
                evaluation_strategy="no",
                save_total_limit=1,
                remove_unused_columns=False,
            )
            
            # Initialize reward trainer - dataset should already have the required columns
            # CRITICAL FIX: Explicitly set max_length for RewardDataCollatorWithPadding
            trainer = RewardTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=tokenizer,  # Use tokenizer directly in trl 0.7.0
                max_length=512,  # Explicitly set max_length to match tokenization
            )
            
            # Train the model
            logger.info("Starting reward model training...")
            trainer.train()
            
            # Save the trained model
            logger.info(f"Saving reward model to {output_dir}")
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Verify config.json exists
            output_config_path = os.path.join(output_dir, "config.json")
            if not os.path.exists(output_config_path):
                logger.warning(f"config.json not found at {output_config_path}, creating it manually")
                
                # Create a minimal config.json
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
                
                # Verify the file was created
                if os.path.exists(output_config_path):
                    with open(output_config_path, 'r') as f:
                        config_content = f.read()
                    logger.info(f"Verified config.json was created with size: {len(config_content)} bytes")
                else:
                    logger.error(f"Failed to create config.json at {output_config_path}")
            
            # Final verification
            logger.info(f"Final verification of files in {output_dir}:")
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                file_size = os.path.getsize(file_path)
                logger.info(f"  - {file} ({file_size} bytes)")
            
            # Reset flag
            self._reward_training = False
            
            return True
            
        except Exception as e:
            logger.error(f"Error training reward model: {str(e)}")
            # Reset flag on error
            self._reward_training = False
            return False
    
    def train_with_reinforce(self, dataset, reward_model_path=None, output_dir=None):
        """
        Train a model with REINFORCE++ algorithm.
        
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
        
        logger.info("Setting up REINFORCE++ training...")
        
        try:
            # Set up tokenizer
            tokenizer = self.setup_tokenizer()
            
            # Load base model
            logger.info("Loading base model for REINFORCE++ training...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Load reference model (same as base model but with frozen weights)
            logger.info("Loading reference model...")
            reference_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Freeze reference model weights
            for param in reference_model.parameters():
                param.requires_grad = False
            
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
                    device_map="auto"
                )
            
            logger.info(f"Successfully loaded reward model")
            
            # Create reward function
            def compute_reward(samples):
                inputs = tokenizer(samples, return_tensors="pt", padding=True).to(reward_model.device)
                with torch.no_grad():
                    outputs = reward_model(**inputs)
                rewards = outputs.logits[:, -1].cpu().tolist()
                return rewards
            
            # Set up optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            
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
            
            # Limit prompts to avoid memory issues
            prompts = prompts[:24]  # Use 24 prompts (divisible by batch_size=8)
            
            # Set up model in evaluation mode for generation
            model.eval()
            reference_model.eval()
            
            # Add extensive logging for debugging
            logger.info(f"Training configuration: batch_size={self.batch_size}, mini_batch_size={self.mini_batch_size}, gradient_accumulation_steps={self.gradient_accumulation_steps}")
            
            # REINFORCE++ training loop
            for epoch in range(3):  # 3 epochs
                logger.info(f"Starting epoch {epoch+1}/3")
                
                # Process prompts in batches
                for batch_idx in range(0, len(prompts), self.batch_size):
                    batch_prompts = prompts[batch_idx:batch_idx + self.batch_size]
                    logger.info(f"Processing batch {batch_idx//self.batch_size + 1}/{len(prompts)//self.batch_size + (1 if len(prompts) % self.batch_size > 0 else 0)} in epoch {epoch+1}")
                    
                    # Initialize batch collection lists
                    batch_log_probs = []
                    batch_ref_log_probs = []
                    batch_rewards = []
                    batch_responses = []
                    
                    # Process each prompt in the batch
                    for prompt in batch_prompts:
                        try:
                            # 1. Tokenize prompt
                            prompt_tokens = tokenizer(
                                prompt, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True,
                                max_length=self.max_prompt_length
                            ).to(model.device)
                            
                            # 2. Generate response using model
                            with torch.no_grad():
                                # Use greedy decoding to avoid CUDA errors
                                logger.info("Using greedy decoding to avoid CUDA probability tensor errors")
                                generation_kwargs = {
                                    "max_new_tokens": self.max_new_tokens,
                                    "do_sample": False,  # Use greedy decoding
                                    "pad_token_id": tokenizer.eos_token_id,
                                    "eos_token_id": tokenizer.eos_token_id,
                                    "use_cache": True,
                                    # Block common special characters that tend to repeat
                                    "bad_words_ids": [
                                        [33],  # ! exclamation mark
                                        [40],  # ( open parenthesis
                                        [41],  # ) close parenthesis
                                        [91],  # [ open bracket
                                        [93],  # ] close bracket
                                        [123], # { open brace
                                        [125], # } close brace
                                    ]
                                }
                                
                                try:
                                    outputs = model.generate(
                                        input_ids=prompt_tokens.input_ids,
                                        attention_mask=prompt_tokens.attention_mask,
                                        **generation_kwargs
                                    )
                                    
                                    # Get the full generated text
                                    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                                    
                                    # Extract only the generated part (remove the prompt)
                                    prompt_text = tokenizer.decode(prompt_tokens.input_ids[0], skip_special_tokens=True)
                                    if response_text.startswith(prompt_text):
                                        response_text = response_text[len(prompt_text):].strip()
                                    
                                    logger.info(f"Generated response: '{response_text[:50]}...'")
                                    
                                except Exception as gen_error:
                                    logger.warning(f"Generation failed with error: {str(gen_error)}. Using fallback response.")
                                    response_text = "I apologize, but I cannot provide a response at this time."
                            
                            # 3. Compute reward for the generated response
                            reward = compute_reward([prompt + " " + response_text])[0]
                            logger.info(f"Computed reward: {reward}")
                            
                            # 4. Compute log probabilities for policy gradient
                            # Tokenize the full sequence (prompt + response)
                            full_text = prompt + " " + response_text
                            full_tokens = tokenizer(
                                full_text,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=self.max_length
                            ).to(model.device)
                            
                            # Get the length of the prompt tokens to separate prompt from response
                            prompt_length = prompt_tokens.input_ids.shape[1]
                            
                            # Forward pass through model to get logits
                            with torch.no_grad():
                                outputs = model(
                                    input_ids=full_tokens.input_ids,
                                    attention_mask=full_tokens.attention_mask
                                )
                                logits = outputs.logits
                                
                                # Forward pass through reference model
                                ref_outputs = reference_model(
                                    input_ids=full_tokens.input_ids,
                                    attention_mask=full_tokens.attention_mask
                                )
                                ref_logits = ref_outputs.logits
                            
                            # Compute log probabilities for the response tokens only
                            # Shift logits and input_ids for calculating log probs
                            shift_logits = logits[:, :-1, :].contiguous()
                            shift_ref_logits = ref_logits[:, :-1, :].contiguous()
                            shift_ids = full_tokens.input_ids[:, 1:].contiguous()
                            
                            # Only consider response tokens (skip prompt tokens)
                            response_shift_logits = shift_logits[:, prompt_length-1:]
                            response_shift_ref_logits = shift_ref_logits[:, prompt_length-1:]
                            response_shift_ids = shift_ids[:, prompt_length-1:]
                            
                            # Compute log probabilities
                            log_probs = F.log_softmax(response_shift_logits, dim=-1)
                            ref_log_probs = F.log_softmax(response_shift_ref_logits, dim=-1)
                            
                            # Gather the log probs for the actual tokens
                            token_log_probs = log_probs.gather(-1, response_shift_ids.unsqueeze(-1)).squeeze(-1)
                            token_ref_log_probs = ref_log_probs.gather(-1, response_shift_ids.unsqueeze(-1)).squeeze(-1)
                            
                            # Sum log probs over sequence
                            sequence_log_prob = token_log_probs.sum()
                            sequence_ref_log_prob = token_ref_log_probs.sum()
                            
                            # Add to batch collections
                            batch_log_probs.append(sequence_log_prob)
                            batch_ref_log_probs.append(sequence_ref_log_prob)
                            batch_rewards.append(reward)
                            batch_responses.append(response_text)
                            
                        except Exception as e:
                            logger.error(f"Error processing prompt: {str(e)}")
                            continue
                    
                    # Skip batch if empty
                    if len(batch_log_probs) == 0:
                        logger.warning("Skipping empty batch")
                        continue
                    
                    # Convert lists to tensors
                    batch_log_probs = torch.stack(batch_log_probs)
                    batch_ref_log_probs = torch.stack(batch_ref_log_probs)
                    batch_rewards = torch.tensor(batch_rewards, device=model.device)
                    
                    # Normalize rewards for stability
                    batch_rewards = (batch_rewards - batch_rewards.mean()) / (batch_rewards.std() + 1e-8)
                    
                    # Compute KL divergence penalty
                    kl_div = batch_log_probs - batch_ref_log_probs
                    
                    # Compute REINFORCE++ loss
                    # Loss = -log_prob * (reward - kl_coef * kl_div)
                    advantages = batch_rewards - self.kl_coef * kl_div.detach()
                    policy_loss = -(batch_log_probs * advantages).mean()
                    
                    # Backward pass and optimization
                    model.train()
                    optimizer.zero_grad()
                    policy_loss.backward()
                    optimizer.step()
                    model.eval()
                    
                    # Log training progress
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx//self.batch_size + 1}: Loss = {policy_loss.item():.4f}, Mean Reward = {batch_rewards.mean().item():.4f}, Mean KL = {kl_div.mean().item():.4f}")
                    
                    # Log a sample response
                    if len(batch_responses) > 0:
                        sample_idx = np.random.randint(0, len(batch_responses))
                        logger.info(f"Sample response: '{batch_responses[sample_idx][:100]}...'")
            
            # Save the trained model
            logger.info(f"Saving REINFORCE++ model to {output_dir}")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Save the model
            model.save_pretrained(output_dir)
            
            # 2. Save the tokenizer
            tokenizer.save_pretrained(output_dir)
            
            # 3. Verify config.json exists
            output_config_path = os.path.join(output_dir, "config.json")
            if os.path.exists(output_config_path):
                logger.info(f"Verified config.json exists at {output_config_path}")
            else:
                logger.warning(f"config.json not found at {output_config_path}, creating it manually")
                # Create minimal config if it doesn't exist
                config = model.config.to_dict()
                with open(output_config_path, 'w') as f:
                    json.dump(config, f)
                logger.info(f"Created config.json at {output_config_path}")
            
            # Final verification
            logger.info(f"Final verification of files in {output_dir}:")
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                file_size = os.path.getsize(file_path)
                logger.info(f"  - {file} ({file_size} bytes)")
            
            logger.info(f"REINFORCE++ model trained and saved to {output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during REINFORCE++ training: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def generate_text(self, prompt, model_path=None, max_new_tokens=None):
        """
        Generate text using the trained model.
        
        Args:
            prompt: Input prompt
            model_path (str, optional): Path to the model
            max_new_tokens (int, optional): Maximum number of new tokens to generate
            
        Returns:
            Generated text
        """
        if model_path is None:
            model_path = self.rlhf_model_path
        
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        logger.info(f"Generating text using model: {model_path}")
        
        try:
            # Set up tokenizer
            tokenizer = self.setup_tokenizer()
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate text
            with torch.no_grad():
                try:
                    # Use beam search with repetition penalties to avoid repetitive outputs
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=max_new_tokens,
                        num_beams=5,
                        do_sample=False,
                        repetition_penalty=1.5,
                        no_repeat_ngram_size=3,
                        length_penalty=1.0,
                        early_stopping=True,
                        bad_words_ids=[
                            [33],  # ! exclamation mark
                            [40],  # ( open parenthesis
                            [41],  # ) close parenthesis
                            [91],  # [ open bracket
                            [93],  # ] close bracket
                            [123], # { open brace
                            [125], # } close brace
                        ]
                    )
                except Exception as e:
                    logger.error(f"Error in beam search generation: {str(e)}")
                    # Fallback to greedy decoding
                    logger.info("Falling back to greedy decoding")
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        repetition_penalty=2.0,
                        no_repeat_ngram_size=4
                    )
            
            # Decode the generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove the prompt)
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return f"Error generating text: {str(e)}"

# Import required for reward model training
try:
    from trl import RewardTrainer
except ImportError:
    logger.warning("trl package not found. Reward model training will not be available.")
    
    # Define a dummy RewardTrainer class for type checking
    class RewardTrainer:
        def __init__(self, *args, **kwargs):
            pass
