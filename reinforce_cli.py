import argparse
import logging
import os
import pandas as pd
from datasets import Dataset
from reinforce_trainer import ReinforceTrainer

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("reinforce_cli")

def parse_args():
    parser = argparse.ArgumentParser(description="Train and use REINFORCE++ models for RLHF")
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True, choices=["train_reward", "train_rlhf", "generate"],
                        help="Operation mode: train_reward, train_rlhf, or generate")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                        help="Base model name or path (default: distilgpt2)")
    parser.add_argument("--reward_model_path", type=str, default="models/reward_model",
                        help="Path to the reward model (default: models/reward_model)")
    parser.add_argument("--rlhf_model_path", type=str, default="models/rlhf_model",
                        help="Path to save/load the RLHF model (default: models/rlhf_model)")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for optimizer (default: 1e-5)")
    parser.add_argument("--kl_coef", type=float, default=0.1,
                        help="Coefficient for KL penalty (default: 0.1)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training (default: 8)")
    parser.add_argument("--mini_batch_size", type=int, default=1,
                        help="Mini batch size for training (default: 1)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of steps to accumulate gradients (default: 8)")
    
    # Generation parameters
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length (default: 512)")
    parser.add_argument("--max_prompt_length", type=int, default=128,
                        help="Maximum prompt length (default: 128)")
    parser.add_argument("--max_new_tokens", type=int, default=384,
                        help="Maximum number of new tokens to generate (default: 384)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p for nucleus sampling (default: 0.9)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k for sampling (default: 50)")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the data file (CSV format)")
    parser.add_argument("--prompt_column", type=str, default="prompt",
                        help="Column name for prompts (default: prompt)")
    parser.add_argument("--chosen_column", type=str, default="chosen",
                        help="Column name for chosen responses (default: chosen)")
    parser.add_argument("--rejected_column", type=str, default="rejected",
                        help="Column name for rejected responses (default: rejected)")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt for text generation (required for generate mode)")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    return parser.parse_args()

def load_dataset(args):
    """Load dataset from CSV file."""
    logger.info(f"Loading dataset from {args.data_path}")
    
    try:
        # Read CSV file
        df = pd.read_csv(args.data_path)
        
        # Check if required columns exist
        if args.mode == "train_reward":
            required_columns = [args.chosen_column, args.rejected_column]
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Dataset missing required columns for reward training: {required_columns}")
                return None
            
            # Rename columns to match expected format
            df = df.rename(columns={
                args.chosen_column: "chosen",
                args.rejected_column: "rejected"
            })
            
        elif args.mode == "train_rlhf":
            if args.prompt_column not in df.columns:
                logger.error(f"Dataset missing required column for RLHF training: {args.prompt_column}")
                return None
            
            # Rename columns to match expected format
            df = df.rename(columns={
                args.prompt_column: "prompt"
            })
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_pandas(df)
        
        # Ensure dataset size is divisible by batch_size
        if len(dataset) % args.batch_size != 0:
            target_size = (len(dataset) // args.batch_size) * args.batch_size
            logger.warning(f"Dataset size ({len(dataset)}) is not divisible by batch_size ({args.batch_size})")
            logger.warning(f"Truncating dataset to {target_size} examples")
            dataset = dataset.select(range(target_size))
        
        logger.info(f"Loaded dataset with {len(dataset)} examples")
        return dataset
    
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def main():
    args = parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(args.reward_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.rlhf_model_path), exist_ok=True)
    
    # Initialize trainer
    trainer = ReinforceTrainer(
        model_name=args.model_name,
        reward_model_path=args.reward_model_path,
        rlhf_model_path=args.rlhf_model_path,
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed
    )
    
    if args.mode == "train_reward":
        # Load dataset
        dataset = load_dataset(args)
        if dataset is None:
            return
        
        # Train reward model
        logger.info("Starting reward model training")
        success = trainer.train_reward_model(dataset, args.reward_model_path)
        
        if success:
            logger.info(f"Reward model training completed successfully. Model saved to {args.reward_model_path}")
        else:
            logger.error("Reward model training failed")
    
    elif args.mode == "train_rlhf":
        # Load dataset
        dataset = load_dataset(args)
        if dataset is None:
            return
        
        # Train RLHF model
        logger.info("Starting REINFORCE++ training")
        success = trainer.train_with_reinforce(dataset, args.reward_model_path, args.rlhf_model_path)
        
        if success:
            logger.info(f"REINFORCE++ training completed successfully. Model saved to {args.rlhf_model_path}")
        else:
            logger.error("REINFORCE++ training failed")
    
    elif args.mode == "generate":
        # Check if prompt is provided
        if args.prompt is None:
            logger.error("Prompt is required for generate mode")
            return
        
        # Generate text
        logger.info(f"Generating text for prompt: {args.prompt}")
        generated_text = trainer.generate_text(args.prompt, args.rlhf_model_path, args.max_new_tokens)
        
        logger.info(f"Generated text: {generated_text}")
        print("\n--- Generated Text ---")
        print(generated_text)
        print("---------------------")

if __name__ == "__main__":
    main()
