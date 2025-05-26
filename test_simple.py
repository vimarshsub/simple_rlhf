from rlhf_trainer import RLHFTrainer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Initialize trainer
    trainer = RLHFTrainer()
    
    # Test just the dataset preparation first
    logger.info("Testing dataset preparation...")
    dataset = trainer.prepare_dataset_from_csv("data/simple_training.csv")
    
    if dataset is None:
        logger.error("Failed to prepare dataset")
        return
    
    logger.info(f"Dataset prepared successfully with {len(dataset)} examples")
    logger.info(f"Dataset columns: {dataset.column_names}")
    logger.info(f"First example: {dataset[0]}")
    
    # Test reward model training
    logger.info("Testing reward model training...")
    success = trainer.train_reward_model(dataset)
    
    if success:
        logger.info("Reward model training completed successfully!")
    else:
        logger.error("Reward model training failed!")

if __name__ == "__main__":
    main() 