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
    
    # Run training pipeline with sample data
    logger.info("Starting RLHF training pipeline...")
    success = trainer.run_full_training_pipeline("data/sample_training.csv", min_rating=3)
    
    if success:
        logger.info("Training completed successfully!")
    else:
        logger.error("Training failed!")

if __name__ == "__main__":
    main() 