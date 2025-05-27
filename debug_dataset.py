#!/usr/bin/env python3
"""
Debug script to test dataset creation and identify 'chosen' error issues
"""

import logging
import pandas as pd
from datasets import Dataset
from rlhf_trainer import RLHFTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dataset_creation():
    """Test dataset creation with sample data"""
    
    # Create sample CSV data
    sample_data = {
        'prompt': [
            'What is Python?',
            'What is Python?', 
            'How to print in Python?',
            'How to print in Python?'
        ],
        'output': [
            'Python is a programming language.',
            'Python is a snake.',
            'Use print() function.',
            'I don\'t know.'
        ],
        'rating': [5, 2, 5, 1]
    }
    
    # Save as CSV
    df = pd.DataFrame(sample_data)
    df.to_csv('debug_data.csv', index=False)
    logger.info(f"Created debug CSV with {len(df)} rows")
    
    # Test dataset preparation
    trainer = RLHFTrainer()
    
    try:
        logger.info("Testing dataset preparation...")
        dataset = trainer.prepare_dataset_from_csv('debug_data.csv', min_rating=3)
        
        if dataset:
            logger.info("✅ Dataset creation successful!")
            logger.info(f"Dataset columns: {dataset.column_names}")
            logger.info(f"Dataset length: {len(dataset)}")
            logger.info(f"Sample data: {dataset[0]}")
            
            # Test reward model training
            logger.info("Testing reward model training...")
            success = trainer.train_reward_model(dataset)
            
            if success:
                logger.info("✅ Reward model training successful!")
            else:
                logger.error("❌ Reward model training failed!")
                
        else:
            logger.error("❌ Dataset creation failed!")
            
    except Exception as e:
        logger.error(f"❌ Error during testing: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_dataset_creation() 