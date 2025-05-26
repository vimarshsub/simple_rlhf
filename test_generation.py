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
    
    # Test basic text generation
    logger.info("Testing basic text generation...")
    
    prompts = [
        "What is Python?",
        "How to print in Python?",
        "What is a variable?"
    ]
    
    for prompt in prompts:
        logger.info(f"Generating response for: {prompt}")
        response = trainer.generate_text(prompt, max_new_tokens=50, temperature=0.7)
        logger.info(f"Response: {response}")
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}\n")

if __name__ == "__main__":
    main() 