import os
import logging
import torch
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("vllm_integration")

class VLLMGenerator:
    """
    Integration with vLLM for efficient text generation.
    
    vLLM is a high-throughput, memory-efficient library for LLM inference,
    which significantly speeds up the generation process during RLHF training.
    """
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 2048,
        dtype: str = "half",
        seed: int = 42
    ):
        """
        Initialize the vLLM generator.
        
        Args:
            model_name: Base model name or path
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
            dtype: Data type for model weights (half, float, bfloat16)
            seed: Random seed
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.seed = seed
        
        # Flag to track initialization
        self._initialized = False
        self.llm = None
    
    def initialize(self):
        """Initialize the vLLM engine."""
        if self._initialized:
            logger.info("vLLM engine already initialized")
            return
        
        try:
            logger.info(f"Initializing vLLM engine with model: {self.model_name}")
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                seed=self.seed
            )
            self._initialized = True
            logger.info("vLLM engine initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vLLM engine: {str(e)}")
            raise
    
    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 384,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stop_tokens: Optional[List[str]] = None,
        bad_words: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate text for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            top_k: Top-k for sampling
            repetition_penalty: Penalty for repetition
            presence_penalty: Penalty for token presence
            frequency_penalty: Penalty for token frequency
            stop_tokens: List of tokens to stop generation
            bad_words: List of words to avoid generating
            
        Returns:
            List of generated texts
        """
        if not self._initialized:
            self.initialize()
        
        try:
            # Configure sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stop=stop_tokens,
                bad_words=bad_words
            )
            
            # Generate text
            logger.info(f"Generating text for {len(prompts)} prompts")
            outputs = self.llm.generate(prompts, sampling_params)
            
            # Extract generated text
            generated_texts = []
            for output in outputs:
                generated_text = output.outputs[0].text
                generated_texts.append(generated_text)
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"Error generating text with vLLM: {str(e)}")
            # Fallback to empty strings
            return [""] * len(prompts)
    
    def shutdown(self):
        """Shutdown the vLLM engine."""
        if self._initialized and self.llm is not None:
            logger.info("Shutting down vLLM engine")
            # vLLM doesn't have an explicit shutdown method, but we can release the reference
            self.llm = None
            self._initialized = False
            # Force CUDA cache clearing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("vLLM engine shut down successfully")

# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = VLLMGenerator(model_name="distilgpt2")
    
    # Generate text
    prompts = [
        "What is the capital of France?",
        "How does a computer network function?"
    ]
    
    generated_texts = generator.generate(prompts)
    
    # Print results
    for prompt, text in zip(prompts, generated_texts):
        print(f"Prompt: {prompt}")
        print(f"Generated: {text}")
        print("-" * 50)
    
    # Shutdown
    generator.shutdown()
