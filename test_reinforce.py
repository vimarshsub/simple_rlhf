import argparse
import logging
import os
import torch
from reinforce_trainer import ReinforceTrainer
from vllm_integration import VLLMGenerator

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("test_reinforce")

def parse_args():
    parser = argparse.ArgumentParser(description="Test REINFORCE++ implementation")
    
    # Test mode
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["test_generation", "test_vllm", "compare_performance"],
                        help="Test mode: test_generation, test_vllm, or compare_performance")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                        help="Base model name or path (default: distilgpt2)")
    parser.add_argument("--rlhf_model_path", type=str, default="models/rlhf_model",
                        help="Path to the RLHF model (default: models/rlhf_model)")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=384,
                        help="Maximum number of new tokens to generate (default: 384)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p for nucleus sampling (default: 0.9)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k for sampling (default: 50)")
    
    # Test prompts
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="Path to file containing test prompts (one per line)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single test prompt")
    
    # vLLM parameters
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs to use for tensor parallelism (default: 1)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                        help="Fraction of GPU memory to use (default: 0.8)")
    
    return parser.parse_args()

def load_test_prompts(args):
    """Load test prompts from file or use the provided prompt."""
    prompts = []
    
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    
    if args.prompt:
        if not prompts:
            prompts = [args.prompt]
        else:
            prompts.append(args.prompt)
        logger.info(f"Added prompt from command line: {args.prompt}")
    
    if not prompts:
        # Default test prompts
        prompts = [
            "How do I check the IP address of my Linux machine?",
            "My Wi-Fi is connected but I have no internet access. What should I do?",
            "What is the difference between TCP and UDP?",
            "How can I test the network latency to google.com?"
        ]
        logger.info(f"Using {len(prompts)} default test prompts")
    
    return prompts

def test_standard_generation(args, prompts):
    """Test text generation using the standard ReinforceTrainer."""
    logger.info("Testing standard generation with ReinforceTrainer")
    
    # Initialize trainer
    trainer = ReinforceTrainer(
        model_name=args.model_name,
        rlhf_model_path=args.rlhf_model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    
    # Generate text for each prompt
    results = []
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    for prompt in prompts:
        logger.info(f"Generating text for prompt: {prompt}")
        try:
            generated_text = trainer.generate_text(
                prompt=prompt,
                model_path=args.rlhf_model_path,
                max_new_tokens=args.max_new_tokens
            )
            results.append((prompt, generated_text))
            logger.info(f"Generated text: {generated_text[:100]}...")
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            results.append((prompt, f"Error: {str(e)}"))
    
    end_time.record()
    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
    
    logger.info(f"Standard generation completed in {elapsed_time:.2f} seconds")
    
    return results, elapsed_time

def test_vllm_generation(args, prompts):
    """Test text generation using vLLM."""
    logger.info("Testing vLLM generation")
    
    # Initialize vLLM generator
    generator = VLLMGenerator(
        model_name=args.model_name if not os.path.exists(args.rlhf_model_path) else args.rlhf_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # Generate text for all prompts at once
    logger.info(f"Generating text for {len(prompts)} prompts using vLLM")
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    try:
        generated_texts = generator.generate(
            prompts=prompts,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=1.2
        )
        results = list(zip(prompts, generated_texts))
        
        for prompt, text in results:
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Generated text: {text[:100]}...")
    except Exception as e:
        logger.error(f"Error generating text with vLLM: {str(e)}")
        results = [(prompt, f"Error: {str(e)}") for prompt in prompts]
    
    end_time.record()
    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
    
    logger.info(f"vLLM generation completed in {elapsed_time:.2f} seconds")
    
    # Shutdown vLLM
    generator.shutdown()
    
    return results, elapsed_time

def compare_performance(args, prompts):
    """Compare performance between standard generation and vLLM."""
    logger.info("Comparing performance between standard generation and vLLM")
    
    # Test standard generation
    standard_results, standard_time = test_standard_generation(args, prompts)
    
    # Test vLLM generation
    vllm_results, vllm_time = test_vllm_generation(args, prompts)
    
    # Calculate speedup
    speedup = standard_time / vllm_time if vllm_time > 0 else float('inf')
    
    logger.info(f"Performance comparison:")
    logger.info(f"  Standard generation: {standard_time:.2f} seconds")
    logger.info(f"  vLLM generation: {vllm_time:.2f} seconds")
    logger.info(f"  Speedup: {speedup:.2f}x")
    
    # Compare output quality
    logger.info("Output comparison:")
    for i, ((prompt1, text1), (prompt2, text2)) in enumerate(zip(standard_results, vllm_results)):
        logger.info(f"Prompt {i+1}: {prompt1}")
        logger.info(f"  Standard: {text1[:100]}...")
        logger.info(f"  vLLM: {text2[:100]}...")
    
    return {
        "standard_time": standard_time,
        "vllm_time": vllm_time,
        "speedup": speedup,
        "standard_results": standard_results,
        "vllm_results": vllm_results
    }

def main():
    args = parse_args()
    
    # Load test prompts
    prompts = load_test_prompts(args)
    
    if args.mode == "test_generation":
        # Test standard generation
        results, elapsed_time = test_standard_generation(args, prompts)
        
        # Print results
        print("\n--- Generation Results ---")
        for prompt, text in results:
            print(f"Prompt: {prompt}")
            print(f"Generated: {text}")
            print("-" * 50)
        
        print(f"Generation completed in {elapsed_time:.2f} seconds")
    
    elif args.mode == "test_vllm":
        # Test vLLM generation
        results, elapsed_time = test_vllm_generation(args, prompts)
        
        # Print results
        print("\n--- vLLM Generation Results ---")
        for prompt, text in results:
            print(f"Prompt: {prompt}")
            print(f"Generated: {text}")
            print("-" * 50)
        
        print(f"vLLM generation completed in {elapsed_time:.2f} seconds")
    
    elif args.mode == "compare_performance":
        # Compare performance
        comparison = compare_performance(args, prompts)
        
        # Print comparison
        print("\n--- Performance Comparison ---")
        print(f"Standard generation: {comparison['standard_time']:.2f} seconds")
        print(f"vLLM generation: {comparison['vllm_time']:.2f} seconds")
        print(f"Speedup: {comparison['speedup']:.2f}x")
        
        print("\n--- Output Comparison ---")
        for i, ((prompt1, text1), (prompt2, text2)) in enumerate(zip(comparison['standard_results'], comparison['vllm_results'])):
            print(f"Prompt {i+1}: {prompt1}")
            print(f"  Standard: {text1[:100]}...")
            print(f"  vLLM: {text2[:100]}...")
            print("-" * 50)

if __name__ == "__main__":
    main()
