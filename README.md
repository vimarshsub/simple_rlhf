# REINFORCE++ RLHF Implementation

This repository contains an implementation of the REINFORCE++ algorithm for Reinforcement Learning from Human Feedback (RLHF), inspired by the OpenRLHF framework. This implementation offers several advantages over the original PPO-based approach:

- More stable training compared to PPO
- Faster training time
- Better output quality
- Efficient text generation using vLLM

## Installation

### Requirements

Install the required packages:

```bash
pip install -r requirements-reinforce.txt
```

This will install all necessary dependencies, including:
- transformers
- trl
- peft
- accelerate
- vllm (for efficient text generation)

## Usage

### Training a Reward Model

First, train a reward model using your preference data:

```bash
python reinforce_cli.py \
    --mode train_reward \
    --model_name distilgpt2 \
    --data_path example_feedback_diverse.csv \
    --reward_model_path models/reward_model
```

The CSV file should contain columns for chosen and rejected responses.

### Training with REINFORCE++

Once you have a reward model, train your model using REINFORCE++:

```bash
python reinforce_cli.py \
    --mode train_rlhf \
    --model_name distilgpt2 \
    --data_path example_feedback_diverse.csv \
    --reward_model_path models/reward_model \
    --rlhf_model_path models/rlhf_model \
    --learning_rate 1e-5 \
    --kl_coef 0.1 \
    --batch_size 8
```

### Generating Text

Generate text using your trained model:

```bash
python reinforce_cli.py \
    --mode generate \
    --model_name distilgpt2 \
    --rlhf_model_path models/rlhf_model \
    --prompt "How do I check the IP address of my Linux machine?"
```

### Testing and Performance Comparison

Compare the performance of standard generation vs. vLLM-accelerated generation:

```bash
python test_reinforce.py \
    --mode compare_performance \
    --model_name distilgpt2 \
    --rlhf_model_path models/rlhf_model \
    --prompt "How do I check the IP address of my Linux machine?"
```

## Key Components

### REINFORCE++ Algorithm

The REINFORCE++ algorithm is implemented in `reinforce_trainer.py`. It offers several advantages over PPO:

1. **Simpler Implementation**: No complex clipping or value function
2. **Faster Training**: Fewer hyperparameters to tune
3. **More Stable**: Less sensitive to batch size and learning rate

### vLLM Integration

The vLLM integration in `vllm_integration.py` provides:

1. **Faster Generation**: Up to 10x speedup compared to standard generation
2. **Memory Efficiency**: Better memory utilization
3. **Batch Processing**: Efficient handling of multiple prompts

### High-Quality Training Data

The repository includes a sample high-quality training dataset (`example_feedback_diverse.csv`) with 24 examples of network troubleshooting Q&A pairs. Key features:

1. **Divisible by Batch Size**: 24 examples (divisible by batch_size=8)
2. **Clear Quality Difference**: Distinct chosen/rejected responses
3. **Domain-Specific**: Focused on network troubleshooting

## Best Practices

For optimal results:

1. **Use a More Capable Base Model**: Consider Mistral-7B or Llama-2-7B for better performance
2. **Ensure Dataset Size is Divisible by Batch Size**: Avoid padding/duplication issues
3. **Use vLLM for Generation**: Significantly improves generation speed
4. **Tune KL Coefficient**: Balance between reward maximization and staying close to the reference model

## Comparison with PPO

| Aspect | REINFORCE++ | PPO |
|--------|-------------|-----|
| Training Speed | Faster | Slower |
| Stability | More stable | Less stable |
| Implementation Complexity | Simpler | More complex |
| Memory Usage | Lower | Higher |
| Output Quality | Better | Variable |

## Troubleshooting

If you encounter issues:

1. **CUDA Errors**: Ensure you have enough GPU memory; reduce batch size if needed
2. **Generation Quality**: Adjust temperature, top_p, and repetition_penalty
3. **Training Stability**: Tune the KL coefficient and learning rate

## References

- [OpenRLHF GitHub Repository](https://github.com/OpenRLHF/OpenRLHF)
- [REINFORCE++ Algorithm Paper](https://arxiv.org/abs/2204.05862)
- [vLLM Documentation](https://github.com/vllm-project/vllm)
