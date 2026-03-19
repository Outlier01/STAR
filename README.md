# STAR: Strategy-driven Automatic Jailbreak Red-teaming For Large Language Model

## Requirements

- Python == 3.12.0
- torch == 2.6.0
- vllm == 0.8.5
- transformers == 4.51.3

## Usage

### Trainer.py

`Trainer.py` implements the main GRPO-based reinforcement learning trainer that uses vLLM for efficient sampling to optimize the attacker model for generating jailbreak prompts.

### steer/SVTrainer.py

`steer/SVTrainer.py` implements the steering vector trainer that learns activation-level perturbations to steer the model's hidden states toward strategy-specific behaviors.
