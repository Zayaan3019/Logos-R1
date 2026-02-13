# Reason-1: System 2 Reasoning Model

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Reason-1** is a state-of-the-art System 2 reasoning model that learns to "think before answering" using **Group Relative Policy Optimization (GRPO)**, a novel reinforcement learning algorithm. The model generates explicit reasoning chains in `<think>...</think>` tags and is trained to maximize mathematical problem-solving accuracy on GSM8K.

## 🚀 Key Features

- **GRPO Algorithm**: Efficient RL training without a value function (critic-free PPO variant)
- **Process Reward Model (PRM)**: Deterministic math verification with symbolic equivalence checking
- **LoRA/QLoRA**: Parameter-efficient fine-tuning for memory-constrained environments
- **StreamingLLM**: KV cache management for 4k+ token reasoning chains
- **Best-of-N Inference**: Rejection sampling with reward-based selection
- **Self-Correction**: Model learns to backtrack and verify its reasoning

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Reason-1 Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: "Solve: John has 5 apples..."                       │
│     ↓                                                        │
│  ┌──────────────────────────────────────┐                   │
│  │   Policy πθ (Llama-3-8B + LoRA)     │                   │
│  └──────────────────────────────────────┘                   │
│     ↓                                                        │
│  <think>                                                     │
│    Step 1: John starts with 5 apples                        │
│    Step 2: He buys 3 more, so 5 + 3 = 8                     │
│    Step 3: Final count is 8 apples                          │
│  </think>                                                    │
│  The answer is 8                                            │
│     ↓                                                        │
│  ┌──────────────────────────────────────┐                   │
│  │   Reward Function (Math Verifier)    │                   │
│  │   • Format: +0.1 (has <think> tags)  │                   │
│  │   • Answer: +1.0 (correct)           │                   │
│  │   • Quality: -0.0 (no repetition)    │                   │
│  │   → Total Reward: 1.1                │                   │
│  └──────────────────────────────────────┘                   │
│     ↓                                                        │
│  ┌──────────────────────────────────────┐                   │
│  │       GRPO Trainer                   │                   │
│  │   1. Sample G=4 outputs per prompt   │                   │
│  │   2. Compute group-relative advantages│                   │
│  │   3. Update policy with PPO-clip     │                   │
│  └──────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (for GPU training)
- 24GB+ GPU VRAM (for Llama-3-8B with LoRA; use QLoRA for less)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Reason-1.git
cd Reason-1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install Flash Attention for faster training
pip install flash-attn --no-build-isolation

# (Optional) Install vLLM for high-throughput inference
pip install vllm
```

## 📚 Project Structure

```
Reason-1/
├── src/
│   ├── config.py              # Hyperparameter configurations
│   ├── data/
│   │   ├── loader.py          # GSM8K dataset loader
│   │   └── tokenizer.py       # Custom tokenizer with <think> tags
│   ├── models/
│   │   ├── policy.py          # Actor model with LoRA
│   │   └── reward.py          # Math verifier (THE JUDGE)
│   ├── rl/
│   │   ├── grpo_trainer.py    # GRPO algorithm implementation
│   │   └── buffer.py          # Experience replay buffer
│   ├── inference/
│   │   ├── search.py          # Best-of-N sampling
│   │   └── kv_cache.py        # StreamingLLM cache manager
│   └── utils/
│       ├── logging.py         # WandB integration
│       └── math_utils.py      # Answer extraction utilities
├── scripts/
│   ├── train_sft.py           # Stage 1: Supervised Fine-Tuning
│   └── train_rl.py            # Stage 2: GRPO RL Training
├── tests/                     # Unit tests
├── requirements.txt
└── README.md
```

## 🎓 Training Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)

First, warm-start the model with reasoning traces:

```bash
python scripts/train_sft.py \
  --model_name meta-llama/Llama-3-8B \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-5 \
  --output_dir ./outputs/sft \
  --use_amp
```

**What this does:**
- Fine-tunes the base model on GSM8K with `<think>` formatting
- Teaches the model to generate step-by-step reasoning
- Saves checkpoints to `./outputs/sft/`

### Stage 2: GRPO Reinforcement Learning

Train with RL to maximize reward:

```bash
python scripts/train_rl.py \
  --sft_model ./outputs/sft/best_model \
  --epochs 5 \
  --batch_size 4 \
  --group_size 4 \
  --learning_rate 1e-5 \
  --kl_coef 0.05 \
  --updates_per_batch 4 \
  --output_dir ./outputs/rl
```

**What this does:**
- Loads the SFT model as initialization
- Samples G=4 outputs per prompt
- Computes group-relative advantages (no critic needed!)
- Updates policy to favor high-reward reasoning chains
- Monitors KL divergence to prevent collapse

## 📊 Monitoring Training

Training metrics are automatically logged to [Weights & Biases](https://wandb.ai):

- **Reward Curves**: Mean/max/min rewards over time
- **KL Divergence**: Policy drift from reference
- **Loss Values**: Policy loss and total loss
- **Sample Outputs**: Generated reasoning traces

View your dashboard at: `https://wandb.ai/<your-entity>/reason-1`

## 🔬 GRPO Algorithm Details

### The Core Innovation

GRPO eliminates the need for a value function by using **group-relative advantages**:

```python
# Traditional PPO
Advantage = Q(s, a) - V(s)  # Requires value function V

# GRPO (Our Approach)
For each prompt q, sample G outputs: {o_1, ..., o_G}
Compute rewards: r_1, ..., r_G
Advantage_i = (r_i - mean(r)) / (std(r) + ε)  # No V needed!
```

### Algorithm Pseudocode

```python
for epoch in range(num_epochs):
    for prompt_batch in dataloader:
        # 1. Sample Group
        outputs = [policy.generate(prompt) for _ in range(G)]
        
        # 2. Compute Rewards
        rewards = [reward_fn(output, ground_truth) for output in outputs]
        
        # 3. Compute Advantages
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # 4. PPO Update
        ratio = policy.prob(output) / old_policy.prob(output)
        clipped_ratio = clip(ratio, 1-ε, 1+ε)
        loss = -min(ratio * advantage, clipped_ratio * advantage)
        
        # 5. Add KL Penalty
        kl = KL(policy || reference_policy)
        total_loss = loss + β * kl
        
        # 6. Backward Pass
        total_loss.backward()
        optimizer.step()
```

## 🧪 Inference

### Best-of-N Sampling

```python
from src.inference.search import create_sampler
from src.models.reward import create_reward_function

# Create sampler
sampler = create_sampler(
    model=policy,
    tokenizer=tokenizer,
    reward_function=reward_fn,
    strategy="best_of_n",
    num_samples=8,
)

# Generate with search
result = sampler.search(
    prompt="Solve: John has 5 apples and buys 3 more. How many does he have?",
    ground_truth="8",
)

print(f"Best Answer: {result.answer}")
print(f"Reasoning:\n{result.reasoning}")
print(f"Score: {result.best_score}")
```

### Beam Search (Alternative)

```python
sampler = create_sampler(
    model=policy,
    tokenizer=tokenizer,
    reward_function=reward_fn,
    strategy="beam_search",
    num_samples=4,
)
```

## 🎯 Reward Function Design

The reward function is **CRITICAL** - if it's buggy, RL will fail.

### Components

1. **Format Reward** (+0.1): Uses `<think>` tags correctly
2. **Answer Reward** (+1.0): Final answer matches ground truth
3. **Length Penalty** (-0.001/token): Prevents verbosity
4. **Repetition Penalty** (-0.1): Penalizes repetitive reasoning

### Verification Strategy

```python
# 1. Symbolic Equivalence (preferred)
are_equivalent("5/10", "1/2")  # True (using sympy)

# 2. Numerical Tolerance
abs(float(pred) - float(gt)) < 1e-6

# 3. String Matching (fallback)
normalize("1,234.00") == normalize("1234")  # True
```

## 📈 Expected Results

After full training (SFT + GRPO), expect:

| Metric | Value |
|--------|-------|
| GSM8K Accuracy | 70-80% |
| Average Reward | 0.9+ |
| Reasoning Length | 150-300 tokens |
| Self-Correction Rate | 15-20% |

## 🐛 Troubleshooting

### Out of Memory (OOM)

```bash
# Option 1: Smaller batch size
--batch_size 2 --group_size 2

# Option 2: Gradient accumulation
--gradient_accumulation_steps 4

# Option 3: Use smaller model
--model_name Qwen/Qwen2.5-3B-Instruct
```

### KL Divergence Exploding

```bash
# Increase KL coefficient
--kl_coef 0.1  # (default: 0.05)
```

### Reward Not Improving

1. Check reward function with unit tests
2. Verify SFT model quality
3. Reduce learning rate
4. Ensure dataset is correct

## 🧪 Testing

Run unit tests for the reward function:

```bash
pytest tests/ -v
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GRPO Algorithm**: Inspired by Group Relative Policy Optimization techniques
- **GSM8K Dataset**: [OpenAI GSM8K](https://github.com/openai/grade-school-math)
- **Base Models**: Meta LLaMA-3, Qwen-2.5
- **Libraries**: HuggingFace Transformers, TRL, PEFT

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{reason1_2026,
  title={Reason-1: A System 2 Reasoning Model with GRPO},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/Reason-1}
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

For questions or collaboration:
- Email: your.email@example.com
- Twitter: [@yourusername](https://twitter.com/yourusername)
- GitHub Issues: [Create an issue](https://github.com/yourusername/Reason-1/issues)

---

**Built with ❤️ for advancing AI reasoning capabilities**
