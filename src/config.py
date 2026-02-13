"""
Configuration management for Reason-1 training.
Uses Pydantic for type-safe hyperparameter management.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import torch


@dataclass
class ModelConfig:
    """Configuration for the base language model."""
    
    model_name: str = "meta-llama/Llama-3-8B"  # or "Qwen/Qwen2.5-3B-Instruct"
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"  # "float16" or "bfloat16"
    device_map: str = "auto"
    
    # LoRA Configuration
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Special Tokens
    think_start_token: str = "<think>"
    think_end_token: str = "</think>"
    
    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        return torch.bfloat16 if self.torch_dtype == "bfloat16" else torch.float16


@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing."""
    
    dataset_name: str = "openai/gsm8k"
    dataset_split: str = "main"
    train_split: str = "train"
    test_split: str = "test"
    
    # Reasoning traces dataset (optional)
    reasoning_dataset: Optional[str] = None  # e.g., "openthought/reasoning-traces"
    
    max_length: int = 2048
    max_reasoning_length: int = 4096  # For longer chains of thought
    
    # Data augmentation
    use_reasoning_traces: bool = True
    mix_ratio: float = 0.3  # Ratio of reasoning traces to original data
    
    # Formatting
    prompt_template: str = (
        "Solve the following math problem step by step.\n\n"
        "Problem: {question}\n\n"
        "Solution:"
    )


@dataclass
class GRPOConfig:
    """Configuration for Group Relative Policy Optimization."""
    
    # Core GRPO Parameters
    group_size: int = 4  # G: Number of outputs per prompt
    learning_rate: float = 1e-5
    num_train_epochs: int = 3
    batch_size: int = 4  # Number of prompts per batch
    gradient_accumulation_steps: int = 4
    
    # PPO-style clipping
    clip_range: float = 0.2
    clip_range_vf: float = 0.2  # Not used in GRPO (no value function)
    
    # KL Divergence Control
    kl_coef: float = 0.05
    target_kl: Optional[float] = None  # Adaptive KL if set
    
    # Advantage Normalization
    adv_epsilon: float = 1e-8
    normalize_advantages: bool = True
    whiten_rewards: bool = True
    
    # Optimization
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Sampling
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = True
    
    # Efficiency
    use_vllm: bool = True
    vllm_tensor_parallel_size: int = 1


@dataclass
class RewardConfig:
    """Configuration for the reward function."""
    
    # Reward Components
    format_reward: float = 0.1  # Bonus for using <think> tags
    answer_reward: float = 1.0  # Reward for correct answer
    length_penalty: float = 0.0  # Penalty per token (discourage verbosity)
    
    # Answer Verification
    use_symbolic_math: bool = True  # Use sympy for equivalence checking
    exact_match_fallback: bool = True
    
    # Reasoning Quality (Optional)
    use_repetition_penalty: bool = True
    repetition_threshold: int = 3  # Max allowed token repetitions
    repetition_penalty_value: float = -0.1
    
    # LLM Judge (Optional - expensive)
    use_llm_judge: bool = False
    judge_model: Optional[str] = None


@dataclass
class InferenceConfig:
    """Configuration for inference and search strategies."""
    
    # Best-of-N Sampling
    num_samples: int = 8  # N: Number of candidates to generate
    max_new_tokens: int = 2048
    
    # KV Cache Management (StreamingLLM)
    use_streaming_kv: bool = True
    attention_sink_size: int = 4  # Keep first 4 tokens
    sliding_window_size: int = 1024  # Keep last 1024 tokens
    
    # Verification
    verify_with_reward: bool = True
    min_confidence_threshold: float = 0.5


@dataclass
class TrainingConfig:
    """Master configuration combining all components."""
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Paths
    output_dir: Path = Path("./outputs")
    checkpoint_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    
    # WandB
    use_wandb: bool = True
    wandb_project: str = "reason-1"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    
    # Hardware
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    
    # Multi-GPU
    local_rank: int = -1
    deepspeed_config: Optional[str] = None
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


def get_default_config() -> TrainingConfig:
    """Return default training configuration."""
    return TrainingConfig()


def load_config(config_path: Optional[Path] = None) -> TrainingConfig:
    """Load configuration from file or return default."""
    if config_path and config_path.exists():
        # TODO: Implement YAML/JSON loading with Hydra
        raise NotImplementedError("Config file loading not yet implemented")
    return get_default_config()
