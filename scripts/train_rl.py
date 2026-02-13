"""
Stage 2: Reinforcement Learning with GRPO

This script trains the policy using Group Relative Policy Optimization (GRPO)
to maximize rewards from the mathematical reasoning task.

Usage:
    python scripts/train_rl.py --sft_model ./outputs/sft/best_model --epochs 5
"""

import os
import sys
import argparse
import torch
import random
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TrainingConfig, GRPOConfig, get_default_config
from src.data.tokenizer import create_tokenizer
from src.data.loader import create_rl_dataloader
from src.models.policy import create_policy, create_reference_policy
from src.models.reward import create_reward_function
from src.rl.grpo_trainer import GRPOTrainer
from src.utils.logging import setup_logging, WandBLogger

import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sft_model(
    model_path: str,
    config,
    tokenizer_len: int,
    device: str,
):
    """
    Load pre-trained SFT model.
    
    Args:
        model_path: Path to saved SFT model
        config: Model configuration
        tokenizer_len: Tokenizer vocabulary size
        device: Device to load model to
    
    Returns:
        Loaded policy model
    """
    logger.info(f"Loading SFT model from {model_path}")
    
    # Create policy
    policy = create_policy(
        config=config,
        tokenizer_len=tokenizer_len,
    )
    
    # Load weights
    policy.load_pretrained(model_path)
    policy = policy.to(device)
    
    logger.info("SFT model loaded successfully")
    
    return policy


def main(args):
    """Main RL training function."""
    
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    setup_logging(
        log_level="INFO",
        log_file=str(log_dir / "train_rl.log"),
    )
    
    logger.info("=" * 80)
    logger.info("Stage 2: Reinforcement Learning with GRPO")
    logger.info("=" * 80)
    
    # Load configuration
    config = get_default_config()
    config.seed = args.seed
    config.grpo.group_size = args.group_size
    config.grpo.learning_rate = args.learning_rate
    config.grpo.batch_size = args.batch_size
    config.grpo.kl_coef = args.kl_coef
    
    # Set random seed
    set_seed(config.seed)
    logger.info(f"Set random seed: {config.seed}")
    
    # Initialize WandB
    wandb_logger = WandBLogger(
        project=config.wandb_project,
        run_name=args.run_name or f"grpo_rl_training",
        config=vars(args),
        enabled=config.use_wandb,
    )
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create tokenizer
    logger.info("Creating tokenizer...")
    tokenizer = create_tokenizer(
        model_name=config.model.model_name,
        think_start_token=config.model.think_start_token,
        think_end_token=config.model.think_end_token,
    )
    
    # Load SFT model as starting point
    if args.sft_model:
        policy = load_sft_model(
            model_path=args.sft_model,
            config=config.model,
            tokenizer_len=len(tokenizer),
            device=device,
        )
    else:
        logger.warning("No SFT model provided, starting from base model")
        policy = create_policy(
            config=config.model,
            tokenizer_len=len(tokenizer),
        ).to(device)
    
    # Create reference policy (frozen copy)
    logger.info("Creating reference policy...")
    reference_policy = create_reference_policy(policy)
    
    # Create reward function
    logger.info("Creating reward function...")
    reward_function = create_reward_function(
        format_reward=config.reward.format_reward,
        answer_reward=config.reward.answer_reward,
        length_penalty=config.reward.length_penalty,
        repetition_penalty=config.reward.repetition_penalty_value,
        use_symbolic_math=config.reward.use_symbolic_math,
        think_start_token=config.model.think_start_token,
        think_end_token=config.model.think_end_token,
    )
    
    # Create RL dataloader
    logger.info("Creating RL dataloader...")
    rl_dataloader = create_rl_dataloader(
        config=config.data,
        tokenizer=tokenizer,
        batch_size=config.grpo.batch_size,
        num_epochs=1,  # Will loop manually
    )
    
    # Create GRPO trainer
    logger.info("Creating GRPO trainer...")
    trainer = GRPOTrainer(
        policy=policy,
        reference_policy=reference_policy,
        reward_function=reward_function,
        tokenizer=tokenizer,
        config=config.grpo,
        device=device,
    )
    
    # Create learning rate scheduler
    total_steps = len(rl_dataloader) * args.epochs * args.updates_per_batch
    trainer.create_scheduler(num_training_steps=total_steps)
    
    logger.info(f"Total training steps: {total_steps}")
    
    # Watch model with WandB
    wandb_logger.watch_model(policy, log="all", log_freq=100)
    
    # Training loop
    logger.info("=" * 80)
    logger.info("Starting GRPO Training")
    logger.info("=" * 80)
    
    best_reward = -float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*80}")
        
        # Train for one epoch
        epoch_stats = trainer.train_epoch(
            dataloader=rl_dataloader,
            num_updates_per_batch=args.updates_per_batch,
        )
        
        # Get buffer statistics
        buffer_stats = trainer.buffer.get_statistics()
        
        # Log statistics
        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Policy Loss: {epoch_stats['policy_loss']:.4f}")
        logger.info(f"  KL Divergence: {epoch_stats['kl_divergence']:.4f}")
        logger.info(f"  Mean Reward: {buffer_stats['mean_reward']:.4f}")
        logger.info(f"  Max Reward: {buffer_stats['max_reward']:.4f}")
        logger.info(f"  Learning Rate: {epoch_stats['learning_rate']:.6f}")
        
        # Log to WandB
        wandb_logger.log({
            "epoch": epoch,
            "train/policy_loss": epoch_stats["policy_loss"],
            "train/kl_divergence": epoch_stats["kl_divergence"],
            "train/total_loss": epoch_stats["total_loss"],
            "train/mean_advantage": epoch_stats["mean_advantage"],
            "reward/mean": buffer_stats["mean_reward"],
            "reward/max": buffer_stats["max_reward"],
            "reward/min": buffer_stats["min_reward"],
            "reward/std": buffer_stats["std_reward"],
            "training/learning_rate": epoch_stats["learning_rate"],
            "training/global_step": trainer.global_step,
        })
        
        # Save best model based on reward
        if buffer_stats["mean_reward"] > best_reward:
            best_reward = buffer_stats["mean_reward"]
            save_path = Path(args.output_dir) / "best_model"
            save_path.mkdir(parents=True, exist_ok=True)
            trainer.policy.save_pretrained(str(save_path))
            logger.info(f"✓ Saved best model (reward: {best_reward:.4f}) to {save_path}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch + 1}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(checkpoint_path / "trainer_state.pt"))
            logger.info(f"✓ Saved checkpoint to {checkpoint_path}")
    
    # Final save
    final_path = Path(args.output_dir) / "final_model"
    final_path.mkdir(parents=True, exist_ok=True)
    trainer.policy.save_pretrained(str(final_path))
    logger.info(f"✓ Saved final model to {final_path}")
    
    # Finish WandB
    wandb_logger.finish()
    
    logger.info("=" * 80)
    logger.info("GRPO RL Training Complete!")
    logger.info(f"Best Reward Achieved: {best_reward:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO RL Training for Reason-1")
    
    # Model arguments
    parser.add_argument(
        "--sft_model",
        type=str,
        default=None,
        help="Path to pre-trained SFT model",
    )
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5, help="Number of RL epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (num prompts)")
    parser.add_argument("--group_size", type=int, default=4, help="Group size G (outputs per prompt)")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--kl_coef", type=float, default=0.05, help="KL divergence coefficient")
    parser.add_argument("--updates_per_batch", type=int, default=4, help="Gradient updates per batch")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/rl", help="Output directory")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    
    # Experiment arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    
    args = parser.parse_args()
    
    main(args)
