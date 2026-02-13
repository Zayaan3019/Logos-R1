"""
Stage 1: Supervised Fine-Tuning (SFT) - Cold Start

This script fine-tunes the base model on reasoning traces to initialize
it with basic reasoning capabilities before RL training.

Usage:
    python scripts/train_sft.py --model_name meta-llama/Llama-3-8B --epochs 3
"""

import os
import sys
import argparse
import torch
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TrainingConfig, ModelConfig, DataConfig, get_default_config
from src.data.tokenizer import create_tokenizer
from src.data.loader import create_dataloader
from src.models.policy import create_policy
from src.utils.logging import setup_logging, WandBLogger, MetricsTracker

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
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model,
    dataloader,
    optimizer,
    device: str,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool = True,
) -> dict:
    """
    Train for one epoch.
    
    Args:
        model: Policy model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device for training
        scaler: AMP gradient scaler
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        Dictionary of training metrics
    """
    model.train()
    metrics_tracker = MetricsTracker()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with AMP
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"]
            
            # Backward with scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"]
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Track metrics
        metrics_tracker.update({
            "loss": loss.item(),
        })
        
        # Update progress bar
        if batch_idx % 10 == 0:
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
            })
    
    return metrics_tracker.get_averages()


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device: str,
) -> dict:
    """
    Evaluate model on validation set.
    
    Args:
        model: Policy model
        dataloader: Validation dataloader
        device: Device for evaluation
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    metrics_tracker = MetricsTracker()
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        metrics_tracker.update({
            "loss": outputs["loss"].item(),
        })
    
    return metrics_tracker.get_averages()


def main(args):
    """Main training function."""
    
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    setup_logging(
        log_level="INFO",
        log_file=str(log_dir / "train_sft.log"),
    )
    
    logger.info("=" * 80)
    logger.info("Stage 1: Supervised Fine-Tuning (SFT)")
    logger.info("=" * 80)
    
    # Load configuration
    config = get_default_config()
    config.model.model_name = args.model_name
    config.seed = args.seed
    
    # Set random seed
    set_seed(config.seed)
    logger.info(f"Set random seed: {config.seed}")
    
    # Initialize WandB
    wandb_logger = WandBLogger(
        project=config.wandb_project,
        run_name=args.run_name or f"sft_{config.model.model_name.split('/')[-1]}",
        config=vars(args),
        enabled=config.use_wandb,
    )
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create tokenizer
    logger.info("Creating tokenizer...")
    tokenizer = create_tokenizer(
        model_name=config.model.model_name,
        think_start_token=config.model.think_start_token,
        think_end_token=config.model.think_end_token,
    )
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader = create_dataloader(
        config=config.data,
        tokenizer=tokenizer,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    test_dataloader = create_dataloader(
        config=config.data,
        tokenizer=tokenizer,
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    logger.info(f"Train samples: {len(train_dataloader.dataset)}")
    logger.info(f"Test samples: {len(test_dataloader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_policy(
        config=config.model,
        tokenizer_len=len(tokenizer),
    )
    model = model.to(device)
    
    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    
    # AMP scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=args.use_amp,
        )
        
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        
        # Evaluate
        val_metrics = evaluate(
            model=model,
            dataloader=test_dataloader,
            device=device,
        )
        
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        
        # Log to WandB
        wandb_logger.log({
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "val/loss": val_metrics["loss"],
        })
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_path = Path(args.output_dir) / "best_model"
            save_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(save_path))
            logger.info(f"Saved best model to {save_path}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch + 1}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(checkpoint_path))
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Final save
    final_path = Path(args.output_dir) / "final_model"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_path))
    logger.info(f"Saved final model to {final_path}")
    
    # Finish WandB
    wandb_logger.finish()
    
    logger.info("=" * 80)
    logger.info("SFT Training Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for Reason-1")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3-8B",
        help="Base model name from HuggingFace"
    )
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/sft", help="Output directory")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    
    # Experiment arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    
    args = parser.parse_args()
    
    main(args)
