"""
WandB logging utilities for tracking training metrics.
"""

from typing import Dict, Any, Optional
import wandb
import logging
import os

logger = logging.getLogger(__name__)


class WandBLogger:
    """
    Wrapper for Weights & Biases logging.
    """
    
    def __init__(
        self,
        project: str = "reason-1",
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        """
        Initialize WandB logger.
        
        Args:
            project: WandB project name
            entity: WandB entity (username/team)
            run_name: Name for this run
            config: Configuration dictionary to log
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        
        if not enabled:
            logger.info("WandB logging disabled")
            return
        
        # Initialize WandB
        try:
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config,
                reinit=True,
            )
            logger.info(f"Initialized WandB: {self.run.url}")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            self.enabled = False
    
    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """
        Log metrics to WandB.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Global step number
            commit: Whether to commit immediately
        """
        if not self.enabled:
            return
        
        try:
            wandb.log(metrics, step=step, commit=commit)
        except Exception as e:
            logger.warning(f"Failed to log to WandB: {e}")
    
    def log_reward_distribution(
        self,
        rewards: list,
        step: int,
        prefix: str = "reward",
    ):
        """
        Log reward distribution as histogram.
        
        Args:
            rewards: List of reward values
            step: Global step
            prefix: Metric name prefix
        """
        if not self.enabled or not rewards:
            return
        
        try:
            wandb.log({
                f"{prefix}/histogram": wandb.Histogram(rewards),
                f"{prefix}/mean": sum(rewards) / len(rewards),
                f"{prefix}/max": max(rewards),
                f"{prefix}/min": min(rewards),
            }, step=step)
        except Exception as e:
            logger.warning(f"Failed to log reward distribution: {e}")
    
    def log_text_samples(
        self,
        prompts: list,
        outputs: list,
        scores: list,
        step: int,
        num_samples: int = 3,
    ):
        """
        Log text generation samples to WandB.
        
        Args:
            prompts: List of prompts
            outputs: List of generated outputs
            scores: List of reward scores
            step: Global step
            num_samples: Number of samples to log
        """
        if not self.enabled:
            return
        
        try:
            # Create table
            table = wandb.Table(columns=["Prompt", "Output", "Score"])
            
            for i in range(min(num_samples, len(prompts))):
                table.add_data(prompts[i], outputs[i], scores[i])
            
            wandb.log({f"samples": table}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log text samples: {e}")
    
    def watch_model(self, model, log: str = "gradients", log_freq: int = 100):
        """
        Watch model gradients and parameters.
        
        Args:
            model: PyTorch model to watch
            log: What to log ("gradients", "parameters", "all")
            log_freq: Logging frequency
        """
        if not self.enabled:
            return
        
        try:
            wandb.watch(model, log=log, log_freq=log_freq)
        except Exception as e:
            logger.warning(f"Failed to watch model: {e}")
    
    def finish(self):
        """Finish WandB run."""
        if self.enabled and hasattr(self, 'run'):
            try:
                wandb.finish()
                logger.info("Finished WandB run")
            except Exception as e:
                logger.warning(f"Failed to finish WandB run: {e}")


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Setup Python logging configuration.
    
    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Optional file to write logs to
    
    Returns:
        Configured logger
    """
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Setup handlers
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,
    )
    
    # Return root logger
    logger = logging.getLogger()
    logger.info(f"Logging configured: level={log_level}, file={log_file}")
    
    return logger


class MetricsTracker:
    """
    Simple metrics tracker for aggregating training statistics.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, float]):
        """
        Update tracked metrics.
        
        Args:
            metrics: Dictionary of metric values
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value
            self.counts[key] += 1
    
    def get_averages(self) -> Dict[str, float]:
        """
        Get average values for all metrics.
        
        Returns:
            Dictionary of averaged metrics
        """
        averages = {}
        for key in self.metrics:
            if self.counts[key] > 0:
                averages[key] = self.metrics[key] / self.counts[key]
        return averages
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.counts.clear()
    
    def __str__(self) -> str:
        """String representation of current averages."""
        averages = self.get_averages()
        return " | ".join([f"{k}: {v:.4f}" for k, v in averages.items()])
