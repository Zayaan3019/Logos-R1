"""
GRPO (Group Relative Policy Optimization) Trainer.

THE CORE ALGORITHM - Implementation of Group Relative Policy Optimization.

GRPO is a PPO variant that:
1. Samples G outputs per prompt from the current policy
2. Computes group-relative advantages (no value function needed!)
3. Uses PPO-clip objective with KL divergence penalty
4. Updates policy to favor high-advantage samples
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import numpy as np

from ..config import GRPOConfig
from ..models.policy import ReasoningPolicy, ReferencePolicy
from ..models.reward import RewardFunction
from .buffer import GroupExperienceBuffer, ExperienceGroup, OnlineExperienceCollector

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """
    Group Relative Policy Optimization Trainer.
    
    Algorithm:
    ----------
    For each training iteration:
        1. Sample G outputs for each prompt: {o_1, ..., o_G} ~ π_θ(·|q)
        2. Compute rewards: r_i = R(o_i, ground_truth)
        3. Compute advantages: A_i = (r_i - mean(r)) / (std(r) + ε)
        4. Update policy using PPO-clip objective with KL penalty
    """
    
    def __init__(
        self,
        policy: ReasoningPolicy,
        reference_policy: ReferencePolicy,
        reward_function: RewardFunction,
        tokenizer,
        config: GRPOConfig,
        device: str = "cuda",
    ):
        """
        Initialize GRPO trainer.
        
        Args:
            policy: Policy model to train
            reference_policy: Frozen reference policy for KL
            reward_function: Reward function
            tokenizer: Tokenizer
            config: GRPO configuration
            device: Device for training
        """
        self.policy = policy
        self.reference_policy = reference_policy
        self.reward_function = reward_function
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Move models to device
        self.policy = self.policy.to(device)
        self.reference_policy = self.reference_policy.to(device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize experience buffer and collector
        self.buffer = GroupExperienceBuffer(
            max_size=1000,
            group_size=config.group_size,
        )
        
        self.collector = OnlineExperienceCollector(
            policy=self.policy,
            reward_function=self.reward_function,
            tokenizer=self.tokenizer,
            group_size=config.group_size,
            max_new_tokens=512,
            temperature=config.temperature,
            top_p=config.top_p,
            device=device,
        )
        
        # Training statistics
        self.global_step = 0
        self.epoch = 0
        self.stats = {
            "total_rewards": [],
            "advantages": [],
            "kl_divergences": [],
            "policy_losses": [],
        }
        
        logger.info("Initialized GRPO Trainer")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer with weight decay.
        
        Returns:
            AdamW optimizer
        """
        # Separate parameters into decay and no-decay groups
        decay_params = []
        no_decay_params = []
        
        for name, param in self.policy.named_parameters():
            if param.requires_grad:
                if "bias" in name or "LayerNorm" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
        
        return optimizer
    
    def create_scheduler(self, num_training_steps: int):
        """
        Create learning rate scheduler.
        
        Args:
            num_training_steps: Total training steps
        """
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    def compute_kl_divergence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between current policy and reference policy.
        
        KL(π_θ || π_ref) = Σ π_θ(a|s) * [log π_θ(a|s) - log π_ref(a|s)]
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
        
        Returns:
            KL divergence scalar
        """
        # Get log probs from current policy
        policy_log_probs = self.policy.get_log_probs(input_ids, attention_mask)
        
        # Get log probs from reference policy
        with torch.no_grad():
            ref_log_probs = self.reference_policy.get_log_probs(input_ids, attention_mask)
        
        # Compute KL: E[log(π) - log(π_ref)]
        # We approximate by taking KL at each position and averaging
        kl_div = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).sum(dim=-1)
        
        # Mask and average
        if attention_mask is not None:
            mask = attention_mask[:, 1:].float()  # Shift for next-token prediction
            kl_div = kl_div[:, :-1]  # Match shifted dimension
            kl_div = (kl_div * mask).sum() / mask.sum()
        else:
            kl_div = kl_div.mean()
        
        return kl_div
    
    def compute_ppo_loss(
        self,
        experience: ExperienceGroup,
        mini_batch_indices: List[int],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO-clip loss for a mini-batch from an experience group.
        
        Args:
            experience: ExperienceGroup with advantages computed
            mini_batch_indices: Indices of outputs to use (subset of G)
        
        Returns:
            Tuple of (loss, stats_dict)
        """
        batch_advantages = experience.advantages[mini_batch_indices].to(self.device)
        
        # Prepare input sequences (prompt + output)
        input_ids_list = []
        attention_mask_list = []
        
        for idx in mini_batch_indices:
            # Concatenate prompt and output
            full_ids = torch.cat([
                experience.prompt_ids,
                experience.output_ids[idx],
            ])
            
            input_ids_list.append(full_ids)
            attention_mask_list.append(torch.ones_like(full_ids))
        
        # Pad sequences to same length
        max_len = max(ids.shape[0] for ids in input_ids_list)
        
        input_ids = torch.zeros(len(mini_batch_indices), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(mini_batch_indices), max_len, dtype=torch.long)
        
        for i, (ids, mask) in enumerate(zip(input_ids_list, attention_mask_list)):
            seq_len = ids.shape[0]
            input_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = mask
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Get current policy log probs
        current_log_probs = self.policy.compute_sequence_log_probs(
            input_ids, attention_mask
        )  # [mini_batch]
        
        # Get old policy log probs (compute once, treat as constant)
        with torch.no_grad():
            old_log_probs = current_log_probs.detach().clone()
        
        # Compute probability ratio: π_θ(a|s) / π_θ_old(a|s)
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # PPO-clip objective
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.config.clip_range,
            1.0 + self.config.clip_range,
        )
        
        # Policy loss: -E[min(ratio * A, clip(ratio) * A)]
        policy_loss_unclipped = ratio * batch_advantages
        policy_loss_clipped = clipped_ratio * batch_advantages
        policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped).mean()
        
        # KL divergence penalty
        kl_div = self.compute_kl_divergence(input_ids, attention_mask)
        kl_penalty = self.config.kl_coef * kl_div
        
        # Total loss
        total_loss = policy_loss + kl_penalty
        
        # Statistics
        stats = {
            "policy_loss": policy_loss.item(),
            "kl_divergence": kl_div.item(),
            "kl_penalty": kl_penalty.item(),
            "total_loss": total_loss.item(),
            "mean_ratio": ratio.mean().item(),
            "mean_advantage": batch_advantages.mean().item(),
        }
        
        return total_loss, stats
    
    def train_step(
        self,
        experiences: List[ExperienceGroup],
    ) -> Dict[str, float]:
        """
        Perform one training step on a batch of experience groups.
        
        Args:
            experiences: List of ExperienceGroup
        
        Returns:
            Dictionary of training statistics
        """
        self.policy.train()
        
        total_loss = 0.0
        all_stats = []
        
        # Iterate through each experience group
        for experience in experiences:
            # Use all G outputs in the group
            mini_batch_indices = list(range(self.config.group_size))
            
            # Compute loss
            loss, stats = self.compute_ppo_loss(experience, mini_batch_indices)
            
            total_loss += loss
            all_stats.append(stats)
        
        # Average loss across groups
        total_loss = total_loss / len(experiences)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.config.max_grad_norm,
        )
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
        
        self.global_step += 1
        
        # Aggregate statistics
        aggregated_stats = {
            key: np.mean([s[key] for s in all_stats])
            for key in all_stats[0].keys()
        }
        
        aggregated_stats["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        aggregated_stats["global_step"] = self.global_step
        
        return aggregated_stats
    
    def train_epoch(
        self,
        dataloader,
        num_updates_per_batch: int = 4,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader yielding prompts and ground truths
            num_updates_per_batch: Number of gradient updates per batch
        
        Returns:
            Epoch statistics
        """
        epoch_stats = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch in progress_bar:
            prompts = batch["prompt"]
            ground_truths = batch["ground_truth"]
            
            # Collect experiences
            experiences = self.collector.collect_batch(prompts, ground_truths)
            
            # Add to buffer
            for exp in experiences:
                self.buffer.add(exp)
            
            # Perform multiple updates on the collected experiences
            for _ in range(num_updates_per_batch):
                # Sample from buffer
                sampled_experiences = self.buffer.sample_batch(
                    batch_size=self.config.batch_size
                )
                
                # Training step
                stats = self.train_step(sampled_experiences)
                epoch_stats.append(stats)
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": stats["total_loss"],
                    "reward": self.buffer.get_statistics()["mean_reward"],
                    "kl": stats["kl_divergence"],
                })
        
        self.epoch += 1
        
        # Aggregate epoch statistics
        epoch_summary = {
            key: np.mean([s[key] for s in epoch_stats])
            for key in epoch_stats[0].keys()
        }
        
        return epoch_summary
    
    def save_checkpoint(self, save_path: str):
        """
        Save training checkpoint.
        
        Args:
            save_path: Path to save checkpoint
        """
        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "optimizer_state": self.optimizer.state_dict(),
            "stats": self.stats,
        }
        
        torch.save(checkpoint, save_path)
        self.policy.save_pretrained(save_path + "_policy")
        
        logger.info(f"Saved checkpoint to {save_path}")
    
    def load_checkpoint(self, load_path: str):
        """
        Load training checkpoint.
        
        Args:
            load_path: Path to load checkpoint
        """
        checkpoint = torch.load(load_path)
        
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.stats = checkpoint["stats"]
        
        self.policy.load_pretrained(load_path + "_policy")
        
        logger.info(f"Loaded checkpoint from {load_path}")
