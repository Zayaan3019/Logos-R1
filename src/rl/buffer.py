"""
Experience Replay Buffer for GRPO training.
Stores groups of (prompt, outputs, rewards) for batch training.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperienceGroup:
    """
    A group of experiences for a single prompt.
    In GRPO, we sample G outputs per prompt and compute relative advantages.
    """
    prompt: str
    prompt_ids: torch.Tensor  # [seq_len]
    outputs: List[str]  # G outputs
    output_ids: List[torch.Tensor]  # G x [seq_len]
    rewards: torch.Tensor  # [G]
    ground_truth: str
    
    # Computed fields
    advantages: Optional[torch.Tensor] = None  # [G]
    mean_reward: Optional[float] = None
    std_reward: Optional[float] = None


class GroupExperienceBuffer:
    """
    Buffer for storing and sampling experience groups.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        group_size: int = 4,
    ):
        """
        Initialize experience buffer.
        
        Args:
            max_size: Maximum number of groups to store
            group_size: Number of outputs per prompt (G)
        """
        self.max_size = max_size
        self.group_size = group_size
        self.buffer: deque = deque(maxlen=max_size)
        
        logger.info(f"Initialized buffer with max_size={max_size}, group_size={group_size}")
    
    def add(self, experience: ExperienceGroup):
        """
        Add an experience group to the buffer.
        
        Args:
            experience: ExperienceGroup to add
        """
        # Compute advantages if not already computed
        if experience.advantages is None:
            experience.advantages = self._compute_advantages(experience.rewards)
            experience.mean_reward = experience.rewards.mean().item()
            experience.std_reward = experience.rewards.std().item()
        
        self.buffer.append(experience)
    
    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute group-relative advantages.
        
        GRPO Key Innovation: Advantage = (r_i - mean(r)) / (std(r) + epsilon)
        
        Args:
            rewards: Reward tensor [G]
            epsilon: Numerical stability constant
        
        Returns:
            Advantages [G]
        """
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        
        advantages = (rewards - mean_reward) / (std_reward + epsilon)
        
        return advantages
    
    def sample_batch(self, batch_size: int) -> List[ExperienceGroup]:
        """
        Sample a batch of experience groups.
        
        Args:
            batch_size: Number of groups to sample
        
        Returns:
            List of ExperienceGroup
        """
        if len(self.buffer) < batch_size:
            # Return all if buffer is smaller
            return list(self.buffer)
        
        # Random sampling without replacement
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def get_all(self) -> List[ExperienceGroup]:
        """
        Get all experiences in buffer.
        
        Returns:
            List of all ExperienceGroup
        """
        return list(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        logger.info("Cleared experience buffer")
    
    def __len__(self) -> int:
        """Return buffer size."""
        return len(self.buffer)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get buffer statistics for logging.
        
        Returns:
            Dictionary of statistics
        """
        if len(self.buffer) == 0:
            return {
                "buffer_size": 0,
                "mean_reward": 0.0,
                "max_reward": 0.0,
                "min_reward": 0.0,
            }
        
        all_rewards = torch.cat([exp.rewards for exp in self.buffer])
        
        return {
            "buffer_size": len(self.buffer),
            "mean_reward": all_rewards.mean().item(),
            "max_reward": all_rewards.max().item(),
            "min_reward": all_rewards.min().item(),
            "std_reward": all_rewards.std().item(),
        }


class OnlineExperienceCollector:
    """
    Collects experiences online during RL training.
    """
    
    def __init__(
        self,
        policy,
        reward_function,
        tokenizer,
        group_size: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        device: str = "cuda",
    ):
        """
        Initialize experience collector.
        
        Args:
            policy: Policy model for generation
            reward_function: Reward function for scoring
            tokenizer: Tokenizer for encoding/decoding
            group_size: Number of outputs per prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            device: Device for computation
        """
        self.policy = policy
        self.reward_function = reward_function
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
    
    @torch.no_grad()
    def collect_group(
        self,
        prompt: str,
        ground_truth: str,
    ) -> ExperienceGroup:
        """
        Collect a group of experiences for a single prompt.
        
        Args:
            prompt: Input prompt
            ground_truth: Correct answer for reward computation
        
        Returns:
            ExperienceGroup with G samples
        """
        # Tokenize prompt
        prompt_encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        
        prompt_ids = prompt_encoded["input_ids"][0]
        
        # Generate G outputs
        generated_ids = self.policy.generate(
            input_ids=prompt_encoded["input_ids"],
            attention_mask=prompt_encoded["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            num_return_sequences=self.group_size,
        )  # [G, total_seq_len]
        
        # Decode outputs
        outputs = []
        output_ids = []
        
        for i in range(self.group_size):
            # Extract only the generated part (after prompt)
            generated = generated_ids[i, prompt_ids.shape[0]:]
            output_ids.append(generated.cpu())
            
            # Decode full output
            full_output = self.tokenizer.decode(generated, skip_special_tokens=False)
            outputs.append(full_output)
        
        # Compute rewards
        rewards = self.reward_function(
            generated_texts=outputs,
            ground_truths=[ground_truth] * self.group_size,
        )  # [G]
        
        # Create experience group
        experience = ExperienceGroup(
            prompt=prompt,
            prompt_ids=prompt_ids.cpu(),
            outputs=outputs,
            output_ids=output_ids,
            rewards=rewards.cpu(),
            ground_truth=ground_truth,
        )
        
        return experience
    
    def collect_batch(
        self,
        prompts: List[str],
        ground_truths: List[str],
    ) -> List[ExperienceGroup]:
        """
        Collect experiences for a batch of prompts.
        
        Args:
            prompts: List of prompts
            ground_truths: List of correct answers
        
        Returns:
            List of ExperienceGroup
        """
        experiences = []
        
        for prompt, gt in zip(prompts, ground_truths):
            exp = self.collect_group(prompt, gt)
            experiences.append(exp)
        
        return experiences
