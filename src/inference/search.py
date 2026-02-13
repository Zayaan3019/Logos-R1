"""
Best-of-N Rejection Sampling for Inference.

Generates N candidate solutions, scores them with the reward function,
and returns the best one. This is the inference strategy for Reason-1.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
from dataclasses import dataclass
import logging

from ..models.reward import RewardFunction, RewardComponents

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from Best-of-N search."""
    best_output: str
    best_score: float
    all_outputs: List[str]
    all_scores: List[float]
    best_index: int
    reasoning: str
    answer: str


class BestOfNSampler:
    """
    Best-of-N rejection sampling with reward-based selection.
    
    Algorithm:
    ----------
    1. Generate N candidate solutions in parallel
    2. Score each with the reward function
    3. Return the highest-scoring solution
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        reward_function: RewardFunction,
        num_samples: int = 8,
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
        device: str = "cuda",
    ):
        """
        Initialize Best-of-N sampler.
        
        Args:
            model: Policy model for generation
            tokenizer: Tokenizer
            reward_function: Reward function for scoring
            num_samples: Number of candidates to generate
            max_new_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.reward_function = reward_function
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        
        logger.info(f"Initialized Best-of-N sampler with N={num_samples}")
    
    @torch.no_grad()
    def generate_candidates(self, prompt: str) -> List[str]:
        """
        Generate N candidate solutions for a prompt.
        
        Args:
            prompt: Input prompt
        
        Returns:
            List of N generated solutions
        """
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        
        # Generate N outputs in parallel
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            num_return_sequences=self.num_samples,
        )
        
        # Decode candidates
        candidates = []
        prompt_length = inputs["input_ids"].shape[1]
        
        for i in range(self.num_samples):
            # Extract only generated part (after prompt)
            generated_ids = outputs[i, prompt_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            candidates.append(generated_text)
        
        return candidates
    
    def score_candidates(
        self,
        candidates: List[str],
        ground_truth: Optional[str] = None,
    ) -> List[float]:
        """
        Score candidates using reward function.
        
        Args:
            candidates: List of candidate solutions
            ground_truth: Optional ground truth for verification
        
        Returns:
            List of scores
        """
        if ground_truth is None:
            # Without ground truth, use format rewards only
            scores = []
            for candidate in candidates:
                comp = self.reward_function.compute_reward(candidate, "")
                # Use format reward + length as heuristic
                score = comp.format_reward - len(candidate.split()) * 0.001
                scores.append(score)
        else:
            # With ground truth, use full reward
            ground_truths = [ground_truth] * len(candidates)
            reward_tensor = self.reward_function(candidates, ground_truths)
            scores = reward_tensor.tolist()
        
        return scores
    
    def search(
        self,
        prompt: str,
        ground_truth: Optional[str] = None,
    ) -> SearchResult:
        """
        Perform Best-of-N search for a prompt.
        
        Args:
            prompt: Input prompt
            ground_truth: Optional ground truth for scoring
        
        Returns:
            SearchResult with best candidate and metadata
        """
        # Generate candidates
        candidates = self.generate_candidates(prompt)
        
        # Score candidates
        scores = self.score_candidates(candidates, ground_truth)
        
        # Find best candidate
        best_idx = scores.index(max(scores))
        best_output = candidates[best_idx]
        best_score = scores[best_idx]
        
        # Extract reasoning and answer from best output
        parsed = self.tokenizer.decode_with_reasoning_sep(
            self.tokenizer.encode(best_output),
            skip_special_tokens=True,
        )
        
        result = SearchResult(
            best_output=best_output,
            best_score=best_score,
            all_outputs=candidates,
            all_scores=scores,
            best_index=best_idx,
            reasoning=parsed.get("reasoning", ""),
            answer=parsed.get("answer", ""),
        )
        
        logger.debug(
            f"Best-of-{self.num_samples} search: "
            f"best_score={best_score:.3f}, mean_score={sum(scores)/len(scores):.3f}"
        )
        
        return result


class BeamSearchReasoning:
    """
    Beam search with reasoning-aware scoring (experimental).
    
    Maintains B beams, expands them, and prunes based on:
    1. Language model probability
    2. Partial reward (if answer detected early)
    3. Reasoning quality heuristics
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        reward_function: RewardFunction,
        beam_width: int = 4,
        max_new_tokens: int = 2048,
        length_penalty: float = 1.0,
        device: str = "cuda",
    ):
        """
        Initialize beam search.
        
        Args:
            model: Policy model
            tokenizer: Tokenizer
            reward_function: Reward function
            beam_width: Number of beams
            max_new_tokens: Maximum tokens to generate
            length_penalty: Length normalization factor
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.reward_function = reward_function
        self.beam_width = beam_width
        self.max_new_tokens = max_new_tokens
        self.length_penalty = length_penalty
        self.device = device
        
        logger.info(f"Initialized Beam Search with beam_width={beam_width}")
    
    @torch.no_grad()
    def search(
        self,
        prompt: str,
        ground_truth: Optional[str] = None,
    ) -> SearchResult:
        """
        Perform beam search (simplified version).
        
        Args:
            prompt: Input prompt
            ground_truth: Optional ground truth
        
        Returns:
            SearchResult with best beam
        """
        # For simplicity, use HuggingFace's beam search
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.beam_width,
            num_return_sequences=self.beam_width,
            length_penalty=self.length_penalty,
            early_stopping=True,
        )
        
        # Decode beams
        candidates = []
        prompt_length = inputs["input_ids"].shape[1]
        
        for i in range(self.beam_width):
            generated_ids = outputs[i, prompt_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            candidates.append(generated_text)
        
        # Score and select best
        scores = []
        for candidate in candidates:
            if ground_truth:
                comp = self.reward_function.compute_reward(candidate, ground_truth)
                scores.append(comp.total_reward)
            else:
                comp = self.reward_function.compute_reward(candidate, "")
                scores.append(comp.format_reward)
        
        best_idx = scores.index(max(scores))
        
        parsed = self.tokenizer.decode_with_reasoning_sep(
            self.tokenizer.encode(candidates[best_idx]),
            skip_special_tokens=True,
        )
        
        return SearchResult(
            best_output=candidates[best_idx],
            best_score=scores[best_idx],
            all_outputs=candidates,
            all_scores=scores,
            best_index=best_idx,
            reasoning=parsed.get("reasoning", ""),
            answer=parsed.get("answer", ""),
        )


def create_sampler(
    model,
    tokenizer,
    reward_function: RewardFunction,
    strategy: str = "best_of_n",
    num_samples: int = 8,
    **kwargs,
) -> Any:
    """
    Factory function to create search sampler.
    
    Args:
        model: Policy model
        tokenizer: Tokenizer
        reward_function: Reward function
        strategy: "best_of_n" or "beam_search"
        num_samples: Number of samples/beams
        **kwargs: Additional parameters
    
    Returns:
        Sampler instance
    """
    if strategy == "best_of_n":
        return BestOfNSampler(
            model=model,
            tokenizer=tokenizer,
            reward_function=reward_function,
            num_samples=num_samples,
            **kwargs,
        )
    elif strategy == "beam_search":
        return BeamSearchReasoning(
            model=model,
            tokenizer=tokenizer,
            reward_function=reward_function,
            beam_width=num_samples,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown search strategy: {strategy}")
