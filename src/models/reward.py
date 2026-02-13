"""
Process Reward Model (PRM) for mathematical reasoning.
Provides fast, vectorized reward computation for GRPO training.

THE REWARD FUNCTION IS THE JUDGE - IF THIS IS BUGGY, THE RL WILL FAIL.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Optional: sympy for symbolic math equivalence
try:
    from sympy import sympify, simplify, N
    from sympy.parsing.sympy_parser import parse_expr
    SYMPY_AVAILABLE = True
except ImportError:
    logger.warning("sympy not available. Using exact string matching only.")
    SYMPY_AVAILABLE = False


@dataclass
class RewardComponents:
    """Individual components of the reward function."""
    format_reward: float  # Bonus for using <think> tags
    answer_reward: float  # Reward for correct answer
    length_penalty: float  # Penalty for verbosity
    repetition_penalty: float  # Penalty for repetitive reasoning
    total_reward: float  # Final combined reward


class MathVerifier:
    """
    Deterministic math verification using symbolic computation and string matching.
    """
    
    def __init__(
        self,
        use_symbolic: bool = True,
        exact_match_fallback: bool = True,
    ):
        """
        Initialize math verifier.
        
        Args:
            use_symbolic: Use sympy for symbolic equivalence checking
            exact_match_fallback: Fall back to string matching if symbolic fails
        """
        self.use_symbolic = use_symbolic and SYMPY_AVAILABLE
        self.exact_match_fallback = exact_match_fallback
    
    def extract_answer(self, text: str) -> Optional[str]:
        """
        Extract numerical answer from generated text.
        
        Looks for patterns like:
        - "The answer is 42"
        - "#### 42"
        - "Final answer: 42"
        - Last number in text
        
        Args:
            text: Generated text containing answer
        
        Returns:
            Extracted answer string or None
        """
        # Pattern 1: "The answer is X"
        match = re.search(r'(?:the\s+)?answer\s+is:?\s*([+-]?\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Pattern 2: GSM8K format "#### X"
        match = re.search(r'####\s*([+-]?\d+(?:\.\d+)?)', text)
        if match:
            return match.group(1).strip()
        
        # Pattern 3: "Final answer: X"
        match = re.search(r'final\s+answer:?\s*([+-]?\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Pattern 4: Number in box notation
        match = re.search(r'\\boxed\{([+-]?\d+(?:\.\d+)?)\}', text)
        if match:
            return match.group(1).strip()
        
        # Fallback: Last number in text
        numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', text)
        if numbers:
            return numbers[-1].strip()
        
        return None
    
    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answer string for comparison.
        
        Args:
            answer: Raw answer string
        
        Returns:
            Normalized answer
        """
        # Remove whitespace, commas, dollar signs
        normalized = re.sub(r'[,\s$]', '', answer)
        
        # Convert to lowercase
        normalized = normalized.lower()
        
        # Remove trailing zeros from decimals
        if '.' in normalized:
            normalized = normalized.rstrip('0').rstrip('.')
        
        return normalized
    
    def symbolic_equal(self, answer1: str, answer2: str) -> bool:
        """
        Check if two answers are symbolically equivalent using sympy.
        
        Args:
            answer1: First answer
            answer2: Second answer
        
        Returns:
            True if symbolically equivalent
        """
        if not self.use_symbolic:
            return False
        
        try:
            # Parse both expressions
            expr1 = sympify(answer1)
            expr2 = sympify(answer2)
            
            # Simplify and compare
            diff = simplify(expr1 - expr2)
            
            # Check if difference is zero (within numerical tolerance)
            if diff.is_number:
                return abs(float(N(diff))) < 1e-6
            
            return diff == 0
        
        except Exception as e:
            # Parsing or computation failed
            return False
    
    def verify(self, predicted: str, ground_truth: str) -> bool:
        """
        Verify if predicted answer matches ground truth.
        
        Args:
            predicted: Model's predicted answer
            ground_truth: Correct answer
        
        Returns:
            True if answers match
        """
        # Extract answers if they're in longer text
        pred_answer = self.extract_answer(predicted)
        gt_answer = self.extract_answer(ground_truth)
        
        if pred_answer is None or gt_answer is None:
            return False
        
        # Try symbolic equivalence first
        if self.use_symbolic:
            if self.symbolic_equal(pred_answer, gt_answer):
                return True
        
        # Fallback to normalized string matching
        if self.exact_match_fallback:
            pred_norm = self.normalize_answer(pred_answer)
            gt_norm = self.normalize_answer(gt_answer)
            return pred_norm == gt_norm
        
        return False


class ReasoningQualityChecker:
    """
    Check reasoning quality for penalty computation.
    Detects repetition and hallucinated steps.
    """
    
    def __init__(
        self,
        repetition_threshold: int = 3,
        max_line_length: int = 500,
    ):
        """
        Initialize quality checker.
        
        Args:
            repetition_threshold: Max allowed identical token sequences
            max_line_length: Max length of a single reasoning line
        """
        self.repetition_threshold = repetition_threshold
        self.max_line_length = max_line_length
    
    def count_token_repetitions(self, text: str, ngram_size: int = 5) -> int:
        """
        Count repeated n-gram sequences in text.
        
        Args:
            text: Text to analyze
            ngram_size: Size of n-grams to check
        
        Returns:
            Number of repeated sequences
        """
        words = text.split()
        if len(words) < ngram_size:
            return 0
        
        ngrams = []
        for i in range(len(words) - ngram_size + 1):
            ngram = ' '.join(words[i:i+ngram_size])
            ngrams.append(ngram)
        
        # Count duplicates
        seen = set()
        duplicates = 0
        for ngram in ngrams:
            if ngram in seen:
                duplicates += 1
            seen.add(ngram)
        
        return duplicates
    
    def detect_hallucination_markers(self, text: str) -> int:
        """
        Detect potential hallucination patterns.
        
        Args:
            text: Reasoning text
        
        Returns:
            Number of hallucination markers found
        """
        markers = [
            r'wait,?\s+(?:let me|i should)',  # Self-correction phrases
            r'(?:hmm|uh|er)',  # Hesitation
            r'i (?:think|believe|assume)',  # Uncertainty
        ]
        
        count = 0
        for pattern in markers:
            matches = re.findall(pattern, text, re.IGNORECASE)
            count += len(matches)
        
        return count
    
    def check_quality(self, reasoning_text: str) -> Dict[str, Any]:
        """
        Comprehensive reasoning quality check.
        
        Args:
            reasoning_text: The reasoning chain to analyze
        
        Returns:
            Dictionary with quality metrics
        """
        # Count repetitions
        repetitions = self.count_token_repetitions(reasoning_text)
        
        # Check for hallucination markers
        hallucinations = self.detect_hallucination_markers(reasoning_text)
        
        # Check length
        lines = reasoning_text.split('\n')
        long_lines = sum(1 for line in lines if len(line) > self.max_line_length)
        
        return {
            "repetitions": repetitions,
            "hallucinations": hallucinations,
            "long_lines": long_lines,
            "has_issues": repetitions > self.repetition_threshold or long_lines > 0,
        }


class RewardFunction:
    """
    Complete reward function for GRPO training.
    Fast, vectorized computation over batches.
    """
    
    def __init__(
        self,
        format_reward: float = 0.1,
        answer_reward: float = 1.0,
        length_penalty: float = 0.0,
        repetition_penalty: float = -0.1,
        repetition_threshold: int = 3,
        use_symbolic_math: bool = True,
        think_start_token: str = "<think>",
        think_end_token: str = "</think>",
    ):
        """
        Initialize reward function.
        
        Args:
            format_reward: Bonus for using reasoning tags
            answer_reward: Reward for correct answer
            length_penalty: Penalty per token
            repetition_penalty: Penalty for repetitive reasoning
            repetition_threshold: Number of repetitions before penalty
            use_symbolic_math: Use sympy for verification
            think_start_token: Start token for reasoning
            think_end_token: End token for reasoning
        """
        self.format_reward = format_reward
        self.answer_reward = answer_reward
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        
        # Initialize verifier and quality checker
        self.verifier = MathVerifier(use_symbolic=use_symbolic_math)
        self.quality_checker = ReasoningQualityChecker(
            repetition_threshold=repetition_threshold
        )
    
    def compute_format_reward(self, text: str) -> float:
        """
        Compute reward for proper use of reasoning tags.
        
        Args:
            text: Generated text
        
        Returns:
            Format reward value
        """
        has_start = self.think_start_token in text
        has_end = self.think_end_token in text
        
        # Reward if both tags are present and in correct order
        if has_start and has_end:
            start_idx = text.index(self.think_start_token)
            end_idx = text.rindex(self.think_end_token)
            if start_idx < end_idx:
                return self.format_reward
        
        return 0.0
    
    def extract_reasoning(self, text: str) -> str:
        """
        Extract reasoning content from between tags.
        
        Args:
            text: Full generated text
        
        Returns:
            Reasoning content or empty string
        """
        if self.think_start_token in text and self.think_end_token in text:
            try:
                start_idx = text.index(self.think_start_token) + len(self.think_start_token)
                end_idx = text.rindex(self.think_end_token)
                return text[start_idx:end_idx].strip()
            except:
                pass
        return ""
    
    def compute_reward(
        self,
        generated_text: str,
        ground_truth: str,
        num_tokens: Optional[int] = None,
    ) -> RewardComponents:
        """
        Compute complete reward for a single generation.
        
        Args:
            generated_text: Model's generated output
            ground_truth: Correct answer
            num_tokens: Number of generated tokens (for length penalty)
        
        Returns:
            RewardComponents with breakdown
        """
        # 1. Format reward
        format_rew = self.compute_format_reward(generated_text)
        
        # 2. Answer correctness reward
        answer_correct = self.verifier.verify(generated_text, ground_truth)
        answer_rew = self.answer_reward if answer_correct else 0.0
        
        # 3. Length penalty
        length_pen = 0.0
        if num_tokens is not None and self.length_penalty > 0:
            length_pen = -self.length_penalty * num_tokens
        
        # 4. Reasoning quality penalty
        repetition_pen = 0.0
        reasoning = self.extract_reasoning(generated_text)
        if reasoning:
            quality = self.quality_checker.check_quality(reasoning)
            if quality["repetitions"] > quality["repetitions"]:
                repetition_pen = self.repetition_penalty * quality["repetitions"]
        
        # Total reward
        total = format_rew + answer_rew + length_pen + repetition_pen
        
        return RewardComponents(
            format_reward=format_rew,
            answer_reward=answer_rew,
            length_penalty=length_pen,
            repetition_penalty=repetition_pen,
            total_reward=total,
        )
    
    def compute_batch_rewards(
        self,
        generated_texts: List[str],
        ground_truths: List[str],
        num_tokens_list: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, List[RewardComponents]]:
        """
        Compute rewards for a batch of generations (vectorized).
        
        Args:
            generated_texts: List of generated outputs
            ground_truths: List of correct answers
            num_tokens_list: List of token counts
        
        Returns:
            Tuple of (reward_tensor, reward_components_list)
        """
        batch_size = len(generated_texts)
        
        if num_tokens_list is None:
            num_tokens_list = [None] * batch_size
        
        # Compute rewards for each example
        rewards = []
        components = []
        
        for text, gt, num_tokens in zip(generated_texts, ground_truths, num_tokens_list):
            comp = self.compute_reward(text, gt, num_tokens)
            rewards.append(comp.total_reward)
            components.append(comp)
        
        # Convert to tensor
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        return reward_tensor, components
    
    def __call__(
        self,
        generated_texts: List[str],
        ground_truths: List[str],
        num_tokens_list: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Callable interface for easy integration.
        
        Args:
            generated_texts: List of generated outputs
            ground_truths: List of correct answers
            num_tokens_list: List of token counts
        
        Returns:
            Tensor of rewards
        """
        rewards, _ = self.compute_batch_rewards(
            generated_texts,
            ground_truths,
            num_tokens_list,
        )
        return rewards


def create_reward_function(
    format_reward: float = 0.1,
    answer_reward: float = 1.0,
    length_penalty: float = 0.0,
    repetition_penalty: float = -0.1,
    use_symbolic_math: bool = True,
    **kwargs,
) -> RewardFunction:
    """
    Factory function to create reward function.
    
    Args:
        format_reward: Bonus for using reasoning tags
        answer_reward: Reward for correct answer
        length_penalty: Penalty per token
        repetition_penalty: Penalty for repetition
        use_symbolic_math: Use sympy for verification
        **kwargs: Additional parameters
    
    Returns:
        Configured RewardFunction instance
    """
    return RewardFunction(
        format_reward=format_reward,
        answer_reward=answer_reward,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        use_symbolic_math=use_symbolic_math,
        **kwargs,
    )
