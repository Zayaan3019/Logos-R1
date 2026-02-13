"""
Unit tests for the Reward Function.

THE REWARD FUNCTION IS CRITICAL - These tests ensure it works correctly.
"""

import pytest
import torch
from src.models.reward import (
    RewardFunction,
    MathVerifier,
    ReasoningQualityChecker,
    create_reward_function,
)


class TestMathVerifier:
    """Test cases for mathematical answer verification."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.verifier = MathVerifier(use_symbolic=True, exact_match_fallback=True)
    
    def test_extract_answer_basic(self):
        """Test basic answer extraction."""
        text = "The answer is 42"
        answer = self.verifier.extract_answer(text)
        assert answer == "42"
    
    def test_extract_answer_gsm8k_format(self):
        """Test GSM8K format extraction."""
        text = "Step 1: Do something\n#### 123"
        answer = self.verifier.extract_answer(text)
        assert answer == "123"
    
    def test_extract_answer_with_decimals(self):
        """Test decimal number extraction."""
        text = "The final answer is 3.14159"
        answer = self.verifier.extract_answer(text)
        assert answer == "3.14159"
    
    def test_extract_answer_negative(self):
        """Test negative number extraction."""
        text = "The result is -42"
        answer = self.verifier.extract_answer(text)
        assert answer == "-42"
    
    def test_normalize_answer(self):
        """Test answer normalization."""
        assert self.verifier.normalize_answer("1,234") == "1234"
        assert self.verifier.normalize_answer("$100") == "100"
        assert self.verifier.normalize_answer("42.00") == "42"
    
    def test_verify_exact_match(self):
        """Test exact answer matching."""
        assert self.verifier.verify("The answer is 42", "42")
        assert not self.verifier.verify("The answer is 42", "43")
    
    def test_verify_with_formatting(self):
        """Test matching with different formatting."""
        assert self.verifier.verify("The answer is 1,234", "1234")
        assert self.verifier.verify("#### 100.0", "100")
    
    def test_verify_no_answer(self):
        """Test verification when no answer is found."""
        assert not self.verifier.verify("There is no answer here", "42")


class TestReasoningQualityChecker:
    """Test cases for reasoning quality checking."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.checker = ReasoningQualityChecker(repetition_threshold=3)
    
    def test_no_repetition(self):
        """Test text without repetition."""
        text = "Step 1: Add 5 and 3 to get 8. Step 2: Multiply by 2 to get 16."
        result = self.checker.check_quality(text)
        assert result["repetitions"] == 0
        assert not result["has_issues"]
    
    def test_with_repetition(self):
        """Test text with repetition."""
        text = "Let me think about this. Let me think about this. Let me think about this."
        result = self.checker.check_quality(text)
        assert result["repetitions"] > 0
    
    def test_hallucination_markers(self):
        """Test detection of hallucination markers."""
        text = "I think the answer is 5, but wait, let me double check."
        result = self.checker.check_quality(text)
        assert result["hallucinations"] > 0


class TestRewardFunction:
    """Test cases for the complete reward function."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.reward_fn = create_reward_function(
            format_reward=0.1,
            answer_reward=1.0,
            length_penalty=0.0,
            repetition_penalty=-0.1,
            think_start_token="<think>",
            think_end_token="</think>",
        )
    
    def test_format_reward_present(self):
        """Test format reward when tags are present."""
        text = "<think>Step 1: Add 5 + 3</think> The answer is 8"
        reward = self.reward_fn.compute_format_reward(text)
        assert reward == 0.1
    
    def test_format_reward_absent(self):
        """Test format reward when tags are absent."""
        text = "The answer is 8"
        reward = self.reward_fn.compute_format_reward(text)
        assert reward == 0.0
    
    def test_extract_reasoning(self):
        """Test reasoning extraction."""
        text = "<think>This is my reasoning</think> Answer: 42"
        reasoning = self.reward_fn.extract_reasoning(text)
        assert "This is my reasoning" in reasoning
    
    def test_correct_answer_reward(self):
        """Test full reward for correct answer with formatting."""
        generated = "<think>5 + 3 = 8</think> The answer is 8"
        ground_truth = "8"
        
        components = self.reward_fn.compute_reward(generated, ground_truth)
        
        assert components.format_reward == 0.1
        assert components.answer_reward == 1.0
        assert components.total_reward > 1.0
    
    def test_wrong_answer_no_format(self):
        """Test reward for wrong answer without formatting."""
        generated = "The answer is 42"
        ground_truth = "8"
        
        components = self.reward_fn.compute_reward(generated, ground_truth)
        
        assert components.format_reward == 0.0
        assert components.answer_reward == 0.0
        assert components.total_reward == 0.0
    
    def test_batch_rewards(self):
        """Test batch reward computation."""
        generated_texts = [
            "<think>5+3=8</think> The answer is 8",
            "The answer is 42",
            "<think>3*3=9</think> The answer is 9",
        ]
        ground_truths = ["8", "8", "9"]
        
        rewards, components = self.reward_fn.compute_batch_rewards(
            generated_texts, ground_truths
        )
        
        assert rewards.shape == (3,)
        assert len(components) == 3
        assert rewards[0] > rewards[1]  # First has format + correct answer
        assert rewards[2] > rewards[1]  # Third has format + correct answer
    
    def test_callable_interface(self):
        """Test that reward function is callable."""
        generated_texts = ["<think>test</think> The answer is 8"]
        ground_truths = ["8"]
        
        rewards = self.reward_fn(generated_texts, ground_truths)
        
        assert isinstance(rewards, torch.Tensor)
        assert rewards.shape == (1,)


class TestRewardFunctionEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.reward_fn = create_reward_function()
    
    def test_empty_text(self):
        """Test with empty generated text."""
        components = self.reward_fn.compute_reward("", "42")
        assert components.total_reward == 0.0
    
    def test_malformed_tags(self):
        """Test with malformed reasoning tags."""
        text = "<think>Reasoning but no closing tag. The answer is 42"
        components = self.reward_fn.compute_reward(text, "42")
        assert components.format_reward == 0.0  # Malformed tags
    
    def test_multiple_answers(self):
        """Test with multiple potential answers."""
        text = "First I got 5, then 10, but the answer is 15"
        components = self.reward_fn.compute_reward(text, "15")
        # Should extract the last number (15)
        assert components.answer_reward == 1.0
    
    def test_very_long_reasoning(self):
        """Test with very long reasoning."""
        long_reasoning = "Step 1. " * 1000
        text = f"<think>{long_reasoning}</think> The answer is 8"
        components = self.reward_fn.compute_reward(text, "8", num_tokens=1000)
        
        # Should still work but may have penalties
        assert components.format_reward > 0
        assert components.answer_reward > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
