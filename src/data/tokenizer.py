"""
Custom tokenizer setup with special tokens for reasoning.
Handles <think> ... </think> tags and chat templates.
"""

from typing import Optional, Dict, Any
from transformers import AutoTokenizer, PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


class ReasoningTokenizer:
    """
    Wrapper around HuggingFace tokenizer with reasoning-specific modifications.
    """
    
    def __init__(
        self,
        model_name: str,
        think_start_token: str = "<think>",
        think_end_token: str = "</think>",
    ):
        """
        Initialize tokenizer with special reasoning tokens.
        
        Args:
            model_name: HuggingFace model identifier
            think_start_token: Token marking start of reasoning
            think_end_token: Token marking end of reasoning
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        
        # Add special tokens if not present
        special_tokens = {
            "additional_special_tokens": [think_start_token, think_end_token]
        }
        
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        logger.info(f"Added {num_added} special tokens to vocabulary")
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")
        
        # Cache special token IDs
        self.think_start_id = self.tokenizer.convert_tokens_to_ids(think_start_token)
        self.think_end_id = self.tokenizer.convert_tokens_to_ids(think_end_token)
        
        # Setup chat template
        self._setup_chat_template()
    
    def _setup_chat_template(self):
        """Configure chat template for reasoning format."""
        # Custom template that includes reasoning tags
        # Format: <|user|>Question<|assistant|><think>reasoning</think>answer
        
        if self.tokenizer.chat_template is None:
            # Fallback template if model doesn't have one
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                "{{ '<|user|>\n' + message['content'] + '\n' }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ '<|assistant|>\n' + message['content'] + '\n' }}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|assistant|>\n' }}"
                "{% endif %}"
            )
            logger.info("Set custom chat template")
    
    def format_prompt(self, question: str, include_reasoning_prompt: bool = True) -> str:
        """
        Format a math problem as a prompt.
        
        Args:
            question: The math problem to solve
            include_reasoning_prompt: Whether to encourage step-by-step reasoning
        
        Returns:
            Formatted prompt string
        """
        if include_reasoning_prompt:
            prompt = (
                f"Solve the following math problem step by step. "
                f"Show your work inside {self.think_start_token} tags, "
                f"then provide the final answer.\n\n"
                f"Problem: {question}\n\n"
                f"Solution:"
            )
        else:
            prompt = f"Problem: {question}\n\nSolution:"
        
        return prompt
    
    def format_training_example(
        self,
        question: str,
        reasoning: str,
        answer: str,
    ) -> str:
        """
        Format a complete training example with reasoning tags.
        
        Args:
            question: The math problem
            reasoning: The step-by-step reasoning
            answer: The final answer
        
        Returns:
            Formatted training string
        """
        prompt = self.format_prompt(question)
        
        # Format with reasoning tags
        completion = (
            f"{self.think_start_token}\n"
            f"{reasoning}\n"
            f"{self.think_end_token}\n"
            f"The answer is {answer}"
        )
        
        return prompt + " " + completion
    
    def encode_for_training(
        self,
        question: str,
        reasoning: str,
        answer: str,
        max_length: int = 2048,
    ) -> Dict[str, Any]:
        """
        Encode a training example with proper attention mask.
        
        Args:
            question: The math problem
            reasoning: The step-by-step reasoning
            answer: The final answer
            max_length: Maximum sequence length
        
        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        # Format the full text
        full_text = self.format_training_example(question, reasoning, answer)
        
        # Tokenize
        encoded = self.tokenizer(
            full_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Create labels (for causal LM, labels = input_ids)
        encoded["labels"] = encoded["input_ids"].clone()
        
        # Optionally mask the prompt (only train on completion)
        # This is commented out but can be enabled for instruction-following
        # prompt_text = self.format_prompt(question)
        # prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        # encoded["labels"][:, :len(prompt_ids)] = -100
        
        return {k: v.squeeze(0) for k, v in encoded.items()}
    
    def decode_with_reasoning_sep(self, token_ids, skip_special_tokens: bool = False) -> Dict[str, str]:
        """
        Decode tokens and separate reasoning from answer.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output
        
        Returns:
            Dictionary with 'reasoning' and 'answer' keys
        """
        # Decode full text
        full_text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        
        # Extract reasoning and answer
        reasoning = ""
        answer = ""
        
        if self.think_start_token in full_text and self.think_end_token in full_text:
            # Split by reasoning tags
            parts = full_text.split(self.think_start_token, 1)
            if len(parts) > 1:
                reasoning_and_rest = parts[1].split(self.think_end_token, 1)
                reasoning = reasoning_and_rest[0].strip()
                if len(reasoning_and_rest) > 1:
                    answer = reasoning_and_rest[1].strip()
        else:
            # No reasoning tags found, treat as answer only
            answer = full_text.strip()
        
        return {
            "reasoning": reasoning,
            "answer": answer,
            "full_text": full_text,
        }
    
    def __getattr__(self, name):
        """Delegate unknown attributes to underlying tokenizer."""
        return getattr(self.tokenizer, name)
    
    def __len__(self):
        """Return vocabulary size."""
        return len(self.tokenizer)


def create_tokenizer(
    model_name: str,
    think_start_token: str = "<think>",
    think_end_token: str = "</think>",
) -> ReasoningTokenizer:
    """
    Factory function to create a reasoning tokenizer.
    
    Args:
        model_name: HuggingFace model identifier
        think_start_token: Start token for reasoning
        think_end_token: End token for reasoning
    
    Returns:
        Configured ReasoningTokenizer instance
    """
    return ReasoningTokenizer(
        model_name=model_name,
        think_start_token=think_start_token,
        think_end_token=think_end_token,
    )
