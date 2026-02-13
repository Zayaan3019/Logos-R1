"""
GSM8K dataset loader with reasoning trace formatting.
Handles data loading, preprocessing, and batching for training.
"""

from typing import Optional, Dict, List, Any, Iterator
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader, IterableDataset
import torch
import logging
import re
import random

from ..config import DataConfig
from .tokenizer import ReasoningTokenizer

logger = logging.getLogger(__name__)


class GSM8KDataset:
    """
    Wrapper for GSM8K dataset with reasoning trace support.
    """
    
    def __init__(
        self,
        config: DataConfig,
        tokenizer: ReasoningTokenizer,
        split: str = "train",
    ):
        """
        Initialize GSM8K dataset.
        
        Args:
            config: Data configuration
            tokenizer: Reasoning tokenizer instance
            split: Dataset split ('train' or 'test')
        """
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        
        # Load main GSM8K dataset
        logger.info(f"Loading GSM8K dataset: {config.dataset_name}")
        self.dataset = load_dataset(
            config.dataset_name,
            config.dataset_split,
            split=config.train_split if split == "train" else config.test_split,
        )
        
        # Optionally load reasoning traces
        self.reasoning_dataset = None
        if config.use_reasoning_traces and config.reasoning_dataset:
            logger.info(f"Loading reasoning traces: {config.reasoning_dataset}")
            try:
                self.reasoning_dataset = load_dataset(config.reasoning_dataset, split=split)
            except Exception as e:
                logger.warning(f"Could not load reasoning dataset: {e}")
        
        logger.info(f"Loaded {len(self.dataset)} examples for {split} split")
    
    def extract_answer(self, answer_text: str) -> str:
        """
        Extract the final numerical answer from GSM8K format.
        
        GSM8K answers are in format: "#### 42"
        
        Args:
            answer_text: Raw answer string from dataset
        
        Returns:
            Extracted numerical answer
        """
        # GSM8K format: solution text followed by #### answer
        if "####" in answer_text:
            parts = answer_text.split("####")
            answer = parts[-1].strip()
            # Remove any remaining formatting
            answer = re.sub(r'[,$\s]', '', answer)
            return answer
        
        # Fallback: extract last number
        numbers = re.findall(r'-?\d+\.?\d*', answer_text)
        return numbers[-1] if numbers else answer_text.strip()
    
    def extract_reasoning(self, answer_text: str) -> str:
        """
        Extract the reasoning steps from GSM8K solution.
        
        Args:
            answer_text: Raw answer string from dataset
        
        Returns:
            Reasoning steps (before ####)
        """
        if "####" in answer_text:
            reasoning = answer_text.split("####")[0].strip()
            return reasoning
        return answer_text.strip()
    
    def format_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        Format a dataset example into prompt/completion format.
        
        Args:
            example: Raw dataset example
        
        Returns:
            Formatted dictionary with 'question', 'reasoning', 'answer'
        """
        question = example["question"].strip()
        answer_text = example["answer"]
        
        # Extract reasoning and final answer
        reasoning = self.extract_reasoning(answer_text)
        answer = self.extract_answer(answer_text)
        
        return {
            "question": question,
            "reasoning": reasoning,
            "answer": answer,
        }
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single formatted and tokenized example.
        
        Args:
            idx: Example index
        
        Returns:
            Tokenized example ready for training
        """
        # Get raw example
        example = self.dataset[idx]
        
        # Format into reasoning structure
        formatted = self.format_example(example)
        
        # Tokenize for training
        encoded = self.tokenizer.encode_for_training(
            question=formatted["question"],
            reasoning=formatted["reasoning"],
            answer=formatted["answer"],
            max_length=self.config.max_length,
        )
        
        # Add metadata
        encoded["question_text"] = formatted["question"]
        encoded["answer_text"] = formatted["answer"]
        
        return encoded
    
    def get_prompts_only(self, batch_size: int = 1) -> Iterator[List[str]]:
        """
        Generate batches of prompts without answers (for RL sampling).
        
        Args:
            batch_size: Number of prompts per batch
        
        Yields:
            Batches of formatted prompts
        """
        prompts = []
        for example in self.dataset:
            formatted = self.format_example(example)
            prompt = self.tokenizer.format_prompt(formatted["question"])
            prompts.append(prompt)
            
            if len(prompts) == batch_size:
                yield prompts
                prompts = []
        
        # Yield remaining
        if prompts:
            yield prompts


class RLExperienceDataset(IterableDataset):
    """
    Iterable dataset that continuously generates prompts for RL training.
    """
    
    def __init__(
        self,
        base_dataset: GSM8KDataset,
        num_epochs: int = 1,
        shuffle: bool = True,
    ):
        """
        Initialize RL experience dataset.
        
        Args:
            base_dataset: Underlying GSM8K dataset
            num_epochs: Number of times to iterate through data
            shuffle: Whether to shuffle data
        """
        self.base_dataset = base_dataset
        self.num_epochs = num_epochs
        self.shuffle = shuffle
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate through dataset, yielding prompts and ground truth.
        """
        for epoch in range(self.num_epochs):
            indices = list(range(len(self.base_dataset.dataset)))
            
            if self.shuffle:
                random.shuffle(indices)
            
            for idx in indices:
                example = self.base_dataset.dataset[idx]
                formatted = self.base_dataset.format_example(example)
                
                # Return prompt and answer (for reward computation)
                yield {
                    "prompt": self.base_dataset.tokenizer.format_prompt(
                        formatted["question"]
                    ),
                    "question": formatted["question"],
                    "ground_truth": formatted["answer"],
                    "reasoning_reference": formatted["reasoning"],
                }


def create_dataloader(
    config: DataConfig,
    tokenizer: ReasoningTokenizer,
    split: str = "train",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for training.
    
    Args:
        config: Data configuration
        tokenizer: Reasoning tokenizer
        split: Dataset split
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
    
    Returns:
        Configured DataLoader
    """
    dataset = GSM8KDataset(config, tokenizer, split)
    
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function to stack tensors properly."""
        # Stack tensor fields
        collated = {}
        for key in ["input_ids", "attention_mask", "labels"]:
            collated[key] = torch.stack([item[key] for item in batch])
        
        # Keep text fields as lists
        collated["question_text"] = [item["question_text"] for item in batch]
        collated["answer_text"] = [item["answer_text"] for item in batch]
        
        return collated
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def create_rl_dataloader(
    config: DataConfig,
    tokenizer: ReasoningTokenizer,
    batch_size: int = 4,
    num_epochs: int = 1,
) -> DataLoader:
    """
    Create a DataLoader for RL training that yields prompts.
    
    Args:
        config: Data configuration
        tokenizer: Reasoning tokenizer
        batch_size: Batch size
        num_epochs: Number of epochs
    
    Returns:
        DataLoader yielding prompts and ground truth
    """
    base_dataset = GSM8KDataset(config, tokenizer, split="train")
    rl_dataset = RLExperienceDataset(base_dataset, num_epochs=num_epochs, shuffle=True)
    
    return DataLoader(
        rl_dataset,
        batch_size=batch_size,
        num_workers=0,  # Iterable datasets work better with num_workers=0
    )
