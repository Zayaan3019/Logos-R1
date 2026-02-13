"""
Policy Model (Actor) with LoRA for parameter-efficient training.
Wraps base LLM with PEFT adapters for reasoning fine-tuning.
"""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedModel,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training,
)
import logging

from ..config import ModelConfig

logger = logging.getLogger(__name__)


class ReasoningPolicy(nn.Module):
    """
    Policy model for reasoning generation with LoRA adapters.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        tokenizer_len: int,
    ):
        """
        Initialize policy model.
        
        Args:
            config: Model configuration
            tokenizer_len: Vocabulary size (for resizing embeddings)
        """
        super().__init__()
        self.config = config
        
        # Load base model
        logger.info(f"Loading base model: {config.model_name}")
        self.base_model = self._load_base_model()
        
        # Resize embeddings for special tokens
        self.base_model.resize_token_embeddings(tokenizer_len)
        logger.info(f"Resized token embeddings to {tokenizer_len}")
        
        # Apply LoRA if configured
        if config.use_lora:
            self.base_model = self._apply_lora()
            logger.info("Applied LoRA adapters")
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
    
    def _load_base_model(self) -> PreTrainedModel:
        """
        Load base language model with optimizations.
        
        Returns:
            Loaded base model
        """
        model_config = AutoConfig.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        
        # Configure flash attention if available
        if self.config.use_flash_attention:
            model_config._attn_implementation = "flash_attention_2"
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            config=model_config,
            torch_dtype=self.config.get_torch_dtype(),
            device_map=self.config.device_map,
            trust_remote_code=True,
        )
        
        return model
    
    def _apply_lora(self) -> PeftModel:
        """
        Apply LoRA adapters to base model.
        
        Returns:
            Model with LoRA adapters
        """
        # Prepare model for k-bit training if using quantization
        # (Remove this if not using quantization)
        # model = prepare_model_for_kbit_training(self.base_model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            inference_mode=False,
        )
        
        # Apply LoRA
        model = get_peft_model(self.base_model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return model
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for language modeling [batch_size, seq_len]
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with loss, logits, etc.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states if outputs.hidden_states else None,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text using the policy model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to sample or use greedy decoding
            num_return_sequences: Number of sequences to generate per input
            **kwargs: Additional generation parameters
        
        Returns:
            Generated token IDs [batch_size * num_return, seq_len + new_tokens]
        """
        outputs = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.base_model.config.pad_token_id,
            eos_token_id=self.base_model.config.eos_token_id,
            **kwargs,
        )
        
        return outputs
    
    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get log probabilities for given sequences (for PPO).
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Log probabilities [batch_size, seq_len, vocab_size]
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        return log_probs
    
    def compute_sequence_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log probability of entire sequence.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Sequence log probabilities [batch_size]
        """
        log_probs = self.get_log_probs(input_ids, attention_mask)
        
        # Get log probs of actual tokens (shift by 1)
        # log_probs: [batch, seq_len, vocab]
        # input_ids: [batch, seq_len]
        batch_size, seq_len = input_ids.shape
        
        # Gather log probs of actual next tokens
        token_log_probs = torch.gather(
            log_probs[:, :-1, :],  # Exclude last position
            dim=2,
            index=input_ids[:, 1:].unsqueeze(-1),  # Shift targets
        ).squeeze(-1)  # [batch, seq_len-1]
        
        # Mask out padding
        if attention_mask is not None:
            mask = attention_mask[:, 1:].float()
            token_log_probs = token_log_probs * mask
            seq_log_probs = token_log_probs.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            seq_log_probs = token_log_probs.mean(dim=1)
        
        return seq_log_probs
    
    def save_pretrained(self, save_directory: str):
        """
        Save model to directory.
        
        Args:
            save_directory: Path to save directory
        """
        if isinstance(self.base_model, PeftModel):
            # Save LoRA adapters
            self.base_model.save_pretrained(save_directory)
        else:
            # Save full model
            self.base_model.save_pretrained(save_directory)
        
        logger.info(f"Saved model to {save_directory}")
    
    def load_pretrained(self, load_directory: str):
        """
        Load model from directory.
        
        Args:
            load_directory: Path to load directory
        """
        if isinstance(self.base_model, PeftModel):
            # Load LoRA adapters
            self.base_model = PeftModel.from_pretrained(
                self.base_model.base_model,
                load_directory,
            )
        else:
            # Load full model
            self.base_model = AutoModelForCausalLM.from_pretrained(load_directory)
        
        logger.info(f"Loaded model from {load_directory}")


class ReferencePolicy(nn.Module):
    """
    Frozen reference policy for KL penalty computation.
    """
    
    def __init__(self, policy: ReasoningPolicy):
        """
        Initialize reference policy from trained policy.
        
        Args:
            policy: Policy to copy and freeze
        """
        super().__init__()
        
        # Deep copy the base model
        import copy
        self.base_model = copy.deepcopy(policy.base_model)
        
        # Freeze all parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        self.base_model.eval()
        logger.info("Created frozen reference policy")
    
    @torch.no_grad()
    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get log probabilities from reference policy.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Log probabilities
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        return log_probs


def create_policy(
    config: ModelConfig,
    tokenizer_len: int,
) -> ReasoningPolicy:
    """
    Factory function to create policy model.
    
    Args:
        config: Model configuration
        tokenizer_len: Vocabulary size
    
    Returns:
        Initialized ReasoningPolicy
    """
    return ReasoningPolicy(config, tokenizer_len)


def create_reference_policy(policy: ReasoningPolicy) -> ReferencePolicy:
    """
    Create frozen reference policy from current policy.
    
    Args:
        policy: Current policy
    
    Returns:
        Frozen reference policy
    """
    return ReferencePolicy(policy)
