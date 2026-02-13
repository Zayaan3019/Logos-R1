"""
StreamingLLM KV Cache Management for Long Reasoning Chains.

Implements efficient KV cache eviction to handle 4k+ token reasoning chains
while maintaining attention coherence.

Key Idea: Keep "attention sink" (first few tokens) + sliding window (recent tokens),
evict the middle to save VRAM.
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class StreamingKVCache:
    """
    Streaming KV cache with attention sink and sliding window.
    
    Based on "Efficient Streaming Language Models with Attention Sinks"
    (Xiao et al., 2023)
    
    Structure:
    [Attention Sink: 4 tokens] [.... evicted ....] [Sliding Window: 1024 tokens]
    """
    
    def __init__(
        self,
        attention_sink_size: int = 4,
        sliding_window_size: int = 1024,
        eviction_batch_size: int = 128,
    ):
        """
        Initialize streaming KV cache.
        
        Args:
            attention_sink_size: Number of initial tokens to keep (attention sink)
            sliding_window_size: Number of recent tokens to keep
            eviction_batch_size: Number of tokens to evict at once
        """
        self.attention_sink_size = attention_sink_size
        self.sliding_window_size = sliding_window_size
        self.eviction_batch_size = eviction_batch_size
        
        # Cache storage: layer_idx -> (key_cache, value_cache)
        self.key_cache: Dict[int, torch.Tensor] = {}
        self.value_cache: Dict[int, torch.Tensor] = {}
        
        self.current_length = 0
        self.total_generated = 0
        
        logger.info(
            f"Initialized StreamingKVCache: "
            f"sink={attention_sink_size}, window={sliding_window_size}"
        )
    
    @property
    def max_cache_length(self) -> int:
        """Maximum cache length before eviction."""
        return self.attention_sink_size + self.sliding_window_size
    
    def should_evict(self) -> bool:
        """
        Check if cache should be evicted.
        
        Returns:
            True if cache is full
        """
        return self.current_length >= self.max_cache_length
    
    def evict(self, layer_idx: int):
        """
        Evict middle tokens from cache for a specific layer.
        
        Keeps: [sink tokens] + [recent window tokens]
        Removes: middle tokens
        
        Args:
            layer_idx: Layer index to evict from
        """
        if layer_idx not in self.key_cache:
            return
        
        key = self.key_cache[layer_idx]
        value = self.value_cache[layer_idx]
        
        # Split into attention sink, middle (to evict), and window
        # Shape: [batch, num_heads, seq_len, head_dim]
        
        sink_keys = key[:, :, :self.attention_sink_size, :]
        sink_values = value[:, :, :self.attention_sink_size, :]
        
        window_start = self.current_length - self.sliding_window_size
        window_keys = key[:, :, window_start:, :]
        window_values = value[:, :, window_start:, :]
        
        # Concatenate sink + window
        new_key = torch.cat([sink_keys, window_keys], dim=2)
        new_value = torch.cat([sink_values, window_values], dim=2)
        
        self.key_cache[layer_idx] = new_key
        self.value_cache[layer_idx] = new_value
        
        # Update length
        self.current_length = new_key.shape[2]
    
    def update(
        self,
        layer_idx: int,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
    ):
        """
        Update cache with new key/value tensors.
        
        Args:
            layer_idx: Layer index
            new_key: New key tensor [batch, num_heads, new_seq_len, head_dim]
            new_value: New value tensor [batch, num_heads, new_seq_len, head_dim]
        """
        if layer_idx not in self.key_cache:
            # Initialize cache for this layer
            self.key_cache[layer_idx] = new_key
            self.value_cache[layer_idx] = new_value
            self.current_length = new_key.shape[2]
        else:
            # Append new tokens
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], new_key], dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], new_value], dim=2
            )
            self.current_length = self.key_cache[layer_idx].shape[2]
        
        # Check if eviction needed
        if self.should_evict():
            self.evict(layer_idx)
        
        self.total_generated += new_key.shape[2]
    
    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get cached key/value for a layer.
        
        Args:
            layer_idx: Layer index
        
        Returns:
            Tuple of (key_cache, value_cache) or None
        """
        if layer_idx not in self.key_cache:
            return None
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def clear(self):
        """Clear all caches."""
        self.key_cache.clear()
        self.value_cache.clear()
        self.current_length = 0
        self.total_generated = 0
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "current_length": self.current_length,
            "total_generated": self.total_generated,
            "num_layers": len(self.key_cache),
            "max_length": self.max_cache_length,
        }


class CachedGenerator:
    """
    Generator wrapper that uses StreamingKVCache for long sequences.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        attention_sink_size: int = 4,
        sliding_window_size: int = 1024,
        max_new_tokens: int = 2048,
    ):
        """
        Initialize cached generator.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            attention_sink_size: Size of attention sink
            sliding_window_size: Size of sliding window
            max_new_tokens: Maximum tokens to generate
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        
        self.kv_cache = StreamingKVCache(
            attention_sink_size=attention_sink_size,
            sliding_window_size=sliding_window_size,
        )
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        **kwargs,
    ) -> str:
        """
        Generate text with streaming KV cache.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        # Note: Full StreamingLLM integration requires modifying the model's
        # forward pass to use the custom cache. This is a simplified version.
        
        # For production, use the model's native generate with custom cache hooks
        # or integrate with frameworks that support custom caching (e.g., vLLM)
        
        self.kv_cache.clear()
        
        # Standard generation (simplified)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            use_cache=True,  # Use native KV caching
            **kwargs,
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        return generated_text


# Note: Full StreamingLLM implementation requires deeper integration with
# the transformer's attention mechanism. For production use, consider:
# 1. Using vLLM which has built-in PagedAttention
# 2. Modifying the model's attention layers to use custom cache management
# 3. Using frameworks like FlashAttention with custom masking

logger.info(
    "StreamingKVCache implemented. For production, integrate with vLLM or "
    "custom attention implementations for full streaming support."
)
