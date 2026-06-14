"""
Normal KV Cache Implementation (FP16 Baseline)

Standard KV cache that stores keys and values in FP16 precision.
This is the baseline implementation that all compression methods are compared against.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import time


class NormalKVCache:
    """
    Standard FP16 KV cache implementation.
    
    Stores keys and values in full precision (FP16/FP32) without any compression.
    This is the baseline that compression methods are compared against.
    """
    
    def __init__(self, max_seq_len: int, num_heads: int, head_dim: int, 
                 batch_size: int = 1, dtype=torch.float16, device='cuda'):
        """
        Initialize KV cache.
        
        Args:
            max_seq_len: Maximum sequence length
            num_heads: Number of attention heads
            head_dim: Dimension per head
            batch_size: Batch size
            dtype: Data type (FP16 or FP32)
            device: Device to store cache
        """
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device
        
        # Initialize empty cache
        # Shape: [batch_size, num_heads, seq_len, head_dim]
        self.key_cache = torch.zeros(
            batch_size, num_heads, 0, head_dim, 
            dtype=dtype, device=device
        )
        self.value_cache = torch.zeros(
            batch_size, num_heads, 0, head_dim, 
            dtype=dtype, device=device
        )
        
        self.current_len = 0
        
    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """
        Append new key-value pairs to cache.
        
        Args:
            key: [batch_size, num_heads, 1, head_dim]
            value: [batch_size, num_heads, 1, head_dim]
        """
        self.key_cache = torch.cat([self.key_cache, key], dim=2)
        self.value_cache = torch.cat([self.value_cache, value], dim=2)
        self.current_len += 1
        
    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the full KV cache.
        
        Returns:
            key_cache: [batch_size, num_heads, seq_len, head_dim]
            value_cache: [batch_size, num_heads, seq_len, head_dim]
        """
        return self.key_cache, self.value_cache
    
    def get_memory_usage(self) -> int:
        """
        Calculate memory usage in bytes.
        
        Returns:
            Total memory in bytes
        """
        key_bytes = self.key_cache.numel() * self.key_cache.element_size()
        value_bytes = self.value_cache.numel() * self.value_cache.element_size()
        return key_bytes + value_bytes
    
    def get_compression_ratio(self) -> float:
        """
        Calculate compression ratio compared to FP16 baseline.
        
        Returns:
            Compression ratio (always 1.0 for normal KV cache)
        """
        return 1.0
    
    def reset(self):
        """Reset the cache."""
        self.key_cache = torch.zeros(
            self.batch_size, self.num_heads, 0, self.head_dim,
            dtype=self.dtype, device=self.device
        )
        self.value_cache = torch.zeros(
            self.batch_size, self.num_heads, 0, self.head_dim,
            dtype=self.dtype, device=self.device
        )
        self.current_len = 0


def compute_attention(query: torch.Tensor, key_cache: torch.Tensor, 
                     value_cache: torch.Tensor, 
                     scale: Optional[float] = None) -> torch.Tensor:
    """
    Compute attention using cached keys and values.
    
    Args:
        query: [batch_size, num_heads, 1, head_dim] - current token's query
        key_cache: [batch_size, num_heads, seq_len, head_dim]
        value_cache: [batch_size, num_heads, seq_len, head_dim]
        scale: Attention scale factor (default: 1/sqrt(head_dim))
    
    Returns:
        output: [batch_size, num_heads, 1, head_dim]
    """
    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)
    
    # Compute attention scores
    # [batch, heads, 1, head_dim] @ [batch, heads, head_dim, seq_len]
    # -> [batch, heads, 1, seq_len]
    scores = torch.matmul(query, key_cache.transpose(-2, -1)) * scale
    
    # Softmax over sequence dimension
    attn_weights = F.softmax(scores, dim=-1)
    
    # Weighted sum of values
    # [batch, heads, 1, seq_len] @ [batch, heads, seq_len, head_dim]
    # -> [batch, heads, 1, head_dim]
    output = torch.matmul(attn_weights, value_cache)
    
    return output


def benchmark_normal_kv(model, tokenizer, prompt: str, max_new_tokens: int = 100,
                       device='cuda'):
    """
    Benchmark normal KV cache on a given model and prompt.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        max_new_tokens: Number of tokens to generate
        device: Device to run on
    
    Returns:
        Dictionary with benchmark results
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    
    # Warmup
    for _ in range(3):
        _ = model.generate(input_ids, max_new_tokens=10, use_cache=True)
    
    torch.cuda.synchronize()
    
    # Benchmark generation
    start_time = time.time()
    output = model.generate(
        input_ids, 
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=False
    )
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    # Calculate metrics
    num_tokens = output.shape[1] - input_ids.shape[1]
    tokens_per_second = num_tokens / total_time
    
    # Memory usage (approximate)
    torch.cuda.reset_peak_memory_stats()
    memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    
    return {
        'method': 'Normal KV (FP16)',
        'total_time': total_time,
        'tokens_generated': num_tokens,
        'tokens_per_second': tokens_per_second,
        'memory_gb': memory_allocated,
        'compression_ratio': 1.0
    }


if __name__ == "__main__":
    # Quick test
    cache = NormalKVCache(
        max_seq_len=2048,
        num_heads=12,
        head_dim=64,
        batch_size=1,
        dtype=torch.float16,
        device='cuda'
    )
    
    # Simulate appending tokens
    for i in range(100):
        key = torch.randn(1, 12, 1, 64, dtype=torch.float16, device='cuda')
        value = torch.randn(1, 12, 1, 64, dtype=torch.float16, device='cuda')
        cache.append(key, value)
    
    print(f"Cache length: {cache.current_len}")
    print(f"Memory usage: {cache.get_memory_usage() / (1024**2):.2f} MB")
    print(f"Compression ratio: {cache.get_compression_ratio()}")
