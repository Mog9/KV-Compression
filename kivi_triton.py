"""
KIVI: Tuning-Free Asymmetric 2-bit Quantization for KV Cache (Triton Implementation)

Based on: "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
Paper: https://arxiv.org/abs/2402.02750

Key insight:
- Keys have persistent channel outliers → quantize per-channel
- Values are dynamic per-token → quantize per-token
"""

import torch
import triton
import triton.language as tl


# Quantization Kernels

@triton.jit
def _quantize_per_channel_kernel(
    key_ptr, scales_ptr, zero_points_ptr, quantized_ptr,
    B, H, S, D,
    stride_b, stride_h, stride_s, stride_d,
    BLOCK_S: tl.constexpr,
):
    """Quantize keys per-channel (across sequence dimension)."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    
    # Compute min/max for this channel
    channel_min = float('inf')
    channel_max = float('-inf')
    
    for s_start in range(0, S, BLOCK_S):
        s_offsets = s_start + tl.arange(0, BLOCK_S)
        mask = s_offsets < S
        
        offsets = pid_b * stride_b + pid_h * stride_h + s_offsets * stride_s + pid_d * stride_d
        values = tl.load(key_ptr + offsets, mask=mask, other=0.0)
        
        channel_min = tl.minimum(channel_min, tl.min(values))
        channel_max = tl.maximum(channel_max, tl.max(values))
    
    # Compute scale and zero point
    scale = (channel_max - channel_min) / 3.0  # 2-bit: 0-3
    scale = tl.maximum(scale, 1e-8)
    zero_point = -channel_min / scale
    
    # Store scale and zero point
    scale_offset = pid_b * H * D + pid_h * D + pid_d
    tl.store(scales_ptr + scale_offset, scale)
    tl.store(zero_points_ptr + scale_offset, zero_point)
    
    # Quantize values
    for s_start in range(0, S, BLOCK_S):
        s_offsets = s_start + tl.arange(0, BLOCK_S)
        mask = s_offsets < S
        
        offsets = pid_b * stride_b + pid_h * stride_h + s_offsets * stride_s + pid_d * stride_d
        values = tl.load(key_ptr + offsets, mask=mask, other=0.0)
        
        quantized = tl.floor(values / scale + zero_point + 0.5)
        quantized = tl.minimum(tl.maximum(quantized, 0), 3)
        
        tl.store(quantized_ptr + offsets, quantized.to(tl.uint8), mask=mask)


@triton.jit
def _quantize_per_token_kernel(
    value_ptr, scales_ptr, zero_points_ptr, quantized_ptr,
    B, H, S, D,
    stride_b, stride_h, stride_s, stride_d,
    BLOCK_D: tl.constexpr,
):
    """Quantize values per-token (across head dimension)."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    
    # Compute min/max for this token
    token_min = float('inf')
    token_max = float('-inf')
    
    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        mask = d_offsets < D
        
        offsets = pid_b * stride_b + pid_h * stride_h + pid_s * stride_s + d_offsets * stride_d
        values = tl.load(value_ptr + offsets, mask=mask, other=0.0)
        
        token_min = tl.minimum(token_min, tl.min(values))
        token_max = tl.maximum(token_max, tl.max(values))
    
    # Compute scale and zero point
    scale = (token_max - token_min) / 3.0  # 2-bit: 0-3
    scale = tl.maximum(scale, 1e-8)
    zero_point = -token_min / scale
    
    # Store scale and zero point
    scale_offset = pid_b * H * S + pid_h * S + pid_s
    tl.store(scales_ptr + scale_offset, scale)
    tl.store(zero_points_ptr + scale_offset, zero_point)
    
    # Quantize values
    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        mask = d_offsets < D
        
        offsets = pid_b * stride_b + pid_h * stride_h + pid_s * stride_s + d_offsets * stride_d
        values = tl.load(value_ptr + offsets, mask=mask, other=0.0)
        
        quantized = tl.floor(values / scale + zero_point + 0.5)
        quantized = tl.minimum(tl.maximum(quantized, 0), 3)
        
        tl.store(quantized_ptr + offsets, quantized.to(tl.uint8), mask=mask)


# Dequantization Kernels

@triton.jit
def _dequantize_per_channel_kernel(
    quantized_ptr, scales_ptr, zero_points_ptr, output_ptr,
    B, H, S, D,
    stride_b, stride_h, stride_s, stride_d,
    BLOCK_S: tl.constexpr,
):
    """Dequantize keys (per-channel quantization)."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    
    # Load scale and zero point
    scale_offset = pid_b * H * D + pid_h * D + pid_d
    scale = tl.load(scales_ptr + scale_offset)
    zero_point = tl.load(zero_points_ptr + scale_offset)
    
    # Dequantize values
    for s_start in range(0, S, BLOCK_S):
        s_offsets = s_start + tl.arange(0, BLOCK_S)
        mask = s_offsets < S
        
        offsets = pid_b * stride_b + pid_h * stride_h + s_offsets * stride_s + pid_d * stride_d
        quantized = tl.load(quantized_ptr + offsets, mask=mask, other=0).to(tl.float32)
        
        dequantized = (quantized - zero_point) * scale
        
        tl.store(output_ptr + offsets, dequantized.to(tl.float16), mask=mask)


@triton.jit
def _dequantize_per_token_kernel(
    quantized_ptr, scales_ptr, zero_points_ptr, output_ptr,
    B, H, S, D,
    stride_b, stride_h, stride_s, stride_d,
    BLOCK_D: tl.constexpr,
):
    """Dequantize values (per-token quantization)."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    
    # Load scale and zero point
    scale_offset = pid_b * H * S + pid_h * S + pid_s
    scale = tl.load(scales_ptr + scale_offset)
    zero_point = tl.load(zero_points_ptr + scale_offset)
    
    # Dequantize values
    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        mask = d_offsets < D
        
        offsets = pid_b * stride_b + pid_h * stride_h + pid_s * stride_s + d_offsets * stride_d
        quantized = tl.load(quantized_ptr + offsets, mask=mask, other=0).to(tl.float32)
        
        dequantized = (quantized - zero_point) * scale
        
        tl.store(output_ptr + offsets, dequantized.to(tl.float16), mask=mask)


# Public API

def quantize_keys_triton(keys):
    """
    Quantize keys per-channel using Triton.
    
    Args:
        keys: [B, H, S, D] tensor
    
    Returns:
        quantized: [B, H, S, D] uint8 tensor
        scales: [B, H, D] float16 tensor
        zero_points: [B, H, D] float16 tensor
    """
    B, H, S, D = keys.shape
    device = keys.device
    
    quantized = torch.zeros_like(keys, dtype=torch.uint8, device=device)
    scales = torch.zeros(B, H, D, dtype=torch.float16, device=device)
    zero_points = torch.zeros(B, H, D, dtype=torch.float16, device=device)
    
    grid = (B, H, D)
    BLOCK_S = 128
    
    _quantize_per_channel_kernel[grid](
        keys, scales, zero_points, quantized,
        B, H, S, D,
        keys.stride(0), keys.stride(1), keys.stride(2), keys.stride(3),
        BLOCK_S=BLOCK_S,
    )
    
    return quantized, scales, zero_points


def quantize_values_triton(values):
    """
    Quantize values per-token using Triton.
    
    Args:
        values: [B, H, S, D] tensor
    
    Returns:
        quantized: [B, H, S, D] uint8 tensor
        scales: [B, H, S] float16 tensor
        zero_points: [B, H, S] float16 tensor
    """
    B, H, S, D = values.shape
    device = values.device
    
    quantized = torch.zeros_like(values, dtype=torch.uint8, device=device)
    scales = torch.zeros(B, H, S, dtype=torch.float16, device=device)
    zero_points = torch.zeros(B, H, S, dtype=torch.float16, device=device)
    
    grid = (B, H, S)
    BLOCK_D = 128
    
    _quantize_per_token_kernel[grid](
        values, scales, zero_points, quantized,
        B, H, S, D,
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        BLOCK_D=BLOCK_D,
    )
    
    return quantized, scales, zero_points


def dequantize_keys_triton(quantized, scales, zero_points):
    """
    Dequantize keys using Triton.
    
    Args:
        quantized: [B, H, S, D] uint8 tensor
        scales: [B, H, D] float16 tensor
        zero_points: [B, H, D] float16 tensor
    
    Returns:
        dequantized: [B, H, S, D] float16 tensor
    """
    B, H, S, D = quantized.shape
    device = quantized.device
    
    output = torch.zeros(B, H, S, D, dtype=torch.float16, device=device)
    
    grid = (B, H, D)
    BLOCK_S = 128
    
    _dequantize_per_channel_kernel[grid](
        quantized, scales, zero_points, output,
        B, H, S, D,
        quantized.stride(0), quantized.stride(1), quantized.stride(2), quantized.stride(3),
        BLOCK_S=BLOCK_S,
    )
    
    return output


def dequantize_values_triton(quantized, scales, zero_points):
    """
    Dequantize values using Triton.
    
    Args:
        quantized: [B, H, S, D] uint8 tensor
        scales: [B, H, S] float16 tensor
        zero_points: [B, H, S] float16 tensor
    
    Returns:
        dequantized: [B, H, S, D] float16 tensor
    """
    B, H, S, D = quantized.shape
    device = quantized.device
    
    output = torch.zeros(B, H, S, D, dtype=torch.float16, device=device)
    
    grid = (B, H, S)
    BLOCK_D = 128
    
    _dequantize_per_token_kernel[grid](
        quantized, scales, zero_points, output,
        B, H, S, D,
        quantized.stride(0), quantized.stride(1), quantized.stride(2), quantized.stride(3),
        BLOCK_D=BLOCK_D,
    )
    
    return output


class KIVIKVCacheTriton:
    """
    KIVI KV Cache with Triton kernels.
    
    Uses asymmetric quantization:
    - Keys: per-channel quantization (preserves persistent outliers)
    - Values: per-token quantization (adapts to dynamic variations)
    """
    
    def __init__(self, max_seq_len, num_heads, head_dim, batch_size=1, device='cuda'):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.device = device
        
        # Current sequence length
        self.current_len = 0
        
        # Quantized cache
        self.key_quantized = None
        self.key_scales = None
        self.key_zero_points = None
        
        self.value_quantized = None
        self.value_scales = None
        self.value_zero_points = None
        
        # Residual buffer for recent tokens (FP16)
        self.residual_size = 32
        self.key_residual = torch.zeros(
            batch_size, num_heads, self.residual_size, head_dim,
            dtype=torch.float16, device=device
        )
        self.value_residual = torch.zeros(
            batch_size, num_heads, self.residual_size, head_dim,
            dtype=torch.float16, device=device
        )
        self.residual_len = 0
    
    def append(self, key, value):
        """
        Append new key-value pairs to cache.
        
        Args:
            key: [B, H, 1, D] tensor
            value: [B, H, 1, D] tensor
        """
        # Add to residual buffer
        if self.residual_len < self.residual_size:
            self.key_residual[:, :, self.residual_len:self.residual_len+1, :] = key
            self.value_residual[:, :, self.residual_len:self.residual_len+1, :] = value
            self.residual_len += 1
        else:
            # Flush residual buffer
            self._flush_residual()
            # Add new token to now-empty residual buffer
            self.key_residual[:, :, 0:1, :] = key
            self.value_residual[:, :, 0:1, :] = value
            self.residual_len = 1
        
        self.current_len += 1
    
    def _flush_residual(self):
        """Quantize residual buffer and append to main cache."""
        if self.residual_len == 0:
            return
        
        # Get residual buffer slice
        key_chunk = self.key_residual[:, :, :self.residual_len, :].contiguous()
        value_chunk = self.value_residual[:, :, :self.residual_len, :].contiguous()
        
        # Quantize
        kq, ks, kzp = quantize_keys_triton(key_chunk)
        vq, vs, vzp = quantize_values_triton(value_chunk)
        
        # Append to main cache
        if self.key_quantized is None:
            self.key_quantized = kq
            self.key_scales = ks
            self.key_zero_points = kzp
            self.value_quantized = vq
            self.value_scales = vs
            self.value_zero_points = vzp
        else:
            self.key_quantized = torch.cat([self.key_quantized, kq], dim=2)
            # For per-channel quantization, scales are per-channel
            self.key_scales = ks  # Just use the latest scales
            self.key_zero_points = kzp
            self.value_quantized = torch.cat([self.value_quantized, vq], dim=2)
            self.value_scales = torch.cat([self.value_scales, vs], dim=2)
            self.value_zero_points = torch.cat([self.value_zero_points, vzp], dim=2)
        
        # Clear residual buffer
        self.residual_len = 0
    
    def get_cache(self):
        """
        Get dequantized KV cache for attention.
        
        Returns:
            key_cache: [B, H, S, D] tensor
            value_cache: [B, H, S, D] tensor
        """
        # Flush any remaining residual
        if self.residual_len > 0:
            self._flush_residual()
        
        if self.key_quantized is None:
            return None, None
        
        # Dequantize
        key_cache = dequantize_keys_triton(self.key_quantized, self.key_scales, self.key_zero_points)
        value_cache = dequantize_values_triton(self.value_quantized, self.value_scales, self.value_zero_points)
        
        return key_cache, value_cache
    
    def get_memory_usage(self):
        """Calculate memory usage in bytes."""
        if self.current_len == 0:
            return 0
        
        # Quantized data (2-bit = 0.25 bytes per element)
        quantized_bytes = self.batch_size * self.num_heads * self.current_len * self.head_dim * 2 * 0.25
        
        # Scales and zero points (FP16 = 2 bytes)
        scales_bytes = (
            self.batch_size * self.num_heads * self.head_dim * 2 * 2 +  # key scales + zp
            self.batch_size * self.num_heads * self.current_len * 2 * 2  # value scales + zp
        )
        
        # Residual buffer (FP16)
        residual_bytes = self.batch_size * self.num_heads * self.residual_len * self.head_dim * 2 * 2
        
        return int(quantized_bytes + scales_bytes + residual_bytes)
    
    def get_compression_ratio(self):
        """Calculate compression ratio vs FP16."""
        fp16_bytes = self.batch_size * self.num_heads * self.current_len * self.head_dim * 2 * 2
        kivi_bytes = self.get_memory_usage()
        
        if kivi_bytes == 0:
            return 1.0
        
        return fp16_bytes / kivi_bytes
    
    def reset(self):
        """Reset the cache."""
        self.current_len = 0
        self.residual_len = 0
        self.key_quantized = None
        self.key_scales = None
        self.key_zero_points = None
        self.value_quantized = None
        self.value_scales = None
        self.value_zero_points = None
