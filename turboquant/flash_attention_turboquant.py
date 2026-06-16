"""
Flash Attention implementation for TurboQuant quantized KV cache.
Uses tiled attention with online softmax for efficient computation.
"""
import torch
import triton
import triton.language as tl
import sys
from pathlib import Path

# Add turboquant directory to path
sys.path.insert(0, str(Path(__file__).parent))

from turboquant_triton import dequantize_keys_turboquant, dequantize_values_turboquant


@triton.jit
def _flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    Q_LEN, KV_LEN,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr,
):
    """
    Flash Attention kernel with online softmax.
    Computes attention in tiles to avoid materializing full attention matrix.
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    # Compute batch and head indices
    pid_b = pid_bh // 12  # Assuming 12 heads
    pid_h = pid_bh % 12
    
    # Query block offsets
    q_start = pid_m * BLOCK_M
    q_offsets = q_start + tl.arange(0, BLOCK_M)
    q_mask = q_offsets < Q_LEN
    
    # Load query block
    q_ptrs = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + q_offsets[:, None] * stride_qm + tl.arange(0, HEAD_DIM)[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    
    # Online softmax accumulators
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)  # max
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)  # sum of exp
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)  # output accumulator
    
    # Loop over KV blocks
    for kv_start in range(0, KV_LEN, BLOCK_N):
        kv_offsets = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_offsets < KV_LEN
        
        # Load key block
        k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + kv_offsets[:, None] * stride_kn + tl.arange(0, HEAD_DIM)[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
        
        # Compute attention scores: Q @ K^T
        scores = tl.dot(q, tl.trans(k)) * SCALE
        
        # Apply mask for invalid positions
        scores = tl.where(q_mask[:, None] & kv_mask[None, :], scores, float('-inf'))
        
        # Online softmax: update max
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        
        # Compute exp(scores - m_new)
        exp_scores = tl.exp(scores - m_new[:, None])
        
        # Update sum
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(exp_scores, axis=1)
        
        # Rescale previous accumulator
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        
        # Load value block
        v_ptrs = V_ptr + pid_b * stride_vb + pid_h * stride_vh + kv_offsets[:, None] * stride_vn + tl.arange(0, HEAD_DIM)[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)
        
        # Update accumulator: acc += exp_scores @ V
        acc += tl.dot(exp_scores.to(tl.float16), v)
        
        # Update max and sum
        m_i = m_new
        l_i = l_new
    
    # Normalize by sum
    output = acc / l_i[:, None]
    
    # Store output
    o_ptrs = O_ptr + pid_b * stride_ob + pid_h * stride_oh + q_offsets[:, None] * stride_om + tl.arange(0, HEAD_DIM)[None, :] * stride_od
    tl.store(o_ptrs, output.to(tl.float16), mask=q_mask[:, None])


def flash_attention_turboquant(query, turboquant_cache):
    """
    Compute Flash Attention with TurboQuant quantized KV cache.
    
    Args:
        query: [B, H, Q_LEN, D] query tensor
        turboquant_cache: TurboQuantKVCache instance
    
    Returns:
        output: [B, H, Q_LEN, D] attention output
    """
    B, H, Q_LEN, D = query.shape
    
    # Get dequantized KV cache
    key_cache, value_cache = turboquant_cache.get_cache()
    KV_LEN = key_cache.shape[2]
    
    # Allocate output
    output = torch.zeros_like(query)
    
    # Compute scale
    scale = 1.0 / (D ** 0.5)
    
    # Launch kernel
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid = (triton.cdiv(Q_LEN, BLOCK_M), B * H)
    
    _flash_attention_kernel[grid](
        query, key_cache, value_cache, output,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        Q_LEN, KV_LEN,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        SCALE=scale,
    )
    
    return output
