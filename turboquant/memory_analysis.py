"""
Detailed Memory Analysis: Qwen3.5 9B FP16 vs TurboQuant
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def analyze_memory_usage():
    """Analyze detailed memory usage for Qwen3.5 9B."""
    
    print("=" * 80)
    print("DETAILED MEMORY ANALYSIS: Qwen3.5 9B")
    print("=" * 80)
    
    # Load model config
    print("\nLoading model config...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-9B",
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Model architecture details
    total_layers = 32
    full_attn_layers = 8  # Layers with traditional KV cache
    linear_attn_layers = 24  # Layers with linear attention (no KV cache)
    
    # Full attention layer config
    num_kv_heads = 4  # Grouped Query Attention
    head_dim = 256
    
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    print(f"Total layers: {total_layers}")
    print(f"  - Full attention layers: {full_attn_layers} (have KV cache)")
    print(f"  - Linear attention layers: {linear_attn_layers} (no KV cache)")
    print(f"\nFull attention layer config:")
    print(f"  - num_kv_heads: {num_kv_heads}")
    print(f"  - head_dim: {head_dim}")
    print(f"  - KV cache per position: {num_kv_heads * head_dim * 2 * 2} bytes (K + V, FP16)")
    
    # Calculate memory for different sequence lengths
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    
    print("\n" + "=" * 80)
    print("MEMORY USAGE BY SEQUENCE LENGTH")
    print("=" * 80)
    print(f"\n{'Seq Len':<10} {'FP16 (MB)':<12} {'TurboQuant (MB)':<16} {'Compression':<12} {'Memory Saved':<12}")
    print("-" * 80)
    
    results = []
    
    for seq_len in seq_lengths:
        # FP16 memory calculation
        # Per layer: 2 (K+V) * num_kv_heads * seq_len * head_dim * 2 bytes (FP16)
        fp16_per_layer = 2 * num_kv_heads * seq_len * head_dim * 2
        fp16_total = fp16_per_layer * full_attn_layers
        fp16_mb = fp16_total / (1024 ** 2)
        
        # TurboQuant memory calculation (2-bit)
        # Quantized data: 2 (K+V) * num_kv_heads * seq_len * head_dim * 0.25 bytes (2-bit)
        quantized_per_layer = 2 * num_kv_heads * seq_len * head_dim * 0.25
        # Scale factors: 2 (K+V) * num_kv_heads * seq_len * 2 bytes (FP16 scales)
        scales_per_layer = 2 * num_kv_heads * seq_len * 2
        # Residual buffer: 2 (K+V) * num_kv_heads * 32 * head_dim * 2 bytes (FP16)
        residual_per_layer = 2 * num_kv_heads * 32 * head_dim * 2
        
        tq_per_layer = quantized_per_layer + scales_per_layer + residual_per_layer
        tq_total = tq_per_layer * full_attn_layers
        tq_mb = tq_total / (1024 ** 2)
        
        # Compression metrics
        compression_ratio = fp16_total / tq_total
        memory_saved_pct = (1 - tq_total / fp16_total) * 100
        memory_saved_mb = fp16_mb - tq_mb
        
        print(f"{seq_len:<10} {fp16_mb:<12.2f} {tq_mb:<16.2f} {compression_ratio:<12.2f}x {memory_saved_pct:<11.1f}%")
        
        results.append({
            "seq_len": seq_len,
            "fp16_mb": fp16_mb,
            "turboquant_mb": tq_mb,
            "compression_ratio": compression_ratio,
            "memory_saved_pct": memory_saved_pct,
            "memory_saved_mb": memory_saved_mb
        })
    
    # Detailed breakdown for 2048 tokens
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN (2048 tokens)")
    print("=" * 80)
    
    seq_len = 2048
    
    # FP16 breakdown
    k_cache_fp16 = num_kv_heads * seq_len * head_dim * 2
    v_cache_fp16 = num_kv_heads * seq_len * head_dim * 2
    total_fp16_per_layer = k_cache_fp16 + v_cache_fp16
    total_fp16 = total_fp16_per_layer * full_attn_layers
    
    print(f"\nFP16 (Baseline):")
    print(f"  Per layer:")
    print(f"    - K cache: {k_cache_fp16 / (1024**2):.3f} MB")
    print(f"    - V cache: {v_cache_fp16 / (1024**2):.3f} MB")
    print(f"    - Total: {total_fp16_per_layer / (1024**2):.3f} MB")
    print(f"  All {full_attn_layers} layers:")
    print(f"    - Total: {total_fp16 / (1024**2):.3f} MB ({total_fp16 / (1024**3):.3f} GB)")
    
    # TurboQuant breakdown
    k_quantized = num_kv_heads * seq_len * head_dim * 0.25
    v_quantized = num_kv_heads * seq_len * head_dim * 0.25
    k_scales = num_kv_heads * seq_len * 2
    v_scales = num_kv_heads * seq_len * 2
    k_residual = num_kv_heads * 32 * head_dim * 2
    v_residual = num_kv_heads * 32 * head_dim * 2
    
    total_tq_per_layer = k_quantized + v_quantized + k_scales + v_scales + k_residual + v_residual
    total_tq = total_tq_per_layer * full_attn_layers
    
    print(f"\nTurboQuant (2-bit):")
    print(f"  Per layer:")
    print(f"    - K quantized (2-bit): {k_quantized / (1024**2):.3f} MB")
    print(f"    - V quantized (2-bit): {v_quantized / (1024**2):.3f} MB")
    print(f"    - K scales (FP16): {k_scales / (1024**2):.3f} MB")
    print(f"    - V scales (FP16): {v_scales / (1024**2):.3f} MB")
    print(f"    - K residual (FP16): {k_residual / (1024**2):.3f} MB")
    print(f"    - V residual (FP16): {v_residual / (1024**2):.3f} MB")
    print(f"    - Total: {total_tq_per_layer / (1024**2):.3f} MB")
    print(f"  All {full_attn_layers} layers:")
    print(f"    - Total: {total_tq / (1024**2):.3f} MB ({total_tq / (1024**3):.3f} GB)")
    
    # Comparison
    compression = total_fp16 / total_tq
    saved_pct = (1 - total_tq / total_fp16) * 100
    
    print(f"\nComparison:")
    print(f"  Compression ratio: {compression:.2f}x")
    print(f"  Memory saved: {saved_pct:.1f}%")
    print(f"  Absolute savings: {(total_fp16 - total_tq) / (1024**2):.2f} MB")
    
    # Scale to full model
    print("\n" + "=" * 80)
    print("FULL MODEL CONTEXT (Real-world scenario)")
    print("=" * 80)
    
    # Typical inference scenario: 4K context, batch size 1
    context_len = 4096
    batch_size = 1
    
    fp16_4k = 2 * num_kv_heads * context_len * head_dim * 2 * full_attn_layers * batch_size
    tq_4k = (2 * num_kv_heads * context_len * head_dim * 0.25 + 
             2 * num_kv_heads * context_len * 2 +
             2 * num_kv_heads * 32 * head_dim * 2) * full_attn_layers * batch_size
    
    print(f"\nScenario: {context_len} token context, batch size {batch_size}")
    print(f"  FP16 KV cache: {fp16_4k / (1024**2):.2f} MB ({fp16_4k / (1024**3):.2f} GB)")
    print(f"  TurboQuant: {tq_4k / (1024**2):.2f} MB ({tq_4k / (1024**3):.2f} GB)")
    print(f"  Compression: {fp16_4k / tq_4k:.2f}x")
    print(f"  Memory saved: {(1 - tq_4k / fp16_4k) * 100:.1f}%")
    
    # Multiple concurrent requests
    print(f"\nConcurrent requests (batch size 8):")
    fp16_batch = fp16_4k * 8
    tq_batch = tq_4k * 8
    print(f"  FP16 KV cache: {fp16_batch / (1024**2):.2f} MB ({fp16_batch / (1024**3):.2f} GB)")
    print(f"  TurboQuant: {tq_batch / (1024**2):.2f} MB ({tq_batch / (1024**3):.2f} GB)")
    print(f"  Compression: {fp16_batch / tq_batch:.2f}x")
    print(f"  Memory saved: {(1 - tq_batch / fp16_batch) * 100:.1f}%")
    
    # Save results
    output_file = Path(__file__).parent / "memory_analysis.json"
    with open(output_file, "w") as f:
        json.dump({
            "model": "Qwen/Qwen3.5-9B",
            "architecture": {
                "total_layers": total_layers,
                "full_attn_layers": full_attn_layers,
                "linear_attn_layers": linear_attn_layers,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim
            },
            "sequence_lengths": results
        }, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    analyze_memory_usage()
