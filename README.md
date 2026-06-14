# KV Cache Compression

Research implementation of KV cache compression methods for LLM inference, benchmarking memory savings, throughput, and quality tradeoffs.

## Overview

KV cache compression addresses a critical bottleneck in LLM inference: as sequence lengths grow, the key-value cache can consume more GPU memory than the model weights themselves. This repository implements and benchmarks compression methods to reduce KV cache footprint while maintaining generation quality.

### Why KV Cache Compression Matters

During autoregressive generation, every new token adds keys and values to the cache. At 128K+ context lengths, the KV cache becomes the dominant memory consumer:
- Memory bandwidth bound: GPU must read massive amounts of KV data per token
- Limits batch size and context length
- KV cache is the single largest cost driver for long-context inference

## Methods

### 1. Normal KV Cache (FP16 Baseline)

Standard uncompressed KV cache storing keys and values in full FP16 precision.

**Implementation:** `normal_kv.py`

### 2. KIVI (2-bit Asymmetric Quantization)

Based on the paper: ["KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"](https://arxiv.org/abs/2402.02750) (Liu et al., ICML 2024)

**Key Insight:** Keys and values behave fundamentally differently:
- **Keys:** Have persistent channel outliers that stay important across many tokens → quantize per-channel
- **Values:** Are dynamic and vary token by token → quantize per-token

**Implementation:** `kivi_kv.py` (PyTorch)

**Algorithm:**
1. Keys: Compute min/max per channel across sequence dimension, quantize to 2-bit with per-channel scales
2. Values: Compute min/max per token across head dimension, quantize to 2-bit with per-token scales
3. Maintain residual buffer (last 128 tokens) in FP16 for local attention
4. Dequantize on-the-fly during attention computation

## Benchmark Results (PyTorch Implementation)

**Model:** GPT-2 (124M parameters)  
**Config:** 12 layers × 12 heads × 64 dim  
**Device:** CUDA

### Memory Usage

| Sequence Length | Normal (FP16) | KIVI (2-bit) | Compression |
|----------------|---------------|--------------|-------------|
| 128 tokens     | 4.50 MB       | 5.17 MB      | 0.87x       |
| 512 tokens     | 18.00 MB      | 7.07 MB      | 2.55x       |
| 1024 tokens    | 36.00 MB      | 9.60 MB      | 3.75x       |
| 2048 tokens    | 72.00 MB      | 14.66 MB     | **4.91x**   |

**Key Observation:** At short sequence lengths (128 tokens), KIVI uses MORE memory due to the residual buffer overhead. Compression becomes effective at longer sequences where 2-bit quantization dominates.

### Generation Performance

| Method | Speed (tok/s) | Memory (GB) | Compression |
|--------|---------------|-------------|-------------|
| Normal | 197.95        | 0.24        | 1.0x        |
| KIVI   | 227.64        | 0.14        | ~4-5x*      |

*Theoretical compression at typical generation lengths

**Speedup:** 1.15x faster with KIVI (estimated)

### Quality Analysis

| Method | Perplexity | Degradation |
|--------|------------|-------------|
| Normal | 1.92       | baseline    |
| KIVI   | 1.95       | +1.50%      |

**Quality Impact:** Minimal - only 1.5% perplexity increase, aligning with the KIVI paper's findings.

## Analysis

### Why KIVI Shows Poor Compression at Short Lengths

The KIVI implementation maintains a **residual buffer** of the last 128 tokens in FP16 for local attention. This is necessary because:

1. Per-channel quantization requires multiple tokens to compute channel statistics
2. Recent tokens benefit from full precision for accurate local attention
3. Streaming inference needs a buffer before quantization

**At 128 tokens:**
- Quantized cache: 0.56 MB
- Scales/zero-points: 0.11 MB  
- Residual buffer: 4.50 MB (FP16 for all 128 tokens)
- **Total: 5.17 MB** (worse than FP16's 4.50 MB)

**At 2048 tokens:**
- Quantized cache: 9.00 MB
- Scales/zero-points: 1.16 MB
- Residual buffer: 4.50 MB (only last 128 tokens)
- **Total: 14.66 MB** (4.91x better than FP16's 72.00 MB)

### Why KIVI is Faster

The theoretical speedup comes from:

1. **Reduced memory bandwidth:** 2-bit data requires 8x less HBM reads during attention
2. **Better cache utilization:** Smaller KV cache fits better in GPU caches
3. **Larger batch sizes:** Freed memory allows processing more requests in parallel

### Quality Preservation

KIVI's asymmetric quantization strategy is key to quality preservation:

- **Keys (per-channel):** Preserves persistent outlier channels that are critical for attention
- **Values (per-token):** Adapts to dynamic per-token variations

This is why KIVI achieves only 1.5% perplexity degradation at 2-bit, while naive quantization would cause much larger quality loss.

## Benchmark Visualizations

![Memory Usage](images/kivi-benchmarks/memory_usage.png)

![Compression Ratio](images/kivi-benchmarks/compression_ratio.png)

![Speed Comparison](images/kivi-benchmarks/speed_comparison.png)

![Quality Comparison](images/kivi-benchmarks/quality_comparison.png)

![Summary](images/kivi-benchmarks/summary.png)

## Project Structure

```
KV-Compression/
├── normal_kv.py              # FP16 baseline implementation
├── kivi_kv.py                # KIVI 2-bit asymmetric quantization
├── plot_results.py           # Benchmark visualization
├── benchmarks/
│   ├── benchmark.py          # Comprehensive benchmark suite
│   └── benchmark_results_*.json
└── images/
    └── kivi-benchmarks/      # Benchmark plots
```

## Usage

### Run Benchmarks

```bash
cd benchmarks
python benchmark.py
```

This will:
1. Load GPT-2 model
2. Test memory usage at different sequence lengths
3. Benchmark generation speed
4. Measure quality (perplexity)
5. Compare Normal vs KIVI
6. Save results to JSON

### Generate Visualizations

```bash
python plot_results.py
```

Generates plots in `images/kivi-benchmarks/`:
- Memory usage over sequence length
- Compression ratio progression
- Speed comparison
- Quality comparison
- Comprehensive summary

### Test Individual Components

```bash
# Test normal KV cache
python normal_kv.py

# Test KIVI implementation
python kivi_kv.py
```

## Implementation Notes

This is a **PyTorch implementation** for educational purposes and algorithm understanding. The benchmarks show theoretical/estimated performance for KIVI since the quantization/dequantization overhead in pure PyTorch is significant.

**For production-grade performance:**
- Implement fused CUDA/Triton kernels for quantization/dequantization
- Integrate with vLLM or similar inference frameworks
- Optimize memory layout for GPU efficiency
- The KIVI paper reports 2.35x-3.47x throughput with their Triton implementation

## Model Support

Currently tested with:
- GPT-2 (124M)
- GPT-2 Medium (355M)
- GPT-2 Large (774M)

To use a different model, change `model_name` in `benchmarks/benchmark.py`:
```python
benchmark = KVCacheBenchmark(model_name='gpt2-medium')
```

## Practical Implications

### When to Use KIVI

**Good for:**
- Long-context inference (1024+ tokens)
- Memory-constrained deployments
- High-throughput serving (larger batch sizes)
- Production systems where quality is critical

**Not ideal for:**
- Very short sequences (<256 tokens) - residual buffer overhead dominates
- Latency-critical single-token generation
- When you need maximum compression (consider 4-bit or 8-bit instead)

## Future Work

- Implement Triton kernels for production-grade performance
- Add more compression methods (quantization-based, eviction-based, hybrid)
- Test on larger models (Llama-3, Mistral, etc.)
- Benchmark on real workloads (LongBench, Needle-in-Haystack, etc.)
- Compare different bit-widths and quantization strategies

## References

- KIVI Paper: https://arxiv.org/abs/2402.02750
- Blog Post: https://mog9.github.io/blogs/KV/index.html
