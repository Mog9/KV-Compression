"""
KV Cache Compression Visualization

Creates publication-quality plots comparing Normal KV Cache vs KIVI across:
- Memory usage over sequence length
- Compression ratio over sequence length
- Speed (tokens/sec) comparison
- Latency comparison
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
from pathlib import Path

# Set style for publication-quality plots
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Color scheme
COLORS = {
    'normal': '#2E86AB',      # Blue
    'kivi': '#A23B72',        # Purple
    'normal_fill': '#2E86AB',
    'kivi_fill': '#A23B72',
}


def load_benchmark_results(json_file):
    """Load benchmark results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def plot_memory_usage(results, save_path='memory_usage.png'):
    """Plot memory usage over sequence length."""
    memory_data = results['memory_analysis']
    
    seq_lengths = sorted([int(k) for k in memory_data.keys()])
    normal_memory = [memory_data[str(s)]['normal']['memory_mb'] for s in seq_lengths]
    kivi_memory = [memory_data[str(s)]['kivi']['memory_mb'] for s in seq_lengths]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lines
    ax.plot(seq_lengths, normal_memory, 'o-', color=COLORS['normal'], 
            linewidth=2.5, markersize=8, label='Normal KV (FP16)', zorder=3)
    ax.plot(seq_lengths, kivi_memory, 's-', color=COLORS['kivi'], 
            linewidth=2.5, markersize=8, label='KIVI (2-bit)', zorder=3)
    
    # Fill between to show difference
    ax.fill_between(seq_lengths, normal_memory, kivi_memory, 
                    alpha=0.15, color=COLORS['kivi'], zorder=2)
    
    # Add annotations
    for i, (seq, norm, kivi) in enumerate(zip(seq_lengths, normal_memory, kivi_memory)):
        compression = norm / kivi
        if compression > 1:
            ax.annotate(f'{compression:.1f}x', 
                       xy=(seq, (norm + kivi) / 2),
                       xytext=(10, 0), textcoords='offset points',
                       fontsize=9, color=COLORS['kivi'], fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor=COLORS['kivi'], alpha=0.8))
    
    ax.set_xlabel('Sequence Length (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('KV Cache Memory (MB)', fontsize=12, fontweight='bold')
    ax.set_title('KV Cache Memory Usage: Normal vs KIVI', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray')
    ax.set_xlim(min(seq_lengths) - 50, max(seq_lengths) + 50)
    ax.set_ylim(0, max(normal_memory) * 1.1)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_compression_ratio(results, save_path='compression_ratio.png'):
    """Plot compression ratio over sequence length."""
    memory_data = results['memory_analysis']
    
    seq_lengths = sorted([int(k) for k in memory_data.keys()])
    compression_ratios = [memory_data[str(s)]['kivi']['compression_ratio'] 
                          for s in seq_lengths]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot line
    ax.plot(seq_lengths, compression_ratios, 'o-', color=COLORS['kivi'], 
            linewidth=3, markersize=10, label='KIVI Compression', zorder=3)
    
    # Add baseline at 1x
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, 
               label='No compression (1x)', alpha=0.5, zorder=2)
    
    # Add annotations
    for i, (seq, ratio) in enumerate(zip(seq_lengths, compression_ratios)):
        ax.annotate(f'{ratio:.2f}x', 
                   xy=(seq, ratio),
                   xytext=(15, 10), textcoords='offset points',
                   fontsize=10, color=COLORS['kivi'], fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor=COLORS['kivi'], alpha=0.9, linewidth=1.5))
    
    ax.set_xlabel('Sequence Length (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Compression Ratio', fontsize=12, fontweight='bold')
    ax.set_title('KIVI Compression Ratio Over Sequence Length', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray')
    ax.set_xlim(min(seq_lengths) - 50, max(seq_lengths) + 50)
    ax.set_ylim(0, max(compression_ratios) * 1.2)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_speed_comparison(results, save_path='speed_comparison.png'):
    """Plot generation speed comparison."""
    gen_data = results['generation_benchmark']
    
    prompts = list(gen_data.keys())
    normal_speeds = [gen_data[p]['normal']['tokens_per_second'] for p in prompts]
    kivi_speeds = [gen_data[p]['kivi']['tokens_per_second'] for p in prompts]
    
    x = np.arange(len(prompts))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bars
    bars1 = ax.bar(x - width/2, normal_speeds, width, label='Normal KV (FP16)',
                   color=COLORS['normal'], alpha=0.85, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, kivi_speeds, width, label='KIVI (2-bit)',
                   color=COLORS['kivi'], alpha=0.85, edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add speedup annotations
    for i, (norm, kivi) in enumerate(zip(normal_speeds, kivi_speeds)):
        speedup = kivi / norm
        ax.annotate(f'{speedup:.2f}x',
                   xy=(i, max(norm, kivi) + 5),
                   ha='center', fontsize=10, color=COLORS['kivi'], fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=COLORS['kivi'], alpha=0.9))
    
    ax.set_xlabel('Test Prompt', fontsize=12, fontweight='bold')
    ax.set_ylabel('Generation Speed (tokens/sec)', fontsize=12, fontweight='bold')
    ax.set_title('Generation Speed: Normal vs KIVI', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([p[:30] + '...' for p in prompts], rotation=15, ha='right')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
    ax.set_ylim(0, max(max(normal_speeds), max(kivi_speeds)) * 1.2)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_quality_comparison(results, save_path='quality_comparison.png'):
    """Plot quality (perplexity) comparison."""
    quality_data = results['quality_analysis']
    
    texts = list(quality_data.keys())
    normal_ppl = [quality_data[t]['normal_ppl'] for t in texts]
    kivi_ppl = [quality_data[t]['kivi_ppl'] for t in texts]
    degradations = [quality_data[t]['kivi_degradation_pct'] for t in texts]
    
    x = np.arange(len(texts))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Perplexity
    bars1 = ax1.bar(x - width/2, normal_ppl, width, label='Normal KV (FP16)',
                    color=COLORS['normal'], alpha=0.85, edgecolor='white', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, kivi_ppl, width, label='KIVI (2-bit)',
                    color=COLORS['kivi'], alpha=0.85, edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Test Text', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Perplexity (lower is better)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Quality: Perplexity', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([t[:25] + '...' for t in texts], rotation=15, ha='right')
    ax1.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
    ax1.set_ylim(0, max(max(normal_ppl), max(kivi_ppl)) * 1.15)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Right plot: Degradation
    bars3 = ax2.bar(x, degradations, width=0.5, color=COLORS['kivi'], 
                    alpha=0.85, edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'+{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color=COLORS['kivi'])
    
    ax2.set_xlabel('Test Text', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Perplexity Increase (%)', fontsize=12, fontweight='bold')
    ax2.set_title('KIVI Quality Degradation', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([t[:25] + '...' for t in texts], rotation=15, ha='right')
    ax2.set_ylim(0, max(degradations) * 1.3)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_comprehensive_summary(results, save_path='summary.png'):
    """Create a comprehensive summary plot with all metrics."""
    fig = plt.figure(figsize=(16, 12))
    
    # Create 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Memory Usage (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    memory_data = results['memory_analysis']
    seq_lengths = sorted([int(k) for k in memory_data.keys()])
    normal_memory = [memory_data[str(s)]['normal']['memory_mb'] for s in seq_lengths]
    kivi_memory = [memory_data[str(s)]['kivi']['memory_mb'] for s in seq_lengths]
    
    ax1.plot(seq_lengths, normal_memory, 'o-', color=COLORS['normal'], 
             linewidth=2.5, markersize=8, label='Normal (FP16)')
    ax1.plot(seq_lengths, kivi_memory, 's-', color=COLORS['kivi'], 
             linewidth=2.5, markersize=8, label='KIVI (2-bit)')
    ax1.fill_between(seq_lengths, normal_memory, kivi_memory, alpha=0.15, color=COLORS['kivi'])
    ax1.set_xlabel('Sequence Length', fontsize=11)
    ax1.set_ylabel('Memory (MB)', fontsize=11)
    ax1.set_title('Memory Usage', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Compression Ratio (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    compression_ratios = [memory_data[str(s)]['kivi']['compression_ratio'] for s in seq_lengths]
    ax2.plot(seq_lengths, compression_ratios, 'o-', color=COLORS['kivi'], 
             linewidth=2.5, markersize=8)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    for seq, ratio in zip(seq_lengths, compression_ratios):
        ax2.annotate(f'{ratio:.1f}x', xy=(seq, ratio), xytext=(10, 5), 
                    textcoords='offset points', fontsize=9, fontweight='bold',
                    color=COLORS['kivi'])
    ax2.set_xlabel('Sequence Length', fontsize=11)
    ax2.set_ylabel('Compression Ratio', fontsize=11)
    ax2.set_title('Compression Ratio', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Speed Comparison (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    gen_data = results['generation_benchmark']
    prompts = list(gen_data.keys())
    normal_speeds = [gen_data[p]['normal']['tokens_per_second'] for p in prompts]
    kivi_speeds = [gen_data[p]['kivi']['tokens_per_second'] for p in prompts]
    
    x = np.arange(len(prompts))
    width = 0.35
    ax3.bar(x - width/2, normal_speeds, width, label='Normal', color=COLORS['normal'], alpha=0.85)
    ax3.bar(x + width/2, kivi_speeds, width, label='KIVI', color=COLORS['kivi'], alpha=0.85)
    ax3.set_xlabel('Test Prompt', fontsize=11)
    ax3.set_ylabel('Speed (tok/s)', fontsize=11)
    ax3.set_title('Generation Speed', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'P{i+1}' for i in range(len(prompts))])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Quality (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    quality_data = results['quality_analysis']
    texts = list(quality_data.keys())
    normal_ppl = [quality_data[t]['normal_ppl'] for t in texts]
    kivi_ppl = [quality_data[t]['kivi_ppl'] for t in texts]
    
    x = np.arange(len(texts))
    ax4.bar(x - width/2, normal_ppl, width, label='Normal', color=COLORS['normal'], alpha=0.85)
    ax4.bar(x + width/2, kivi_ppl, width, label='KIVI', color=COLORS['kivi'], alpha=0.85)
    ax4.set_xlabel('Test Text', fontsize=11)
    ax4.set_ylabel('Perplexity', fontsize=11)
    ax4.set_title('Model Quality (PPL)', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'T{i+1}' for i in range(len(texts))])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add main title
    fig.suptitle('KV Cache Compression: Normal vs KIVI Benchmark Summary', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    """Generate all plots."""
    print("="*80)
    print("GENERATING KV CACHE COMPRESSION VISUALIZATIONS")
    print("="*80)
    
    # Find the latest benchmark results file
    results_files = list(Path('benchmarks').glob('benchmark_results_*.json'))
    if not results_files:
        print("ERROR: No benchmark results found. Run benchmarks/benchmark.py first.")
        return
    
    latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
    print(f"\nLoading results from: {latest_results}")
    
    results = load_benchmark_results(latest_results)
    
    print("\nGenerating plots...")
    print("-" * 80)
    
    # Output directory for images
    img_dir = 'images/kivi-benchmarks'
    Path(img_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate individual plots
    plot_memory_usage(results, f'{img_dir}/memory_usage.png')
    plot_compression_ratio(results, f'{img_dir}/compression_ratio.png')
    plot_speed_comparison(results, f'{img_dir}/speed_comparison.png')
    plot_quality_comparison(results, f'{img_dir}/quality_comparison.png')
    
    # Generate comprehensive summary
    plot_comprehensive_summary(results, f'{img_dir}/summary.png')
    
    print("-" * 80)
    print("\nAll plots generated successfully!")
    print("\nGenerated files:")
    print("  1. images/kivi-benchmarks/memory_usage.png       - Memory usage over sequence length")
    print("  2. images/kivi-benchmarks/compression_ratio.png  - Compression ratio progression")
    print("  3. images/kivi-benchmarks/speed_comparison.png   - Generation speed comparison")
    print("  4. images/kivi-benchmarks/quality_comparison.png - Quality (perplexity) comparison")
    print("  5. images/kivi-benchmarks/summary.png            - Comprehensive 4-panel summary")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
