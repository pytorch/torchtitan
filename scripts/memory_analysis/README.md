# Memory Analysis Tools

Tools for analyzing PyTorch memory snapshots and identifying optimization opportunities.

## Directory Structure

```
scripts/memory_analysis/
├── analyze.py                      # Main analysis tool
├── results/                        # All analysis results (organized by timestamp)
│   ├── latest/                    # Symlink to most recent run
│   ├── 20251114_143025_lbs6/      # Timestamped run directories
│   │   ├── *.png                  # Visualizations
│   │   ├── analysis_report.txt    # Detailed report
│   │   └── README.md              # Run-specific findings
│   └── archived/                  # Historical notes and summaries
└── README.md                      # This file (usage guide)
```

Each run automatically creates a timestamped directory: `YYYYMMDD_HHMMSS_name/`
- The `latest` symlink always points to the most recent analysis
- Run names are auto-detected from snapshot paths or can be customized

## Quick Start

### 1. Capture Memory Snapshot

Run training with memory profiling enabled:

```bash
NGPU=8 CONFIG_FILE="path/to/config.toml" ./run_train.sh \
  --training.local_batch_size 6 \
  --training.steps 5 \
  --profiling.enable_memory_snapshot
```

This saves snapshots to: `./outputs/memory_snapshot/iteration_*/rank*_memory_snapshot.pickle`

### 2. Analyze Snapshot

```bash
# Auto-generates timestamped directory (e.g., 20251114_143025_iter1_rank0/)
python3 scripts/memory_analysis/analyze.py \
  ./outputs/memory_snapshot/iteration_1/rank0_memory_snapshot.pickle

# Or with custom name
python3 scripts/memory_analysis/analyze.py \
  ./outputs/memory_snapshot/iteration_1/rank0_memory_snapshot.pickle \
  scripts/memory_analysis/results \
  lbs6_selective_ac
```

### 3. View Results

Results are in timestamped directories under `scripts/memory_analysis/results/`:
- **Latest run**: `results/latest/` (symlink)
- **5 PNG visualizations**: Overview, top consumers, category breakdown, distributions, efficiency
- **Text report**: `analysis_report.txt`
- **Findings summary**: `README.md` (auto-generated per run)

## Analysis Features

### Visualizations Generated

1. **memory_overview.png**
   - Pie chart: Allocated vs Cache/Fragmentation
   - Pie chart: Memory by category (Expert/MoE, Attention, etc.)

2. **top_consumers.png**
   - Bar chart of top 30 individual allocations
   - Color-coded by category

3. **category_breakdown.png**
   - Bar chart showing memory usage by category
   - Shows which components use the most memory

4. **size_distribution.png**
   - Histogram of active allocation sizes
   - Histogram of cached/fragmented block sizes

5. **memory_efficiency.png**
   - Summary metrics: Total Reserved, Allocated, Wasted
   - Visual efficiency indicator

### Text Report

Includes:
- Memory efficiency percentage
- Breakdown by category (Expert/MoE, Attention, etc.)
- Top 50 allocations by size
- Fragmentation analysis
- Optimization recommendations

## Interpreting Results

### Memory Efficiency

- **>98%**: Excellent - minimal waste
- **95-98%**: Good - some room for improvement
- **<95%**: Poor - significant fragmentation/waste

### Common Categories

- **Expert/MoE**: MoE layer weights and activations (largest for MoE models)
- **Attention**: Attention mechanism tensors (Q, K, V, attention scores)
- **Linear/MatMul**: Dense layer activations
- **Optimizer State**: AdamW momentum and variance buffers
- **Communication**: DeepEP/FSDP all-to-all buffers
- **FSDP**: FSDP sharding overhead
- **Gradient**: Gradient tensors

### Optimization Recommendations

The tool automatically identifies:
- High cache overhead (>10%)
- Large unused blocks (>100MB)
- Fragmentation issues
- Potential memory leaks

## Example Usage

```bash
# Simple: Auto-detect run name from path
python3 scripts/memory_analysis/analyze.py \
  outputs/memory_snapshot/iteration_1/rank0_memory_snapshot.pickle
# Creates: results/20251114_143025_iter1_rank0/

# With custom name: LBS=6 baseline
python3 scripts/memory_analysis/analyze.py \
  outputs/memory_snapshot/iteration_1/rank0_memory_snapshot.pickle \
  scripts/memory_analysis/results \
  lbs6_baseline
# Creates: results/20251114_143025_lbs6_baseline/

# Compare LBS=8 with selective AC
python3 scripts/memory_analysis/analyze.py \
  outputs/memory_snapshot_lbs8/iteration_1/rank0_memory_snapshot.pickle \
  scripts/memory_analysis/results \
  lbs8_selective_ac
# Creates: results/20251114_143530_lbs8_selective_ac/

# Compare LBS=8 with full AC
python3 scripts/memory_analysis/analyze.py \
  outputs/memory_snapshot_lbs8_full/iteration_1/rank0_memory_snapshot.pickle \
  scripts/memory_analysis/results \
  lbs8_full_ac
# Creates: results/20251114_143845_lbs8_full_ac/
```

## Troubleshooting

### No snapshot files found

Make sure you ran training with `--profiling.enable_memory_snapshot`

### Import errors

Ensure matplotlib is installed:
```bash
pip install matplotlib numpy
```

### Out of memory during analysis

The analysis script uses minimal memory - this shouldn't happen. The snapshots are typically <100MB.

## Advanced Options

### Custom Output Directory

```bash
python3 scripts/memory_analysis/analyze.py <snapshot> <custom_output_dir>
```

### Analyzing Multiple Ranks

```bash
for rank in 0 1 2 3 4 5 6 7; do
  python3 scripts/memory_analysis/analyze.py \
    outputs/memory_snapshot/iteration_1/rank${rank}_memory_snapshot.pickle \
    scripts/memory_analysis/results/rank${rank}
done
```

### Compare Across Iterations

```bash
for iter in 1 2 3 4 5; do
  python3 scripts/memory_analysis/analyze.py \
    outputs/memory_snapshot/iteration_${iter}/rank0_memory_snapshot.pickle \
    scripts/memory_analysis/results/iter${iter}
done
```

## Implementation Details

### Snapshot Format

PyTorch memory snapshots contain:
- **segments**: Memory segments allocated by CUDA
- **blocks**: Individual allocations within segments
  - `size`: Allocation size in bytes
  - `state`: 'active_allocated', 'inactive', etc.
  - `frames`: Stack trace of allocation location

### Categorization Logic

Tensors are categorized based on their allocation stack traces:
- Searches for keywords in filenames and function names
- Falls back to "Unknown" if no match

### Memory Calculations

- **Allocated**: Sum of all active blocks
- **Reserved**: Total CUDA memory reserved by PyTorch
- **Cache/Wasted**: Reserved - Allocated
- **Efficiency**: Allocated / Reserved * 100%

## Related Files

- `/home/phuc/workspace/moe/torchtitan/torchtitan/tools/profiling.py` - Memory snapshot capture
- `/home/phuc/workspace/moe/torchtitan/torchtitan/train.py:689-698` - Cache clearing fix
- `/home/phuc/workspace/moe/torchtitan/run_train.sh:20` - PYTORCH_ALLOC_CONF setup
