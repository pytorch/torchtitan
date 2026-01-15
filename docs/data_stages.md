# Multi-Stage Data Training

Multi-stage training allows switching between different data mixtures at specified training steps, similar to approaches used in Qwen3, DeepSeek-V3, and Llama 3.

## Quick Start

Add `[[training.data_stages]]` sections to your TOML config:

```toml
[training]
dataset_type = "nanoset"
dataset_folders = ["/data/general", "/data/math", "/data/code"]
dataset_weights = [0.8, 0.1, 0.1]
steps = 150000

[[training.data_stages]]
name = "general"
start_step = 0
end_step = 100000
dataset_weights = [0.8, 0.1, 0.1]

[[training.data_stages]]
name = "reasoning"
start_step = 100000
end_step = 130000
dataset_weights = [0.3, 0.35, 0.35]

[[training.data_stages]]
name = "long_context"
start_step = 130000
dataset_weights = [0.3, 0.35, 0.35]
seq_len = 32768
```

## Configuration Fields

Each `[[training.data_stages]]` section supports:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Stage identifier for logging |
| `start_step` | int | Yes | Step when stage begins (inclusive) |
| `end_step` | int | No | Step when stage ends (exclusive). Omit for final stage |
| `dataset` | string | No | Dataset name. Inherits from `[training]` if not set |
| `dataset_path` | string | No | Path to dataset |
| `dataset_type` | string | No | `"huggingface"`, `"nanoset"`, `"preprocessed"`, `"packed_memmap"` |
| `dataset_folders` | list | No | Folders for nanoset datasets |
| `dataset_weights` | list | No | Weights for blending datasets |
| `dataset_random_seed` | int | No | Random seed for this stage |
| `seq_len` | int | No | Sequence length (for context extension) |

**Note:** Fields set to `None` or omitted inherit from the base `[training]` config.

## Common Patterns

### Pattern 1: Change Data Mixture

```toml
[[training.data_stages]]
name = "pretrain"
start_step = 0
end_step = 100000
dataset_weights = [0.7, 0.2, 0.1]  # 70% web, 20% books, 10% code

[[training.data_stages]]
name = "annealing"
start_step = 100000
dataset_weights = [0.4, 0.3, 0.3]  # More balanced for final phase
```

### Pattern 2: Context Extension

```toml
[[training.data_stages]]
name = "base"
start_step = 0
end_step = 90000
seq_len = 4096

[[training.data_stages]]
name = "long_context"
start_step = 90000
seq_len = 32768
```

### Pattern 3: Different Random Seeds

```toml
[[training.data_stages]]
name = "epoch1"
start_step = 0
end_step = 50000
dataset_random_seed = 1234

[[training.data_stages]]
name = "epoch2"
start_step = 50000
dataset_random_seed = 5678
```

## Logging

At training start, a stage plan is logged:

```
============================================================
DATA STAGE TRAINING PLAN
============================================================
Total stages: 3

Stage 1: general
  Steps: 0 -> 100,000 (100,000 steps)
  Estimated tokens: 409.60B tokens
  Dataset type: nanoset
  Dataset folders: 3 folders
  Weights: [0.800, 0.100, 0.100]
  Sequence length: 4096

Stage 2: reasoning
  Steps: 100,000 -> 130,000 (30,000 steps)
  ...
============================================================
```

At each transition:

```
============================================================
DATA STAGE TRANSITION
============================================================
Step 100000: 'general' -> 'reasoning'
Changes: dataset_weights
New weights: [0.300, 0.350, 0.350]
============================================================
```

## Checkpoint & Resume

Stage state is automatically saved in checkpoints:
- `stage_idx`: Current stage index
- `stage_name`: Current stage name
- `dataloader_state`: Position within the dataset

On resume, the exact stage and dataloader position are restored. No manual intervention needed.

## Backward Compatibility

- If no `[[training.data_stages]]` sections are defined, training runs as single-stage (existing behavior)
- All existing configs work without modification
