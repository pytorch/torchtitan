# Graph Comparison: non_strict (make_fx) vs titan (dynamo+aot_autograd)

## Overview

Comparison of FX joint graphs produced by two tracing approaches on torchtitan models with FSDP (`dp_shard=8`):

- **non_strict**: `make_fx(TrainStepModule)` — traces forward + loss + explicit `autograd.grad` as a flat graph
- **titan**: `export_joint(ForwardLossModule)` — dynamo captures forward+loss, aot_autograd derives backward

Both use identical model configs (debugmodel), weights (`torch.manual_seed(42)`), loss function (`cross_entropy`), and inputs (`batch=2, seq=128`).

## Node Counts

| Model | non_strict | titan | Delta |
|-------|-----------|-------|-------|
| llama3 | 1604 | 1604 | 0 |
| qwen3 | 2504 | 2504 | 0 |
| deepseek_v3 | 2537 | 2502 | -35 |
| llama4 | 1848 | 1788 | -60 |
| gpt_oss | 1872 | 1808 | -64 |

Simple models (llama3, qwen3) have identical node counts. MoE/complex models (deepseek_v3, llama4, gpt_oss) have fewer nodes in titan due to functionalization eliminating redundant ops.

## Per-Model Op Frequency Diffs

### llama3 (3 ops differ)

| Op | non_strict | titan | delta |
|----|-----------|-------|-------|
| `aten._unsafe_view.default` | 85 | 73 | -12 |
| `aten.ones_like.default` | 1 | 0 | -1 |
| `aten.view.default` | 321 | 333 | +12 |

### qwen3 (3 ops differ)

| Op | non_strict | titan | delta |
|----|-----------|-------|-------|
| `aten._unsafe_view.default` | 97 | 81 | -16 |
| `aten.ones_like.default` | 1 | 0 | -1 |
| `aten.view.default` | 425 | 441 | +16 |

### deepseek_v3 (11 ops differ)

| Op | non_strict | titan | delta |
|----|-----------|-------|-------|
| `<built-in function getitem>` | 134 | 124 | -10 |
| `aten._to_copy.default` | 70 | 65 | -5 |
| `aten._unsafe_view.default` | 64 | 52 | -12 |
| `aten.add.Tensor` | 72 | 77 | +5 |
| `aten.add_.Tensor` | 5 | 0 | -5 |
| `aten.index.Tensor` | 25 | 20 | -5 |
| `aten.index_put.default` | 10 | 20 | +10 |
| `aten.index_put_.default` | 10 | 0 | -10 |
| `aten.ones_like.default` | 1 | 0 | -1 |
| `aten.slice.Tensor` | 51 | 46 | -5 |
| `aten.view.default` | 377 | 379 | +2 |

### llama4 (10 ops differ)

| Op | non_strict | titan | delta |
|----|-----------|-------|-------|
| `<built-in function getitem>` | 108 | 90 | -18 |
| `aten._to_copy.default` | 42 | 39 | -3 |
| `aten.add.Tensor` | 60 | 63 | +3 |
| `aten.add_.Tensor` | 3 | 0 | -3 |
| `aten.index_put.default` | 9 | 15 | +6 |
| `aten.index_put_.default` | 6 | 0 | -6 |
| `aten.new_zeros.default` | 45 | 15 | -30 |
| `aten.ones_like.default` | 1 | 0 | -1 |
| `aten.slice.Tensor` | 13 | 10 | -3 |
| `aten.view.default` | 306 | 300 | -6 |

### gpt_oss (13 ops differ)

| Op | non_strict | titan | delta |
|----|-----------|-------|-------|
| `<built-in function getitem>` | 82 | 66 | -16 |
| `aten._to_copy.default` | 60 | 56 | -4 |
| `aten.add.Tensor` | 88 | 92 | +4 |
| `aten.add_.Tensor` | 4 | 0 | -4 |
| `aten.index.Tensor` | 20 | 16 | -4 |
| `aten.index_put.default` | 8 | 16 | +8 |
| `aten.index_put_.default` | 8 | 0 | -8 |
| `aten.logical_and.default` | 0 | 4 | +4 |
| `aten.logical_and_.default` | 4 | 0 | -4 |
| `aten.new_zeros.default` | 36 | 16 | -20 |
| `aten.ones_like.default` | 1 | 0 | -1 |
| `aten.slice.Tensor` | 72 | 68 | -4 |
| `aten.view.default` | 273 | 257 | -16 |

## Analysis

### Differences present in all models

1. **`_unsafe_view` vs `view`**: non_strict uses more `_unsafe_view`, titan uses `view`. These are semantically equivalent — `_unsafe_view` skips contiguity checks but produces identical results. Titan's dynamo normalizes them to `view`.

2. **`ones_like`**: non_strict has exactly 1 extra across all models. This comes from the explicit `autograd.grad` call in `TrainStepModule` which creates a ones tensor for the initial gradient seed. Titan's aot_autograd handles this internally.

### Differences in MoE/complex models only

3. **In-place → out-of-place functionalization**: Titan's dynamo converts in-place ops to their out-of-place equivalents:
   - `add_` → `add` (deepseek_v3: 5, llama4: 3, gpt_oss: 4)
   - `index_put_` → `index_put` (deepseek_v3: 10, llama4: 6, gpt_oss: 8)
   - `logical_and_` → `logical_and` (gpt_oss: 4)

   This is expected: dynamo's functionalization pass eliminates all mutations for cleaner graph analysis.

4. **`new_zeros` reduction**: Titan has significantly fewer `new_zeros` ops (llama4: -30, gpt_oss: -20). These come from MoE expert routing backward — `autograd.grad` in non_strict materializes gradient accumulators eagerly, while aot_autograd fuses or eliminates them.

5. **`getitem` / `_to_copy` / `slice` / `index`**: Titan has fewer of these auxiliary ops. Dynamo's graph capture is more aggressive at constant-folding and dead-code elimination compared to make_fx's eager tracing.

## Reproduction

```bash
# Single model
torchrun --nproc_per_node=8 compare_graphs.py llama3

# All models
torchrun --nproc_per_node=8 compare_graphs.py

# Output
ls graphs/{non_strict,titan,diff}_*.txt
```
