# ParallelDims mesh axes

[`ParallelDims`](parallel_dims.py) builds the `DeviceMesh` views used for
training. Axis names live in `MeshAxisName`. Degrees come from
`ParallelismConfig`.

## Axes

| Axis | Config field | Role |
|------|--------------|------|
| `pp` | `pipeline_parallel_degree` | Pipeline stages |
| `dp_replicate` | `data_parallel_replicate_degree` | DDP / HSDP replicate |
| `dp_shard` | `data_parallel_shard_degree` | FSDP shard on the dense mesh |
| `dp` | derived: `dp_replicate * dp_shard` | Folded DP axis on the dense fwd/bwd mesh |
| `cp` | `context_parallel_degree` | Context (sequence) parallel |
| `tp` | `tensor_parallel_degree` | Tensor parallel |
| `ep` | `expert_parallel_degree` | Expert parallel on the sparse mesh |
| `efsdp` | derived: `dp_shard * cp * tp // ep` | FSDP inside the EP region |

`data_parallel_shard_degree=-1` fills leftover ranks after the other dense
degrees: `world_size // (dp_replicate * cp * tp * pp)`.

## Why EP is not in the world_size product

`ParallelDims` checks

```text
dp_replicate * dp_shard * cp * tp * pp == world_size
```

EP is not a factor because it does not add GPUs. The dense product already
covers every rank. EP is a second unflatten of that same 1D world mesh so
routed experts can regroup those ranks.

## How EP is carved

`build_mesh` starts from a 1D world mesh and unflattens three views (last axis
fastest):

| View | Axes | Degrees |
|------|------|---------|
| Dense storage | `pp, dp_replicate, dp_shard, cp, tp` | config degrees |
| Dense fwd/bwd | `pp, dp, cp, tp` | `dp = dp_replicate * dp_shard` |
| Sparse | `pp, dp_replicate, efsdp, ep` | `efsdp = dp_shard * cp * tp // ep` |

`pp` and `dp_replicate` stay outer on every view. The inner dense region
`dp_shard * cp * tp` is the pool EP borrows from:

```text
ep must divide (dp_shard * cp * tp)
```

Equivalently `dp_shard * cp * tp == efsdp * ep`. Dense modules (attention,
shared experts, embeddings) stay on the dense mesh. Routed experts use the
sparse mesh: expert weights `Shard(0)` on `ep`, leftover FSDP on `efsdp`.
Placements are in [MoE sharding](../models/common/MOE_SHARDING.md).

## Worked example

Take `tp=2`, `cp=2`, `dp_shard=2`, `pp=1`, `dp_replicate=1`. Then
`world_size = 8`. The inner dense region is `2 * 2 * 2 = 8`, so both `ep=4`
and `ep=8` are valid. Size-1 outer axes omitted, ranks `0..7` unflatten as:

Dense `(dp_shard, cp, tp)`:

| Rank | `dp_shard` | `cp` | `tp` |
|------|------------|------|------|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 |
| 2 | 0 | 1 | 0 |
| 3 | 0 | 1 | 1 |
| 4 | 1 | 0 | 0 |
| 5 | 1 | 0 | 1 |
| 6 | 1 | 1 | 0 |
| 7 | 1 | 1 | 1 |

Sparse with `ep=4` (`efsdp = 8 // 4 = 2`):

| Rank | `efsdp` | `ep` |
|------|---------|------|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 2 | 0 | 2 |
| 3 | 0 | 3 |
| 4 | 1 | 0 |
| 5 | 1 | 1 |
| 6 | 1 | 2 |
| 7 | 1 | 3 |

On a Transformer block those views mean:

- Attention (dense mesh): FSDP on `dp_shard=2` (pairs `(0,4)`, `(1,5)`,
  `(2,6)`, `(3,7)`), CP splits the sequence on `cp=2` (pairs `(0,2)`,
  `(1,3)`, `(4,6)`, `(5,7)`), TP shards projections on `tp=2` (pairs
  `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`).
- Routed experts with `ep=4`: 4-way expert parallel on each `efsdp` slice
  (`(0,1,2,3)` and `(4,5,6,7)`); leftover FSDP is `efsdp=2` (same pairs as
  dense `dp_shard` in this example).
- Routed experts with `ep=8`: `efsdp=1`, so every rank holds a different
  expert shard and there is no leftover FSDP in the EP region. Attention is
  unchanged; only the sparse view regroups.

Same eight GPUs in every case. Raising EP does not grow `world_size`.
