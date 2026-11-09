# ParallelismContext mesh axes

[`ParallelismContext`](parallelism_context.py) builds the `DeviceMesh` views used for
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
| `edp_replicate` | `expert_data_parallel_replicate_degree` (default: `dp_replicate`) | HSDP replicate for routed experts |
| `edp_shard` | derived: `K * dp_shard * cp * tp // ep`, `K = dp_replicate // edp_replicate` | FSDP inside the EP region |

`data_parallel_shard_degree=-1` fills leftover ranks after the other dense
degrees: `world_size // (dp_replicate * cp * tp * pp)`.

## Why EP is not in the world_size product

`ParallelismContext` checks

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
| Sparse | `pp, edp_replicate, edp_shard, ep` | `edp_shard = K * dp_shard * cp * tp // ep` |

`pp` stays outer on every view. By default `edp_replicate == dp_replicate`
(`K = 1`), so the replicate axis is outer too and the inner dense region
`dp_shard * cp * tp` is the pool EP borrows from:

```text
ep must divide (dp_shard * cp * tp)
```

Equivalently `dp_shard * cp * tp == edp_shard * ep` when `K = 1`. Dense modules (attention,
shared experts, embeddings) stay on the dense mesh. Routed experts use the
sparse mesh: expert weights `Shard(0)` on `ep`, leftover FSDP on `edp_shard`.
Placements are in [MoE sharding](../models/common/moe/SHARDING.md).

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

Sparse with `ep=4` (`edp_shard = 8 // 4 = 2`):

| Rank | `edp_shard` | `ep` |
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
- Routed experts with `ep=4`: 4-way expert parallel on each `edp_shard` slice
  (`(0,1,2,3)` and `(4,5,6,7)`); leftover FSDP is `edp_shard=2` (same pairs as
  dense `dp_shard` in this example).
- Routed experts with `ep=8`: `edp_shard=1`, so every rank holds a different
  expert shard and there is no leftover FSDP in the EP region. Attention is
  unchanged; only the sparse view regroups.

Same eight GPUs in every case. Raising EP does not grow `world_size`.

## Fewer expert replicas than dense replicas

In a large MoE, one copy of the dense parameters may fit inside one scale-up
domain while one copy of the routed experts needs several. Setting
`expert_data_parallel_replicate_degree` below `data_parallel_replicate_degree`
lets `K = dp_replicate // edp_replicate` consecutive dense replicas share one
copy of the routed experts. EP groups stay inside one dense replica, as with
`K = 1`; only expert FSDP (`edp_shard`) grows by `K` and spans those replicas.

Take `dp_replicate=4`, `dp_shard=4`, `ep=4`, `edp_replicate=2` (`cp = tp = pp
= 1`, 16 ranks). Then `K = 2` and `edp_shard = 2 * 4 // 4 = 2`:

```text
ranks         0  1  2  3 | 4  5  6  7 | 8  9 10 11 |12 13 14 15
dense         replica 0  | replica 1  | replica 2  | replica 3     dp_shard=4 inside each
EP groups     ep 0..3    | ep 0..3    | ep 0..3    | ep 0..3       one EP group per dense replica
expert copy   copy 0 (ranks 0-7)      | copy 1 (ranks 8-15)        edp_shard pairs: (0,4) (1,5) ...
```

Rank 0 reduce-scatters dense gradients over `(0, 1, 2, 3)` and all-reduces
them over `(0, 4, 8, 12)`. It reduce-scatters routed-expert gradients over
`(0, 4)` and all-reduces them over `(0, 8)`. Every rank still processes
different data, and each expert gradient is still summed over the same set of
ranks as with `K = 1`; only the split between reduce-scatter and all-reduce
changes. With `ep = 1`, routed experts use the dense FSDP mesh, so
`edp_replicate` must equal `dp_replicate`.

## Saving the layout

Set `config.parallelism.save_parallelism_folder = "parallelism"` in the recipe
to have `ParallelismContext.from_config` build the meshes and have every rank
`r` write `<dump_folder>/parallelism/rank_<r>.json`. All ranks write the same
fields and no collective is involved. For example, rank 5 with `pp=2`,
`dp_shard=2`, `cp=2`, `ep=2` (indented here for reading):

```json
{
  "host": "node-b",
  "local_rank": 1,
  "global_rank": 5,
  "world_size": 8,
  "degrees": {"pp": 2, "dp_replicate": 1, "dp_shard": 2, "cp": 2, "tp": 1, "ep": 2, "edp_replicate": 1},
  "meshes": {
    "loss": {"axis_names": ["loss_mesh"], "mesh": [4, 5, 6, 7]},
    "dense": {
      "axis_names": ["pp", "dp_replicate", "dp_shard", "cp", "tp"],
      "mesh": [[[[[0], [1]], [[2], [3]]]], [[[[4], [5]], [[6], [7]]]]]
    },
    ...
  }
}
```

`meshes` holds this rank's view of every mesh in `_global_meshes`: the dense
storage and sparse views above, and the `loss`, `spmd_dense_for_fwdbwd` and
`spmd_sparse_for_fwdbwd` meshes sliced from them. A full mesh like `dense` is
the same in every file. A sliced mesh only holds the submesh containing the
rank, so `loss` above is `[4, 5, 6, 7]` on ranks 4-7 and `[0, 1, 2, 3]` on
ranks 0-3; reading all files gives every submesh. A mesh is the nested list of
global ranks; the ranks that share a group along an axis are the ones that
differ only in that axis's coordinate.
