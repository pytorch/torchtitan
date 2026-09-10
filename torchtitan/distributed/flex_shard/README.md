# FlexShard

FlexShard provides PyTorch-native building blocks for running optimizer compute
with layouts that differ from persistent DTensor parameter storage layouts. It
plans packed storage-to-compute redistribution and overlaps communication with
optimizer work. DistMuon is its initial consumer.

## Public API

The public API is exported from `torchtitan.distributed.flex_shard`:

- `ComputeLayout` describes temporary optimizer-compute sharding on named
  `DeviceMesh` axes using PyTorch DTensor placements plus `BlockShard` or
  `Owned`, and optionally the order in which several axes shard one tensor
  dimension.
- `BlockShard` shards complete fixed-size blocks along one tensor dimension. It
  preserves the tensor's rank and global shape and never creates a tensor view.
- `Owned` assigns a complete subgroup-local logical tensor to one dynamically
  selected rank for the compute phase.
- `BucketConfig` groups and orders parameters by fully qualified name for
  packed redistribution and communication-compute overlap.
- `build_dist_muon` consumes optimizer-agnostic per-parameter `ComputeLayout`
  values in `compute_sharding_by_fqn`. DistMuon's `BlockShard` path accepts
  only a 2D parameter `[M * R, C]` with contiguous local DTensor storage. The
  placement must target tensor dimension 0 with `block_size=R`; the leading
  dimension must be nonzero and divisible by `R`. Each consecutive `R` rows
  forms one independent `[R, C]` matrix. FlexShard routes the flat 2D compute
  tensor, and DistMuon applies a zero-copy local `[M_local, R, C]` view
  immediately before Muon compute. A native batch-first 3D `[M, R, C]`
  parameter uses `Shard(0)` to distribute complete matrices. A single 2D
  matrix without `BlockShard` uses whole-matrix compute such as `Owned`. The
  builder validates named DTensor parameters and plans their storage-to-compute
  transitions.

For HSDP storage such as `(Replicate(), Shard(0))`, declaring `Owned()` on both
`dp_replicate` and `dp_shard` assigns each complete matrix parameter to one
owner in the Cartesian group. Parameters are greedily balanced across owners,
and the existing packed all-to-all path gathers compute inputs and restores
updated storage shards. This is opt-in through `compute_sharding_by_fqn`; model
recipes do not enable it automatically.

A native 3D matrix batch `[M, R, C]` stored as `(Replicate(), Shard(0))`
can instead split each storage shard's matrices over its replica group:

```python
compute_sharding_by_fqn = {
    "experts.weight": ComputeLayout(
        shardings_by_mesh_axis={
            "dp_replicate": Shard(0),
            "dp_shard": Shard(0),
        },
        shard_order_by_tensor_dim={0: ("dp_shard", "dp_replicate")},
    ),
}
optimizer = build_dist_muon(
    [{"params": [weight], "param_names": ["experts.weight"]}],
    compute_sharding_by_fqn=compute_sharding_by_fqn,
    bucket_configs=[BucketConfig(patterns=("experts.weight",))],
)
```

Here `weight` is the HSDP DTensor parameter. The order is outermost first:
`dp_shard` keeps its storage-local matrix batch and `dp_replicate` partitions
that batch into complete matrices. If `dp_shard` already precedes
`dp_replicate` in the storage mesh, the default order suffices. Uneven and empty
matrix batches are supported. The packed redistribution is confined to each
replica group; its inverse sends computed directions to all storage holders.
Parameters and momentum retain their original storage placements.

Replica compute partitioning trades repeated Newton-Schulz work for replica
communication and synchronization. It can be useful when expensive matrices
or matrix batches offer enough work to distribute across replica ranks. It is
not an unconditional speedup: small batches, idle compute ranks, and slow
interconnects can favor duplicated compute. Select it per parameter through
the existing API and measure on the target topology. Buckets requiring
different transport groups must use separate `BucketConfig` entries.

Storage placements describe persistent ownership only; they do not define
Muon matrix boundaries. Flat matrix-batch compute supports `BlockShard` on at
most one non-unit mesh axis. Storage on that axis may use exact `Shard(0)` or
`Replicate`; every other non-unit storage mesh axis must be replicated.

Several mesh axes may shard the same tensor dimension. By default they apply
in storage-mesh order; `shard_order_by_tensor_dim` states a different order,
outermost axis first. For example, preserving an EP-axis `Shard(0)` while
repartitioning its local expert domain over a preceding EFSDP axis uses
`Shard(0)` on both axes with `shard_order_by_tensor_dim={0: ("ep", "efsdp")}`.
FlexShard derives each axis's split factor from the bound mesh, then lowers the
EFSDP placement to subgroup-local `Shard(0)` for optimizer execution.

Compute sharding is construction-time configuration. It is validated and
frozen when the optimizer is built, but is not stored in its state dict;
checkpoint restore must rebuild the optimizer with matching values.

## TorchTitan Kimi integration

The [Kimi configuration registry](../../models/kimi_k2_7/config_registry.py)
is the first TorchTitan integration. Its shared optimizer configuration:

- Selects matrix parameters from attention, dense MLPs, routed and shared
  experts, and routers for DistMuon. Other parameters continue to use
  AdamW.
- Defines each selected parameter's compute layout, including per-head Muon
  for compatible attention projections.
- Groups layers into buckets so compute-ready work can overlap packed
  redistribution.

The configuration is shared by the Kimi-family, Kimi-VL, and Moonlight
recipes in that registry.

## Package boundary

FlexShard currently lives in TorchTitan while its API matures. We intend to
annex this directory into a standalone Python package and repository.

Keep this directory self-contained and PyTorch-only. Dependencies should flow
in one direction:

```text
TorchTitan components and models -> FlexShard -> PyTorch
```

FlexShard must not depend on TorchTitan model or training infrastructure.
