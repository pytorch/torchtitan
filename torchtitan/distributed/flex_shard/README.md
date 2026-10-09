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
- `BlockShard` shards complete contiguous matrix blocks by element count. It
  preserves the tensor's rank and global shape.
- `Owned` assigns a complete subgroup-local logical tensor to one dynamically
  selected rank for the compute phase.
- `BucketConfig` groups and orders parameters by fully qualified name for
  packed redistribution and communication-compute overlap.
- `BlockShard.block_sizes` is a nonempty tuple describing a repeating sequence
  of independently shardable matrix element counts. Use `(R * C,)` for uniform
  `[R, C]` matrices. For example, `(128 * C, 64 * C)` partitions contiguous
  storage into alternating `[128, C]` and `[64, C]` matrices. Blocks are
  distributed by count, so ranks can own different amounts of storage.
  DistMuon runs Newton-Schulz and the aspect-ratio learning-rate adjustment
  independently for each block. Kimi's shared `wkv_a` projection uses
  `(512 * C, 64 * C)` to distribute its KV latent and RoPE key matrices
  separately.
- `DistMuon` consumes optimizer-agnostic per-parameter `ComputeLayout`
  values in `compute_sharding_by_fqn`. DistMuon's `BlockShard` path accepts
  a contiguous 2D matrix or 3D matrix batch. Every block size must be divisible
  by the parameter's matrix-column count. For flat 2D storage, multiple block
  sizes allow matrices with different row counts. For a native `[B, R, C]`
  batch, `block_sizes=(R * C,)` assigns each complete matrix as one block. The
  same layout also represents a single `[R, C]` matrix without a separate
  `Owned` configuration. DistMuon constructs zero-copy matrix views after each
  rank's compute ownership is known. The builder validates named DTensor
  parameters and plans their storage-to-compute transitions.

Storage placements describe persistent ownership only; they do not define
Muon matrix boundaries. Matrix-block compute supports `BlockShard` on at most
one non-unit mesh axis. Flat 2D storage on that axis may use exact `Shard(0)` or
`Replicate`; native 3D storage may shard any tensor dimension. Every other
non-unit storage mesh axis must be replicated.

Native `[..., R, C]` parameters can redistribute `Replicate()`, a shard of the
matrix-row dimension, or a shard of the matrix-column dimension to `Shard(0)`
compute on one mesh axis, with every other non-unit storage mesh axis
replicated.

Several mesh axes may shard the same tensor dimension. By default they apply
in storage-mesh order; `shard_order_by_tensor_dim` states a different order,
outermost axis first. For example, preserving an EP-axis `Shard(0)` while
repartitioning its local expert domain over a preceding `edp_shard` axis uses
`Shard(0)` on both axes with `shard_order_by_tensor_dim={0: ("ep", "edp_shard")}`.
FlexShard derives each axis's split factor from the bound mesh, then lowers the
`edp_shard` placement to subgroup-local `Shard(0)` for optimizer execution.

Compute sharding is construction-time configuration. It is validated and
frozen when the optimizer is built, but is not stored in its state dict;
checkpoint restore must rebuild the optimizer with matching values.

## TorchTitan Kimi integration

The [Kimi test recipes](../../../torchtitan_recipes/tests/models/kimi_k2_7.py)
provide the first TorchTitan integration. Their shared optimizer configuration:

- Selects matrix parameters from attention, dense MLPs, routed and shared
  experts, and routers for DistMuon. Other parameters continue to use
  AdamW.
- Defines each selected parameter's compute layout, including per-head Muon
  for compatible attention projections.
- Groups layers into buckets so compute-ready work can overlap packed
  redistribution.

The same construction can be used by Kimi-family, Kimi-VL, and Moonlight
recipes.

## Package boundary

FlexShard currently lives in TorchTitan while its API matures. We intend to
annex this directory into a standalone Python package and repository.

Keep this directory self-contained and PyTorch-only. Dependencies should flow
in one direction:

```text
TorchTitan components and models -> FlexShard -> PyTorch
```

FlexShard must not depend on TorchTitan model or training infrastructure.
