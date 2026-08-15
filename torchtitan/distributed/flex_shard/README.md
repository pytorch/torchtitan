# FlexShard

FlexShard provides PyTorch-native building blocks for running optimizer compute
with layouts that differ from persistent DTensor parameter storage layouts. It
plans packed storage-to-compute redistribution and overlaps communication with
optimizer work. Its Muon integration configures an exact `torch.optim.Muon`
instance through supported PyTorch execution and tensor-operation APIs.

## Public API

The public API is exported from `torchtitan.distributed.flex_shard`:

- `ComputeLayout` describes temporary optimizer-compute sharding on named
  `DeviceMesh` axes using PyTorch DTensor placements, `BlockShard`, or `Owned`.
- `BlockShard` shards complete fixed-size blocks along one tensor dimension. It
  preserves the tensor's rank and global shape and never creates a tensor view.
- `Owned` assigns a complete subgroup-local logical tensor to one dynamically
  selected rank for the compute phase.
- `BucketConfig` groups and orders parameters by fully qualified name for
  packed redistribution and communication-compute overlap.
- `flex_optimizer_reshard` binds a supported optimizer to its per-parameter
  compute shardings and physical buckets without replacing the optimizer
  object, its exact type, or its declared `step` method.
- `build_flex_shard_muon` is the convenience construction path. It constructs
  an exact `torch.optim.Muon` and binds optimizer-agnostic per-parameter
  `ComputeLayout` values in `compute_sharding_by_fqn` through
  `flex_optimizer_reshard`.
- `BlockShard(dim=0, block_size=R)` interprets a 2D parameter
  `[M * R, C]` as `M` independent matrices `[M, R, C]`. FlexShard routes the
  flat 2D compute tensor, and the private Muon integration applies the local
  matrix-batch view immediately before Muon compute. A native 3D `[M, R, C]`
  parameter can use `Shard(0)` to distribute complete matrices. A 2D parameter
  without `BlockShard` must use whole-matrix compute such as `Owned`.

Storage placements describe persistent ownership only; they do not define
Muon matrix boundaries. Flat matrix-batch compute supports `BlockShard` on at
most one non-unit mesh axis. Storage on that axis may use exact `Shard(0)` or
`Replicate`; every other non-unit storage mesh axis must be replicated.

Compute sharding is binding-time configuration. It is validated and frozen by
`flex_optimizer_reshard`, but is not stored in the optimizer state dict;
checkpoint restore must rebuild and bind the optimizer with matching values.

## TorchTitan Kimi integration

The [Kimi configuration registry](../../models/kimi_k2_7/config_registry.py)
is the first TorchTitan integration. Its shared optimizer configuration:

- Selects matrix parameters from attention, dense MLPs, routed and shared
  experts, and routers for Muon. Other parameters continue to use AdamW.
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
