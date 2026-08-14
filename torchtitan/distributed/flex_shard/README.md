# FlexShard

FlexShard provides PyTorch-native building blocks for running optimizer compute
with layouts that differ from persistent DTensor parameter storage layouts. It
plans packed storage-to-compute redistribution and overlaps communication with
optimizer work. DistMuon is its initial consumer.

## Public API

The public API is exported from `torchtitan.distributed.flex_shard`:

- `ComputeLayout` describes temporary optimizer-compute sharding on named
  `DeviceMesh` axes using PyTorch DTensor placements or `Owned`.
- `Owned` assigns a complete subgroup-local logical tensor to one dynamically
  selected rank for the compute phase.
- `BucketConfig` groups and orders parameters by fully qualified name for
  packed redistribution and communication-compute overlap.
- `build_dist_muon` combines optimizer-agnostic per-parameter
  `ComputeLayout` values in `compute_sharding_by_fqn` with optimizer-owned
  `num_stacked_matrices_by_fqn` counts. Each configured count interprets a 2D
  parameter `[M * R, C]` as `M` independent matrices `[M, R, C]`. The builder
  validates named DTensor parameters and plans their storage-to-compute
  transitions. An absent FQN uses ordinary 2D Muon compute; an explicit count
  of one uses the stacked `[1, R, C]` compute path.

Compute sharding and stacked-matrix counts are reconstruction configuration.
They are validated and frozen when the optimizer is built, but are not stored
in its state dict; checkpoint restore must rebuild the optimizer with matching
values.

## TorchTitan Kimi integration

The [Kimi configuration registry](../../models/kimi_k2_7/config_registry.py)
is the first TorchTitan integration. Its shared optimizer configuration:

- Selects matrix parameters from attention, dense MLPs, routed and shared
  experts, and routers for DistMuon. Other parameters continue to use
  AdamW.
- Defines each selected parameter's compute sharding, including per-head Muon
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
