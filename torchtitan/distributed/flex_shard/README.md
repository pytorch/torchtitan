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
- `ComputeLayout.compute_dtype` is the dtype the optimizer computes in for
  that parameter; `None` keeps the storage dtype. DistMuon runs Newton-Schulz
  in BF16 and accepts only `torch.bfloat16`, so every FP32 parameter sets it.
  Inputs are sent to the compute ranks and directions are sent back in this
  dtype.
- `BlockShard` shards complete contiguous blocks along one tensor dimension. It
  preserves the tensor's rank and global shape and never creates a tensor view.
- `Owned` assigns a complete subgroup-local logical tensor to one dynamically
  selected rank for the compute phase.
- `BucketConfig` groups and orders parameters by fully qualified name for
  packed redistribution and communication-compute overlap.
- `BlockShard.block_sizes` is a nonempty tuple describing a repeating sequence
  of independently shardable block sizes. Use `(R,)` for uniform blocks.
  For example, `(128, 64)` partitions
  384 rows into four blocks of 128, 64, 128, and 64 rows. Blocks are distributed
  by count, so ranks can own different numbers of rows. DistMuon runs
  Newton-Schulz and the aspect-ratio learning-rate adjustment independently
  for each block. Kimi's shared `wkv_a` projection uses `(512, 64)` to distribute
  its KV latent and RoPE key matrices separately.
- `DistMuon` consumes optimizer-agnostic per-parameter `ComputeLayout`
  values in `compute_sharding_by_fqn`. DistMuon's `BlockShard` path accepts
  only a 2D parameter with contiguous local DTensor storage. The placement
  must target tensor dimension 0; the leading dimension must be nonzero and
  divisible by the sum of `block_sizes`.
  With `block_sizes=(R,)`, each consecutive `R` rows forms one independent
  `[R, C]` matrix. Multiple block sizes allow these matrices to have different
  row counts. FlexShard routes the flat 2D compute tensor using the block
  boundaries. DistMuon's planning code constructs zero-copy strided
  `[M, R, C]` views directly from each rank's compute shape and logical starting
  row. For `(R,)`, the view is equivalent to unflattening the row dimension.
  A native matrix batch `[..., R, C]` uses `Shard(0)` to distribute its
  outermost batch dimension, or `Owned` to assign the complete batch to one
  rank. A single 2D matrix without
  `BlockShard` uses whole-matrix compute such as `Owned`. The builder validates
  named DTensor parameters and plans their storage-to-compute transitions.

Storage placements describe persistent ownership only; they do not define
Muon matrix boundaries. Flat matrix-batch compute may use `BlockShard` on one
or more non-unit mesh axes. Callers supply a mesh combining those axes into
one optimizer transport group. FlexShard routes complete matrices across it and restores the
original DTensor storage layout after the update. This supports combined DP
and TP storage, including the strided shards produced when TP and FSDP shard
the same tensor dimension. Non-transport mesh axes retain their storage
placement.

Native `[..., R, C]` parameters can redistribute `Replicate()`, a shard of the
matrix-row dimension, or a shard of the matrix-column dimension to `Shard(0)`
compute on one mesh axis, with every other non-unit storage mesh axis
preserving its existing batch ownership. Changed matrix axes may also combine
into one transport group; complete matrices are assigned across that group.
For example, expert storage `Shard(0), Shard(1)` on DP/TP preserves each DP
rank's expert range while TP redistributes matrix rows to whole-expert compute.

`redistribution_mesh_axes(fqn, parameter, compute_layout)` queries the exact
transport axes without communication. The caller creates the transport mesh
and supplies it through `BucketConfig.mesh`; FlexShard validates its axis names,
ranks, and logical rank order, then reuses the bound groups for initialization,
updates, and checkpoint loading. It does not construct communicators or choose
process-group options. A single-axis bucket with `mesh=None` uses the existing
storage group. A combined transport requires an explicit 1D mesh with the
joined name, such as `dp_shard_tp`.

For example, create a whole-matrix DP/TP transport before constructing DistMuon:

```python
from torchtitan.distributed.utils import build_transport_mesh

axes = redistribution_mesh_axes("matrix", parameter, compute_layout)
transport_mesh = build_transport_mesh(parameter.device_mesh, axes)
bucket = BucketConfig(patterns=("matrix",), mesh=transport_mesh)
```

The TorchTitan DistMuon wrapper builds these meshes before calling FlexShard.
Its existing `max_nccl_ctas_for_comms` cap applies only to those optimizer
communicators; FSDP continues to use the original groups. The builder lives in
`torchtitan/distributed/utils.py` and uses native `DeviceMesh._flatten` caching,
with a separate resource-independent NCCL communicator for each positive cap.
Callers may instead supply a precreated mesh with their own process-group
options; the wrapper requires its automatic cap to be disabled in that case.

First native group creation requires all WORLD ranks to request the same axes
and cap in the same order. The wrapper calls `precreate_transport_meshes` on a
common transport union before binding local buckets, so uneven PP buckets do
not desynchronize group creation. This requires the same wrapper-constructor
sequence and WORLD-covering roots with the requested storage views on every
rank. VPP chunks reuse the native cached groups. For unequal optimizer counts,
missing expert views, partial roots, or unsorted views, the caller must create
meshes collectively and supply explicit `BucketConfig.mesh` values. Direct
FlexShard constructors perform no WORLD transport preparation.

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
