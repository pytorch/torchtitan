# FlexShard

FlexShard is an experimental eager data-parallel sharding primitive. This first
version is intentionally narrow: it keeps the library runtime usable while
graph capture, trainer integration, precompile, checkpoint, and multi-mesh
paths are out of scope.

For the support matrix and rollout plan, see
[eager_only_plan.md](eager_only_plan.md).

## Supported Path

The current supported configuration is direct `flex_shard()` use:

- eager execution only
- data-parallel sharding on a named `fsdp` mesh
- `Shard(0)` per-parameter placement
- explicit `BucketSpec` coverage
- eager forward/backward through FlexShard parametrization and hooks

## Unsupported Paths

FlexShard raises `ValueError` for:

- `torch.compile` and graph capture

This PR does not register a TorchTitan trainer/model module, so the following
paths are intentionally not exposed yet:

- precompile artifacts
- checkpoint save/load
- tensor, context, pipeline, expert, and hybrid data parallel composition
- CPU offload

The lower-level prototype still contains placement abstractions used by the
runtime, but the model integration does not expose user-selectable placement
policies yet.

## API Shape

Call `flex_shard()` directly with explicit buckets:

```python
from torch.distributed.fsdp import DataParallelMeshDims

from torchtitan.experiments.flex_shard import (
    BucketSpec,
    flex_shard,
    MixedPrecisionPolicy,
    per_param_placements,
)

flex_shard(
    model,
    fsdp_mesh,
    DataParallelMeshDims(shard="fsdp"),
    shard_placement_fn=per_param_placements,
    buckets=[
        BucketSpec(["embed.*"], mp_policy=mp_policy),
        BucketSpec(["layers.*"], mp_policy=mp_policy),
        BucketSpec(["output.*"], mp_policy=mp_policy),
    ],
)
```

`flex_shard()` mutates the module in place, replaces managed parameters with
sharded local tensors, stores per-bucket `DStorage` objects on the module, and
installs eager all-gather/reduce-scatter hooks for forward/backward execution.

## Validation

CPU checks:

```bash
python -m py_compile \
  torchtitan/experiments/flex_shard/flex_shard.py \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_eager_only.py

pytest torchtitan/experiments/flex_shard/tests/test_flex_shard_eager_only.py -q
```

GPU checks:

```bash
torchrun --nproc_per_node=2 \
  torchtitan/experiments/flex_shard/tests/test_flex_shard.py
```
