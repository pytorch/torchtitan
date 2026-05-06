# FlexShard

FlexShard is an experimental eager data-parallel sharding path. This first
version is intentionally narrow: it keeps the eager runtime usable while
compile, precompile, checkpoint, and multi-mesh paths fail explicitly.

For the support matrix and rollout plan, see
[eager_only_plan.md](eager_only_plan.md).

## Supported Path

Use the Llama 3 experiment module:

```bash
NGPU=4 MODULE=graph_trainer.flex_shard_llama3 \
  CONFIG=graph_trainer_flex_shard_llama3_debugmodel \
  ./run_train.sh --compile.mode None
```

The current supported configuration is:

- eager execution only
- FSDP-style data-parallel sharding on the `fsdp` mesh
- `Shard(0)` per-parameter placement
- one bucket for token embeddings, one bucket per transformer block, and one
  bucket for final norm/lm head
- mixed-precision parameter and reduction dtypes
- eager activation checkpointing through the normal Trainer path
- eager `reshard_after_forward` through
  `--parallelism.fsdp_reshard_after_forward`

## Unsupported Paths

FlexShard raises `ValueError` for:

- `compile.mode` values `jit`, `aot`, and `aot_fx_trace`
- precompile artifacts
- checkpoint save/load
- tensor, context, pipeline, expert, and hybrid data parallel composition
- CPU offload

The lower-level prototype still contains placement abstractions used by the
runtime, but the model integration does not expose user-selectable placement
policies yet.

## API Shape

The model integration calls `flex_shard()` directly with explicit buckets:

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
        BucketSpec(["tok_embeddings.*"], mp_policy=mp_policy),
        *[
            BucketSpec([f"layers.{i}.*"], mp_policy=mp_policy)
            for i in range(len(model.layers))
        ],
        BucketSpec(["norm.*", "lm_head.*", "output.*"], mp_policy=mp_policy),
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
  torchtitan/experiments/graph_trainer/flex_shard_llama3/*.py \
  torchtitan/trainer.py

pytest tests/unit_tests/test_flex_shard_eager_only.py -q
```

GPU checks:

```bash
NGPU=4 MODULE=graph_trainer.flex_shard_llama3 \
  CONFIG=graph_trainer_flex_shard_llama3_debugmodel \
  ./run_train.sh --training.steps 5 --compile.mode None

NGPU=4 MODULE=graph_trainer.flex_shard_llama3 \
  CONFIG=graph_trainer_flex_shard_llama3_debugmodel \
  ./run_train.sh --training.steps 5 --compile.mode None \
  --parallelism.fsdp_reshard_after_forward never
```
