# FlexShard Eager-Only Plan

## Goal

Land a minimal FlexShard primitive that works for eager execution and fails
explicitly for graph capture paths. The first version should prove the eager
runtime without carrying TorchTitan trainer integration, graph capture,
precompile, or multi-mesh composition support.

## Supported

- Eager training only.
- Self-contained FlexShard API tests on a CPU `fsdp` mesh.
- Data-parallel sharding through a named `fsdp` mesh.
- `Shard(0)` per-parameter placement.
- Explicit `BucketSpec` coverage.
- Eager forward/backward through the FlexShard parametrization and hooks.

## Explicitly Unsupported

- `torch.compile` and graph capture.
- TorchTitan trainer/model registration.
- Precompile artifact save/load.
- Tensor, context, pipeline, expert, and hybrid data parallel composition.
- CPU offload.
- User-selectable `FlatShard`, `Owned`, or `RaggedShard` placement policies.
- Checkpoint save/load for FlexShard models.
- FlexShard graph passes and trace/precompile tests.

## Implementation Steps

1. Keep FlexShard as a library-only experiment under
   `torchtitan/experiments/flex_shard`.
2. Do not register a TorchTitan trainer module or config in this PR.
3. Add a FlexShard-level guard that raises `ValueError` during graph capture.
4. Remove graph-pass, trace, precompile, and trainer integration from this PR.
5. Add a self-contained CPU unit test that initializes a single-rank `fsdp`
   mesh, applies `flex_shard()` to a tiny module, verifies eager
   forward/backward, and verifies graph capture raises.

## Validation

Run syntax and targeted unit checks locally:

```bash
python -m py_compile \
  torchtitan/experiments/flex_shard/flex_shard.py \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_eager_only.py

pytest torchtitan/experiments/flex_shard/tests/test_flex_shard_eager_only.py -q
```

Distributed GPU validation can continue to use the experiment-local tests when
GPUs are available:

```bash
torchrun --nproc_per_node=2 \
  torchtitan/experiments/flex_shard/tests/test_flex_shard.py
```
