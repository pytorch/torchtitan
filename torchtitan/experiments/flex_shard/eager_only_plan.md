# FlexShard Eager-Only Plan

## Goal

Land a minimal FlexShard experiment that works for eager training and fails
explicitly for compile paths. The first version should prove the eager runtime
without carrying graph capture, precompile, or multi-mesh composition support.

## Supported

- Llama 3 FlexShard experiment.
- Eager training only.
- Data-parallel sharding through the `fsdp` mesh.
- `Shard(0)` per-parameter placement.
- One communication bucket for embeddings, one per transformer block, and one
  for final norm/lm head.
- Mixed-precision parameter and reduction dtypes.
- Eager `reshard_after_forward` policy through the existing FSDP policy knob.

## Explicitly Unsupported

- `compile.mode` values `jit`, `aot`, and `aot_fx_trace`.
- Precompile artifact save/load.
- Tensor, context, pipeline, expert, and hybrid data parallel composition.
- CPU offload.
- User-selectable `FlatShard`, `Owned`, or `RaggedShard` placement policies.
- Checkpoint save/load for FlexShard models.
- FlexShard graph passes and trace/precompile tests.

## Implementation Steps

1. Keep the base `Trainer` hook points for model build and initialization device.
2. Move CPU model build and FlexShard buffer initialization into a
   FlexShard-specific `GraphTrainer` subclass.
3. Set FlexShard config defaults to eager by using `compile.mode=None` and
   `compile.enable=False`.
4. Add early validation that raises `ValueError` for compile/precompile,
   checkpointing, unsupported parallelisms, CPU offload, and non-FSDP sharding.
5. Simplify `parallelize_llama_flex_shard()` to apply only eager activation
   checkpointing and FlexShard sharding.
6. Remove graph-pass, trace, and precompile integration from this PR.
7. Add a small CPU unit test for the eager-only guard.

## Validation

Run syntax and targeted unit checks locally:

```bash
python -m py_compile \
  torchtitan/experiments/flex_shard/flex_shard.py \
  torchtitan/experiments/graph_trainer/flex_shard_llama3/*.py \
  torchtitan/trainer.py

pytest tests/unit_tests/test_flex_shard_eager_only.py -q
```

GPU validation, when GPUs are available:

```bash
NGPU=4 MODULE=graph_trainer.flex_shard_llama3 \
  CONFIG=graph_trainer_flex_shard_llama3_debugmodel \
  ./run_train.sh --training.steps 5 --compile.mode None

NGPU=4 MODULE=graph_trainer.flex_shard_llama3 \
  CONFIG=graph_trainer_flex_shard_llama3_debugmodel \
  ./run_train.sh --training.steps 5 --compile.mode None \
  --parallelism.fsdp_reshard_after_forward never
```
