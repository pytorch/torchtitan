# Enable FlexShard for TorchTitan Eager Training

## Goal

Enable an experimental TorchTitan training entry point that uses FlexShard for
data-parallel parameter sharding, with a first target of eager MoE training on:

- `data_parallel_shard_degree=8`
- `expert_parallel_degree=4`
- `tensor_parallel_degree=1`
- `pipeline_parallel_degree=1`
- eager model execution

The PR should also provide profiler commands that dump comparable traces for the
existing `fully_shard` path and the new FlexShard path.

## Design

Keep the integration under `torchtitan/experiments/flex_shard` instead of adding
FlexShard-specific branches to core model code. The experimental module should
reuse the existing DeepSeek V3 model config and swap only the model
parallelization function.

The FlexShard path should:

1. Apply context-parallel setup only when supported.
2. Reject model compile, TP, PP, and CPU offload for the initial eager path.
3. Apply expert parallelism without converting expert parameters to DTensor,
   because FlexShard eager currently manages plain parameters only.
4. Shard dense parameters on the `fsdp` mesh.
5. Shard local expert parameters on the `efsdp` mesh after EP slices the expert
   dimension.
6. Use FlexShard mixed precision policy for forward parameter casts and gradient
   reduction casts.
7. Preserve the existing `fully_shard` path unchanged.

## Required Runtime Fix

TorchTitan constructs models on `meta`, calls the model parallelization function,
then calls `to_empty()` and initializes weights. FlexShard currently materializes
bucket storage and installs eager CUDA hooks during `flex_shard()`, which works
for already-materialized test models but not for normal TorchTitan training.

The formal fix is to make FlexShard meta-safe:

1. Allow `flex_shard()` to create bucket metadata and sharded meta parameter
   views when the input module is on `meta`.
2. Defer eager bucket hook installation until bucket storage is materialized on
   CUDA.
3. Override `to_empty()` for FlexShard-managed modules so bucket byte storage is
   materialized as a single CUDA allocation per bucket.
4. Reinstall each sharded parameter as a view into its bucket storage after
   `to_empty()`.
5. Install eager bucket hooks exactly once after materialization.

## Profiling Commands

Fully-shard baseline:

```bash
NGPU=8 MODULE=deepseek_v3 CONFIG=deepseek_v3_debugmodel_ep ./run_train.sh \
  --parallelism.data_parallel_shard_degree=8 \
  --parallelism.expert_parallel_degree=4 \
  --training.steps=8 \
  --activation_checkpoint.mode=none \
  --profiler.enable_profiling \
  --profiler.profile_freq=4 \
  --profiler.profiler_warmup=1 \
  --profiler.profiler_active=1 \
  --profiler.save_traces_folder=profiling/traces_fully_shard \
  --dump_folder=outputs/profile_fully_shard_dp8_ep4
```

FlexShard candidate:

```bash
NGPU=8 MODULE=flex_shard.deepseek_v3 CONFIG=flex_shard_deepseek_v3_debugmodel_dp8_ep4 ./run_train.sh \
  --training.steps=8 \
  --activation_checkpoint.mode=none \
  --profiler.enable_profiling \
  --profiler.profile_freq=4 \
  --profiler.profiler_warmup=1 \
  --profiler.profiler_active=1 \
  --profiler.save_traces_folder=profiling/traces_flex_shard \
  --dump_folder=outputs/profile_flex_shard_dp8_ep4
```

Profiler output is written to:

```text
<dump_folder>/<save_traces_folder>/iteration_<step>/rank<rank>_trace.json
```

## Validation

Run focused FlexShard tests:

```bash
python -m pytest -q torchtitan/experiments/flex_shard/tests/test_flex_shard_runtime.py
python -m pytest -q torchtitan/experiments/flex_shard/tests/test_flex_shard_buckets.py::TestMultiMeshBuckets
```

Run an 8-GPU smoke profile for both paths and compare:

- trace files exist for all ranks
- FlexShard ranges are present in traces
- training reaches the requested step count
- loss is finite
- communication shape matches dense-on-`fsdp`, experts-on-`efsdp`

Numerical proof should be collected separately with `--debug.seed=42` and
`--debug.deterministic` once the eager path is stable.
