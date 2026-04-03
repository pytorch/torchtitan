# GraphTrainer Development Guide

This supplements the root `.claude/CLAUDE.md` which covers core torchtitan
conventions (code style, naming, testing, PR expectations, etc.). Rules there
apply here too unless overridden below.

## Graph Pass Signature

All graph passes must follow the signature:
```python
def my_pass(gm: torch.fx.GraphModule, example_inputs, *, other_kwargs) -> torch.fx.GraphModule:
```
The first two positional args are always `(gm, example_inputs)`. Any additional
parameters must be keyword-only. The pass must return the (possibly transformed)
`GraphModule`.

## Don't Modify Core for This Experiment

Do not add `if graph_trainer:` branches to `torchtitan/train.py`
or other core files. GraphTrainer extends `Trainer` and overrides behavior through
subclassing.


### Local development (debug models, 8 GPUs)

```bash
# Llama3 with FSDP + TP
NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_debugmodel ./run_train.sh \
    --compile.mode aot_fx_trace \
    --parallelism.data_parallel_shard_degree=4 \
    --parallelism.tensor_parallel_degree=2

# DeepSeek-v3 with FSDP + TP + EP (requires H100)
NGPU=8 MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_debugmodel ./run_train.sh \
    --compile.mode aot_fx_trace \
    --parallelism.data_parallel_shard_degree=4 \
    --parallelism.tensor_parallel_degree=2 \
    --parallelism.expert_parallel_degree=4 \
    --parallelism.expert_tensor_parallel_degree=1
```

### Benchmark

Use `benchmark.py` to measure forward_backward_step performance (no optimizer
step). Reports mean step time, peak memory, TFLOPS, and MFU.

```bash
# Llama3 8B eager baseline (8×H100, FSDP+TP)
torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    -m torchtitan.experiments.graph_trainer.benchmark \
    --module graph_trainer.llama3 --config graph_trainer_llama3_8b \
    --compile.no-enable \
    --parallelism.data_parallel_shard_degree=4 \
    --parallelism.tensor_parallel_degree=2

# Llama3 8B aot_fx_trace (8×H100, FSDP+TP)
torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    -m torchtitan.experiments.graph_trainer.benchmark \
    --module graph_trainer.llama3 --config graph_trainer_llama3_8b \
    --compile.mode aot_fx_trace \
    --parallelism.data_parallel_shard_degree=4 \
    --parallelism.tensor_parallel_degree=2

# DeepSeek-v3 16B aot_fx_trace (8×H100, FSDP+TP+EP)
torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    -m torchtitan.experiments.graph_trainer.benchmark \
    --module graph_trainer.deepseek_v3 --config graph_trainer_deepseek_v3_16b \
    --compile.mode aot_fx_trace \
    --parallelism.data_parallel_shard_degree=4 \
    --parallelism.tensor_parallel_degree=2 \
    --parallelism.expert_parallel_degree=2
```

Benchmark-specific flags (parsed before the torchtitan config):
- `--warmup_steps N` (default 3): steps to warm up before timing
- `--benchmark_steps N` (default 10): steps to time
- `--torch_profiler`: capture a single-step chrome trace after benchmarking

### Profiling

Add `--torch_profiler` to any benchmark command to capture a per-rank chrome
trace of a single forward-backward step. Traces go to
`outputs/benchmark_traces/`.

```bash
torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    -m torchtitan.experiments.graph_trainer.benchmark \
    --module graph_trainer.llama3 --config graph_trainer_llama3_8b \
    --compile.mode aot_fx_trace \
    --parallelism.data_parallel_shard_degree=4 \
    --parallelism.tensor_parallel_degree=2 \
    --torch_profiler
```

For full training profiling, add `--profiling.enable_profiling` and/or
`--profiling.enable_memory_snapshot` to any `run_train.sh` command.
Traces go to `outputs/profile_traces/`, memory snapshots to
`outputs/memory_snapshot/`.

### Tests

```bash
# Unit tests (GPU)
pytest torchtitan/experiments/graph_trainer/tests/test_passes.py -x
pytest torchtitan/experiments/graph_trainer/tests/test_precompile.py -x
pytest torchtitan/experiments/graph_trainer/tests/test_trace_module.py -x
pytest torchtitan/experiments/graph_trainer/tests/test_numerics.py -x
pytest torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py -x

# Integration tests (8 GPUs)
python torchtitan/experiments/graph_trainer/tests/integration_tests.py <output_dir> \
    --test_suite graph_trainer_default --ngpu 8
```

### Bitwise Deterministic Guardrail

Before submitting any change, run the bitwise deterministic test first:
```bash
pytest torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py -x
```
This verifies that the aot_fx_trace path produces bitwise identical losses
and gradients across runs, and matches eager numerics exactly. Any change
that breaks this test must be investigated and fixed before proceeding with
other tests.
