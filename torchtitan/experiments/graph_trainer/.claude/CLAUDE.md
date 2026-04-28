# GraphTrainer Development Guide

This supplements the root `.claude/CLAUDE.md` which covers core torchtitan
conventions (code style, naming, testing, PR expectations, etc.). Rules there
apply here too unless overridden below.

## Graph Pass Signature

All graph passes must follow this signature:
```python
def my_pass(gm: torch.fx.GraphModule, example_inputs, *, other_kwargs) -> torch.fx.GraphModule:
```
- The first two positional args are always `(gm, example_inputs)`.
- Any additional parameters must be **keyword-only** (after `*`).
- The pass must return the (possibly transformed) `GraphModule`.
- Passes that don't need `example_inputs` should still accept it (use `example_inputs=None`).

## Pass Configuration

Per-pass configuration (e.g. `static_input_indices` for cudagraph) must be
bound during pass construction in `construct_default_graph_passes` via
`functools.partial`, **not** threaded through `apply_graph_passes` as
parameters. The apply function is a generic pass runner and must not contain
pass-specific arguments.

## Memory Policy Framework

`tag_with_memory_policy_pass` is the unified framework for activation memory
management — selective activation checkpointing (SAC), CPU offload, and
mixtures of both. It is a two-step process:

1. **Tag nodes.** Each saved forward activation is tagged with one of:
   - `MUST_SAVE` — keep the activation in GPU memory.
   - `MUST_RECOMPUTE` — discard and recompute during backward.
   - `MUST_CPU_OFFLOAD` — offload to CPU, reload before backward.

   Tagging can be done manually (per-node annotations) or with an
   advanced solver algorithm that optimizes the save/recompute/offload
   split based on memory budget and compute cost.

2. **Act on tags.** Two passes run unconditionally after tagging (both
   are no-ops when no nodes carry the relevant tag):
   - `apply_cpu_offload_pass` — inserts offload/reload/wait ops for
     `MUST_CPU_OFFLOAD` nodes.
   - `selective_activation_remat_pass` — duplicates `MUST_RECOMPUTE`
     ops before backward and DCEs the originals.

The `--compile.memory_policy` config selects the tagging strategy.
New policies (e.g. budget-aware mixed SAC + offload) should be added
as new branches in `tag_with_memory_policy_pass`.

## Don't Modify Core for This Experiment

Do not add `if graph_trainer:` branches to `torchtitan/train.py`
or other core files. GraphTrainer extends `Trainer` and overrides behavior through
subclassing.


### Local development (debug models, 8 GPUs)

**Run all commands from the repo root.** Use the root `./run_train.sh` with
`MODULE=graph_trainer.llama3` (or `.deepseek_v3`). The 8B/16B configs use
`hf_assets_path` relative to the repo root.

For CooR precompile workflows that need `--virtual-local-rank`, use
`torchtitan/experiments/graph_trainer/run_train_precompile.sh` instead.

```bash
# Llama3 with FSDP + TP
NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_debugmodel \
    ./run_train.sh \
    --compile.mode aot_fx_trace \
    --parallelism.data_parallel_shard_degree=4 \
    --parallelism.tensor_parallel_degree=2

# DeepSeek-v3 with FSDP + TP + EP (requires H100)
NGPU=8 MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_debugmodel \
    ./run_train.sh \
    --compile.mode aot_fx_trace \
    --parallelism.data_parallel_shard_degree=4 \
    --parallelism.tensor_parallel_degree=2 \
    --parallelism.expert_parallel_degree=4 \
    --parallelism.expert_tensor_parallel_degree=1
```

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

### Debugging Graph Passes

Add `--compile.debug_graph_passes` to enable per-pass instrumentation:
timing, before/after tlparse graph dumps, and op-count diff summaries.
Use with `TORCH_TRACE` and `tlparse` to inspect graphs in the browser.

```bash
NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh \
    --compile.mode aot_fx_trace \
    --compile.debug_graph_passes \
    --dataloader.dataset c4_test \
    --training.steps 10
```

### Dumping Graph Modules for Debugging

To inspect a `GraphModule` at any point, dump it to a temporary file:

```python
from pathlib import Path
import tempfile

def dump_gm(gm: torch.fx.GraphModule, name: str) -> None:
    output_path = Path(tempfile.gettempdir()) / f"{name}.txt"
    output_path.write_text(
        gm.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            expanded_def=True,
        )
    )
    print(f"Dumped graph to {output_path}")
```

When debugging a graph pass, dump the graph before and after the pass and
diff the two files to see exactly what changed:

```python
def my_pass(gm, example_inputs):
    dump_gm(gm, "my_pass_before")
    # ... transform gm ...
    dump_gm(gm, "my_pass_after")
    return gm
```

```bash
diff /tmp/my_pass_before.txt /tmp/my_pass_after.txt
```

### Benchmark

Use `./run_train.sh` with a small number of steps. Disable tensorboard,
profiling, and flight recorder for cleaner timing. Always use
`--dataloader.dataset c4_test` for local runs to avoid downloading the
full C4 dataset from HuggingFace:

```bash
# Llama3 8B aot_fx_trace (8×H100, FSDP+TP, 20 steps)
NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh \
    --compile.mode aot_fx_trace \
    --parallelism.data_parallel_shard_degree=4 \
    --parallelism.tensor_parallel_degree=2 \
    --dataloader.dataset c4_test \
    --metrics.no-enable_tensorboard \
    --profiler.no-enable_profiling \
    --comm.trace_buf_size=0 \
    --training.steps 20

# DeepSeek-v3 16B aot_fx_trace (8×H100, FSDP+TP+EP, 20 steps)
NGPU=8 MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_16b ./run_train.sh \
    --compile.mode aot_fx_trace \
    --parallelism.data_parallel_shard_degree=4 \
    --parallelism.tensor_parallel_degree=2 \
    --parallelism.expert_parallel_degree=2 \
    --dataloader.dataset c4_test \
    --metrics.no-enable_tensorboard \
    --profiler.no-enable_profiling \
    --comm.trace_buf_size=0 \
    --training.steps 20
```

Look at the **last logged step** for steady-state metrics (the first few
steps include compilation overhead):

```
step: 20  loss: 11.83506  grad_norm:  9.6669  memory: 48.87GiB(51.44%)  tps: 4,376  tflops: 253.41  mfu: 25.62%
```

### Profiling

Add `--profiler.enable_profiling` to any `./run_train.sh` command.
Set `--profiler.profile_freq` to control which step is captured
(default: 10). Traces are saved to `{dump_folder}/profile_traces/`.

```bash
NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh \
    --compile.mode aot_fx_trace \
    --parallelism.data_parallel_shard_degree=4 \
    --parallelism.tensor_parallel_degree=2 \
    --dataloader.dataset c4_test \
    --profiler.enable_profiling \
    --profiler.profile_freq 10
```

### Memory Snapshot

Add `--profiler.enable_memory_snapshot` to capture a memory snapshot.
The snapshot fires at every `profile_freq`-th step and is saved to
`{dump_folder}/memory_snapshot/` (default: `./outputs/memory_snapshot/`).
Each rank produces its own file:
`iteration_{step}/rank{N}_memory_snapshot.pickle`.

Open the `.pickle` files with the
[PyTorch Memory Viz](https://pytorch.org/memory_viz) tool.

```bash
NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh \
    --compile.mode aot_fx_trace \
    --parallelism.data_parallel_shard_degree=4 \
    --parallelism.tensor_parallel_degree=2 \
    --dataloader.dataset c4_test \
    --profiler.enable_memory_snapshot \
    --profiler.profile_freq 10
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

### CUDA Graph Kernel Annotations

The `insert_kernel_annotations_pass` labels CUDA graph kernels with their
originating `nn.Module` path in profiler traces. It runs automatically in the
`aot_fx_trace` path (bundled with the cudagraph pass). The post-processor
is attached via ``Profiler.Config.trace_post_processors`` (see
``cudagraph_annotate_trace_post_processor``) so exported traces are
annotated automatically — no manual post-processing is needed.

Requirements: `cuda-python` package and CUDA toolkit/driver >= 13.1
(or `cuda-compat >= 13.1` on `LD_LIBRARY_PATH`). The pass is a no-op when
these are unavailable.

To view annotated traces, open the exported JSON in https://ui.perfetto.dev.
Kernel events will have `module_fqn` fields like `layers.0.attention.wq`.
