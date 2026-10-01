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

## Pass Tiers

Graph passes are structured into two tiers:

1. **Default passes** (`passes.py`, `remove_noop_passes.py`, etc.) — always
   applied. These are numerics-preserving: cleanup, memory policy, bucketing,
   async TP, FlexInnerAttention regional Inductor (required for bitwise match with
   eager).

2. **Performance passes** (`performance_passes.py`) -- opt-in via
   `config.compile.numerics_changing_optim = True`. These improve performance but may change numerics
   compared to the uncompiled path (e.g. RMSNorm Inductor fusion).

When adding a new pass, put it in `performance_passes.py` if it changes
numerics; otherwise put it in `passes.py` or a dedicated file like
`remove_noop_passes.py`.

## Memory Policy Framework

PyTorch's module-level `torch.utils.checkpoint` and eager SAC make
coarse save-or-recompute decisions for an entire module's output, and
composing activation checkpointing with CPU offload is difficult. This
framework instead operates on the FX graph at individual tensor
granularity: each activation can independently be saved, recomputed, or
offloaded, and different strategies mix freely within a single layer.

`tag_with_memory_policy_pass` is the unified entry point. It is a
two-step process:

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
     ops in front of their backward consumers and erases originals whose
     consumers were all backward.

The `config.compile.memory_policy` field selects the tagging strategy.
New policies (e.g. budget-aware mixed SAC + offload) should be added
as new branches in `tag_with_memory_policy_pass`.

**NUMA binding for CPU offload:** On multi-NUMA machines (e.g. GB200
NVLink-C2C), D2H/H2D bandwidth is ~350 GB/s NUMA-local vs ~120 GB/s
cross-NUMA. `Trainer` automatically applies NUMA binding
(`AffinityMode.NODE`) on CUDA hardware at init, pinning each worker
to the NUMA node of its GPU. Falls back gracefully on non-CUDA
hardware or when `numactl` is unavailable.

**Inspecting tags:** `log_activation_memory_policy` (`log_activation_memory_policy.py`)
prints all forward nodes consumed by backward, grouped by layer with
identical patterns consolidated. Shows memory, dtype, policy
(SAVE/RECOMPUTE/OFFLOAD), shape, submodule, target op, and source location.
It runs automatically at the end of `tag_with_memory_policy_pass`,
logging to both `logger.debug` and tlparse (via `trace_structured`).

## Don't Modify Core for This Experiment

Do not add `if graph_trainer:` branches to `torchtitan/train.py`
or other core files. GraphTrainer extends `Trainer` and overrides behavior through
subclassing.


### Local development (debug models, 8 GPUs)

**Run all commands from the repo root.** Use the root `./run_train.sh` with
`MODULE=graph_trainer.llama3` (or `.deepseek_v3`, `.qwen3`). The 8B/16B configs
use `hf_assets_path` relative to the repo root.

For CooR precompile workflows that need `--virtual-local-rank`, use
`torchtitan/experiments/graph_trainer/run_train_precompile.sh` instead.

```bash
# Llama3 using the topology defined by the recipe
NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_debugmodel \
    ./run_train.sh

# DeepSeek-v3 using the topology defined by the recipe (requires H100)
NGPU=8 MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_debugmodel \
    ./run_train.sh
```

For a different topology, add a derived Python recipe and update its
`parallelism` fields instead of passing configuration flags on the command line.

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

Every graph pass is always instrumented: per-pass timing and op-count diff
summaries are logged, and before/after graphs are dumped to tlparse. Run with
`TORCH_TRACE` and use `tlparse` to inspect the graphs in the browser.

```bash
TORCH_TRACE=/tmp/trace NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh
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

### Printing and Inspecting Tensors Inside a Compiled Function

To inspect tensor values or gradients *inside* `torch.compile` without graph
breaks:

- **Simple printing**: `torch._higher_order_ops.print("norm={}", x.norm())` —
  format-string, forward-only. DTensors print each rank's local view with an
  automatic `[rank N]` prefix.
- **Gradient norms**: `from torch.utils.debug_log import debug_grad_log` —
  call on intermediates (not direct graph inputs); fires during backward and
  logs per-tensor gradient norms.
- **Custom logic** (arbitrary Python, file logging, rank filtering, fwd+bwd):
  compose `@leaf_function` with `@fn.register_multi_grad_hook` from
  `torch._dynamo.decorators`, following the pattern in
  `torch/utils/debug_log.py`:

  ```python
  from torch._dynamo.decorators import leaf_function

  @leaf_function
  def log_tensor(x, tag=""):
      return None  # no-op in forward

  @log_tensor.register_fake
  def log_tensor_fake(x, tag=""):
      return None

  @log_tensor.register_multi_grad_hook
  def log_tensor_hook(x_grad, tag=""):  # non-tensor args passed through unchanged
      print(f"[{tag}][bwd] grad_norm={x_grad.norm():.4f}")

  log_tensor(x, "after_add")
  ```

### Benchmark

Use `./run_train.sh` with a benchmark recipe that sets a small number of steps
and disables tensorboard, profiling, and the flight recorder. The checked-in
Llama3 autoresearch recipe already sets FSDP=4, TP=2, 20 steps, and these
logging controls:

```bash
# Llama3 8B GraphRuntime (8xH100, FSDP+TP, 20 steps)
NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b_autoresearch \
    ./run_train.sh

# A DeepSeek-v3 benchmark should use a derived recipe with the corresponding
# parallelism and logging fields set in Python.
NGPU=8 MODULE=my_graph_configs CONFIG=deepseek_v3_16b_benchmark ./run_train.sh
```

Look at the **last logged step** for steady-state metrics (the first few
steps include compilation overhead):

```
step: 20  loss: 11.83506  grad_norm:  9.6669  memory: 48.87GiB(51.44%)  tps: 4,376  tflops: 253.41  mfu: 25.62%
```

### Profiling

Set `config.profiler.enable_profiling = True` in the recipe.
Set `config.profiler.profile_freq` to control which step is captured
(default: 10). Traces are saved to `{dump_folder}/profile_traces/`.

```bash
NGPU=8 MODULE=my_graph_configs CONFIG=llama3_8b_profile ./run_train.sh
```

### Memory Snapshot

Set `config.profiler.enable_memory_snapshot = True` in the recipe to capture a
memory snapshot.
The snapshot fires at every `profile_freq`-th step and is saved to
`{dump_folder}/memory_snapshot/` (default: `./outputs/memory_snapshot/`).
Each rank produces its own file:
`iteration_{step}/rank{N}_memory_snapshot.pickle`.

Open the `.pickle` files with the
[PyTorch Memory Viz](https://pytorch.org/memory_viz) tool.

```bash
NGPU=8 MODULE=my_graph_configs CONFIG=llama3_8b_memory_snapshot ./run_train.sh
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

### Numerics Debugging

For investigating numerics divergence, use the `numerics_debugging` skill at
[`.claude/skills/numerics_debugging/SKILL.md`](../../../../.claude/skills/numerics_debugging/SKILL.md).

### Async Tensor Parallel (micro-pipeline TP)

Enable with `config.compile.enable_async_tensor_parallel = True`. This fuses
all-gather + matmul and matmul + reduce-scatter into pipelined ops using
symmetric memory (NVLink).

**When to use:**
- TP is enabled and the model has large hidden dimensions (shard_dim >= 1024
  after TP split; e.g. llama3 8B dim=4096 with TP=4 gives shard=1024).
- Below this threshold the pipeline chunking overhead exceeds the overlap
  benefit — the pass silently skips small shards.
- Requires NVLink-connected GPUs (H100, A100 NVSwitch, etc.).

**Example:**
```bash
NGPU=4 MODULE=my_graph_configs CONFIG=llama3_8b_async_tp4 ./run_train.sh
```

### CUDA Graph Kernel Annotations

The `insert_kernel_annotations_pass` labels CUDA graph kernels with their
originating `nn.Module` path in profiler traces. It runs automatically in the
`aot_fx_trace` path (bundled with the cudagraph pass). The profiler passes the
captured annotations to ``export_chrome_trace``, which bakes them into the trace
as it writes, so they are merged automatically and no post-processing is needed.

Requirements: `cuda-python` package and CUDA toolkit/driver >= 13.1
(or `cuda-compat >= 13.1` on `LD_LIBRARY_PATH`). The pass is a no-op when
these are unavailable.

To view annotated traces, open the exported JSON in https://ui.perfetto.dev.
Kernel events will have `module_fqn` fields like `layers.0.attention.wq`.
