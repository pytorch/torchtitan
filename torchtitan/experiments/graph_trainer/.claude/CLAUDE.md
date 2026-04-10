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

### Bitwise Deterministic Guardrail

Before submitting any change, run the bitwise deterministic test first:
```bash
pytest torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py -x
```
This verifies that the aot_fx_trace path produces bitwise identical losses
and gradients across runs, and matches eager numerics exactly. Any change
that breaks this test must be investigated and fixed before proceeding with
other tests.
