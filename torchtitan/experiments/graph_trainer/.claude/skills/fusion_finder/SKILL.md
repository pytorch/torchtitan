---
name: fusion_finder
description: Profile one GraphTrainer training step with every FX node labelled, attribute each GPU kernel to its node, and rank the memory-bound glue ops (norms, casts, RoPE, routing, permutes, residual adds) by time above their HBM roofline. Use when asked where a graph_trainer step spends GPU time per FX node, or which glue ops are worth fusing.
---

# Finding fusion opportunities in a GraphTrainer step

Capture one training step as GraphTrainer runs it, with every FX node labelled in
the profiler trace; attribute each GPU kernel to its node; compare each glue node
with its HBM roofline. The scripts are in `scripts/` and write to one capture
directory under `outputs/` (gitignored).

Terms:
- **Glue:** a node that launches kernels and is not a GEMM, attention, Dist-MoE
  fused op or collective.
- **Phase:** `forward`, `recompute` (forward ops that `selective_activation_remat_pass`
  copies before their backward consumers, named `<node>_recomputed[_<n>]`) or `backward`.
- **Ideal time:** bytes read (inputs, broadcast dims once) plus bytes written,
  divided by the measured HBM bandwidth. **Gap:** measured kernel time minus
  ideal time, at least 0.

## Workflow

From the repo root:

```bash
SKILL=torchtitan/experiments/graph_trainer/.claude/skills/fusion_finder/scripts
OUT=outputs/fusion_finder/<model>
```

**1. Recipe.** Write `$OUT/ff_recipe.py`: build the model's verified GraphTrainer
config, change only what one GPU needs, and pass it through `configure_capture`:

```python
from fusion_finder_capture import configure_capture
from torchtitan_recipes.tests.graph_trainer.deepseek_v3 import (
    graph_trainer_deepseek_v3_debugmodel,
)


def capture():
    config = graph_trainer_deepseek_v3_debugmodel()
    config.training.num_tokens_per_train_step = -1  # one microbatch per step
    return configure_capture(config, "outputs/fusion_finder/dsv3_debug")
```

`configure_capture` selects the `fusion_finder` pass pipeline, profiles step 10
into `$OUT/profile/`, turns accelerator graphs off (graph replay hides the node
labels) and sets `debug.seed = 42` and `debug.deterministic = True`. It raises
unless the step is one microbatch without pipeline parallelism and Inductor
compilation is regional: other configurations do not run the pass pipeline, or
run one compiled call instead of the graph.

**2. Capture.** A large model runs rank 0 of its world on one GPU with the fake
process group (see the `fake_distributed_backend` skill); `NGPU` is the world size:

```bash
PYTHONPATH=$SKILL:$OUT:. NGPU=8 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=<idle GPU> \
  python -m torchtitan.train --module ff_recipe --config capture --comm-backend fake
```

Run on a GPU that no other process uses: contention changes the kernel times.

**3. Roofline.** On an idle GPU of the same type (it can run during the capture,
on another GPU):

```bash
CUDA_VISIBLE_DEVICES=<idle GPU> python $SKILL/measure_hbm_bandwidth.py $OUT
```

**4. Node table.** `python $SKILL/node_table.py $OUT` (CPU only) prints the step
budget and the glue ops furthest above their roofline, grouped by phase, module
(layer index dropped) and op: the same op in every layer is one group, and a
group is what one kernel would replace.

Check `outside_graph_ms` in the budget: kernels launched outside every node label
(optimizer, gradient clipping). Most of the step should be inside the graph.

## Outputs

| file | content |
|---|---|
| `graph.jsonl.gz` | one record per node of the graph that runs: `name`, `op`, `target`, `args`/`kwargs` (`{"node": name}` for node references), `val` (tensor shape, stride, dtype, device), `phase`, `module_fqn`, `src` (innermost frame outside PyTorch) |
| `profile/iteration_10/rank0_trace.json` | the profiler trace of step 10 |
| `hbm_bandwidth.json` | the device and its measured HBM bandwidth |
| `node_table.csv` | per node, in graph order: phase, module, op, category, kernel time and count, bytes, ideal time, gap, source, kernel names |
| `step_budget.json` | GPU window, busy and idle time, time by category, time outside the graph, total glue gap |

## How attribution works

`custom_codegen_pass` runs each node under a profiler label `<node>: <op> (<file>:<line>)`.
A kernel maps to its CUDA launch call (same `correlation` id), and the launch to
the innermost label enclosing it on the launching thread. Sizes that are symbols in
the graph (unbacked sizes such as MoE routed rows) are resolved from the input
sizes each label records; a glue node whose sizes stay unresolved has no bytes
(`glue_nodes_without_bytes`).

## Traps

- Under the fake process group, collectives do no communication, and Dist-MoE's
  fused ops run with every expert-parallel peer sending this rank's tokens, so
  their time follows this rank's routing. Neither is glue. The routing imbalance
  does not average out and can exceed the receive capacity:
  set `dist_moe_runtime.scratch_capacity_factor` to the expert-parallel degree.
- GPU idle time is large with accelerator graphs off (host launch overhead).
  Rank by kernel time, not by step time.
- A host sync (`_local_scalar_dense` on split sizes) shows up as GPU idle time,
  not as a kernel.
- Tensors over 2^31 elements make some eager kernels split into several launches
  per node (`num_kernels`).
