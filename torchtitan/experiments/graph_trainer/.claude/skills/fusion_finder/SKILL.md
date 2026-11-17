---
name: fusion_finder
description: Profile one GraphTrainer training step with every FX node labelled, attribute each GPU kernel to its node, group the memory-bound glue ops (norms, casts, RoPE, routing, permutes, residual adds) into fusion regions that one kernel could replace, and rank the regions by time above their HBM roofline. Use when asked where a graph_trainer step spends GPU time per FX node, or which glue ops are worth fusing.
---

# Finding fusion opportunities in a GraphTrainer step

Capture one training step as GraphTrainer runs it, with every FX node labelled in
the profiler trace; attribute each GPU kernel to its node; group the glue into
fusion regions and rank them by time above their HBM roofline. The scripts are in
`scripts/` and write to one capture directory under `outputs/` (gitignored).

Terms:
- **Glue:** a node that launches kernels and is not a GEMM, attention, Dist-MoE
  fused op or collective (including the send-buffer pack of a bucketed collective).
- **Phase:** `forward`, `recompute` (forward ops that `selective_activation_remat_pass`
  copies before their backward consumers, named `<node>_recomputed[_<n>]`) or `backward`.
- **Ideal time** of a region instance: the bytes it reads from outside plus the
  bytes it writes for outside, over the measured HBM bandwidth. **Gap:** measured
  kernel time minus ideal time, at least 0.
- **Region:** a set of glue nodes one kernel could run (see "How regions are
  formed"). It usually repeats once per layer; each repetition is an **instance**.

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

**3. Roofline.** For `regions.py`, on an idle GPU of the same type (it can run
during the capture, on another GPU):

```bash
CUDA_VISIBLE_DEVICES=<idle GPU> python $SKILL/measure_hbm_bandwidth.py $OUT
```

**4. Node table.** `python $SKILL/node_table.py $OUT` (CPU only) attributes every
kernel to its node and prints the step budget. Check `outside_graph_ms`: kernels
launched outside every node label (optimizer, gradient clipping); most of the step
should be inside the graph. `nodes_without_bytes` should be 0.

**5. Regions.** `python $SKILL/regions.py $OUT` (CPU only) prints the regions above
the profit floor, by gap: instances, kernels per instance, phases, modules and ops.
A region's gap is the most a kernel replacing it can save.

## Outputs

| file | content |
|---|---|
| `graph.jsonl.gz` | one record per node of the graph that runs: `name`, `op`, `target`, `args`/`kwargs` (`{"node": name}` for node references), `val` (tensor shape, stride, dtype, device), `phase`, `module_fqn`, `src` (innermost frame outside PyTorch), `code` (the node's line of FX code) |
| `profile/iteration_10/rank0_trace.json` | the profiler trace of step 10 |
| `hbm_bandwidth.json` | the device and its measured HBM bandwidth |
| `node_table.csv` | per node, in graph order: phase, module, op, category, kernel time and count, output bytes, source, kernel names |
| `step_budget.json` | GPU window, busy and idle time, time by category, time outside the graph, nodes with unresolved sizes |
| `regions.json` | `profit_floor_ms` and every region by gap: `id`, `gap_ms`, `measured_ms`, `ideal_ms`, `instances`, `kernels_per_instance`, `phases`, `modules`, `ops`, `code` (the first instance's FX code, one line per node), and `members` (the node names of each instance) |

## How attribution works

`custom_codegen_pass` runs each node under a profiler label `<node>: <op> (<file>:<line>)`.
A kernel maps to its CUDA launch call (same `correlation` id), and the launch to
the innermost label enclosing it on the launching thread. Sizes that are symbols in
the graph (unbacked sizes such as MoE routed rows) are resolved from the input
sizes each label records; a node whose sizes stay unresolved has no `out_bytes`
(`nodes_without_bytes`), and `regions.py` counts it as 0 bytes.

## How regions are formed

Boundaries are the nodes a glue kernel cannot absorb: GEMMs, attention, Dist-MoE
fused ops, collectives, and nodes whose output is not a GPU tensor (host values).
An instance starts as one glue node and merges along glue-to-glue edges, in graph
order, unless the merge would put a boundary on a path between two of its members.
Views and allocations between glue nodes join too. So:

- every glue node is in exactly one instance;
- no path leaves an instance and comes back, so an instance can become one
  `call_module` and one kernel can replace it.

Forward glue never merges with recompute or backward glue: that edge is a saved
activation, and fusing across it would move backward work into the forward pass.
Recompute glue merges with the backward glue it feeds.

Instances with the same members up to the layer index (op, phase, module with
`layers.<N>` as `layers.*`, output shapes with symbolic sizes abstracted) are one
region. Ideal time per instance: bytes of the inputs from outside (the views'
bytes when only views read an input, e.g. one slice) plus bytes of the outputs used
outside. The profit floor is max(0.5 ms, 0.05% of GPU busy time): the gaps of
kernels written for regions track what they realize, so a region under it is not
worth a kernel.

Blind spots:
- Packing into a collective's send buffer (FSDP bucket ops) is part of the
  collective, so fusions into it are not found.
- Horizontal fusion: independent ops with no edge between them (per-layer weight
  casts) stay in separate regions.
- An accumulator across layers (the aux loss sum) joins every layer's piece into one
  instance, apart from the per-layer region.
- Instances grow greedily; another partition of the same nodes can be valid too.

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
