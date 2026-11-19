---
name: fusion_finder
description: Profile one GraphTrainer training step with every FX node labelled, attribute each GPU kernel to its node, group the memory-bound glue ops (norms, casts, RoPE, routing, permutes, residual adds) into fusion regions that one kernel could replace, rank the regions by time above their HBM roofline, lift each region into a call_module with a compile.extra_passes graph pass, and write the lifted regions as kernel problems (reference, contract, real inputs, bitwise checker) for kernel-writing agents. Use when asked where a graph_trainer step spends GPU time per FX node, which glue ops are worth fusing, or to prepare fusion targets for handwritten kernels.
---

# Finding fusion opportunities in a GraphTrainer step

Capture one training step as GraphTrainer runs it, with every FX node labelled in
the profiler trace; attribute each GPU kernel to its node; group the glue into
fusion regions and rank them by time above their HBM roofline; lift each region
into a `call_module`, the target a kernel replaces, and check that the lifted step
is bitwise identical; write each lifted region as a kernel problem that a
kernel-writing agent can take. The scripts are in `scripts/` and write to one
capture directory under `outputs/` (gitignored).

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
config, change only what one GPU needs, and pass it through `configure_capture`.
The `lifted` config is the same step with the regions lifted (step 6), and
`problems` also writes them as kernel problems (step 7):

```python
from fusion_finder_capture import configure_capture
from kernel_problems import register_kernel_problems
from lift_regions import register_lift_passes
from torchtitan_recipes.tests.graph_trainer.deepseek_v3 import (
    graph_trainer_deepseek_v3_debugmodel,
)

OUT = "outputs/fusion_finder/dsv3_debug"


def _base():
    config = graph_trainer_deepseek_v3_debugmodel()
    config.training.num_tokens_per_train_step = -1  # one microbatch per step
    return config


def capture():
    return configure_capture(_base(), OUT)


def lifted():
    config = configure_capture(_base(), f"{OUT}/lifted")
    config.compile.extra_passes = register_lift_passes(OUT)
    return config


def problems():
    config = configure_capture(_base(), f"{OUT}/problems_run")
    config.compile.extra_passes = [
        *register_lift_passes(OUT),
        register_kernel_problems(OUT, f"{OUT}/problems"),
    ]
    config.training.steps = 1
    config.profiler.enable_profiling = False
    return config
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
the profit floor: instances, kernels per instance, phases, modules and ops.
A region's gap is the most a kernel replacing it can save.

**6. Lift and check.** Run the capture and lifted configs with
`scripts/loss_compare.py`; loss and grad_norm must be bitwise identical:

```bash
COMM_BACKEND=fake PYTHONPATH=$SKILL:$OUT:. python scripts/loss_compare.py . . \
  --baseline-module ff_recipe --baseline-config capture --test-config lifted \
  --baseline-ngpus 8 --test-ngpus 8 --steps 10 --no-seed-checkpoint \
  --metrics loss,grad_norm --assert-equal --job-dump-folder $OUT/loss_compare
```

Both runs write their outputs under `--job-dump-folder`. Each lift pass logs the
instances it lifted. The passes also apply to other variants of the model (layer
count, shapes, parallelism): check a variant the same way, with its own config
with and without `register_lift_passes(OUT)`.

**7. Kernel problems.** Run the `problems` config like the capture (step 2):

```bash
PYTHONPATH=$SKILL:$OUT:. NGPU=8 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=<idle GPU> \
  python -m torchtitan.train --module ff_recipe --config problems --comm-backend fake
```

It writes `$OUT/problems/` (see "Kernel problems"): one problem per specialization
of a lifted region, by gap in `problems.json`. Hand a problem directory to a
kernel-writing agent; `check_kernel.py` decides whether its kernel is correct.

## Outputs

| file | content |
|---|---|
| `graph.jsonl.gz` | one record per node of the graph that runs: `name`, `op`, `target`, `args`/`kwargs` (`{"node": name}` for node references), `val` (tensor shape, stride, dtype, device), `phase`, `module_fqn`, `src` (innermost frame outside PyTorch), `code` (the node's line of FX code), `match_key` (target, args and kwargs with `"*"` for sizes and scalar operands, and devices as their type: what instances of a region share), `writes` and `aliases` (the inputs whose storage it writes or its output aliases, from the op schema), `before_inductor` (whether it exists where `compile.extra_passes` run) |
| `profile/iteration_10/rank0_trace.json` | the profiler trace of step 10 |
| `hbm_bandwidth.json` | the device and its measured HBM bandwidth |
| `node_table.csv` | per node, in graph order: phase, module, op, category, kernel time and count, output bytes, source, kernel names |
| `step_budget.json` | GPU window, busy and idle time, time by category, time outside the graph, nodes with unresolved sizes |
| `problems/` | the kernel problems of the lifted regions (see "Kernel problems") |
| `regions.json` | `profit_floor_ms` and every region, in the order the lift passes run (see "Lift passes"): `id`, `gap_ms`, `measured_ms`, `ideal_ms`, `instances`, `kernels_per_instance`, `phases`, `modules`, `ops`, `code` (the first instance's FX code, one line per node), and `pattern`: the first instance's `match_key` per node (`nodes`, node names as labels) and the nodes used outside it (`outputs`); `null` for a region with no pattern |

## How attribution works

`custom_codegen_pass` runs each node under a profiler label `<node>: <op> (<file>:<line>)`.
A kernel maps to its CUDA launch call (same `correlation` id), and the launch to
the innermost label enclosing it on the launching thread. Sizes that are symbols in
the graph (unbacked sizes such as MoE routed rows) are resolved from the input
sizes each label records; a node whose sizes stay unresolved has no `out_bytes`
(`nodes_without_bytes`), and `regions.py` counts it as 0 bytes.

## How regions are formed

Boundaries are the nodes a glue kernel cannot absorb: GEMMs, attention, Dist-MoE
fused ops, collectives, nodes whose output is not a GPU tensor (host values), and
nodes that Inductor creates after `compile.extra_passes` (compiled regions such as
flex attention). Paths follow tensors and storage: a write into a storage (an
in-place or `out=` op, through any view of it, e.g. FSDP's all-gather unpack into
the parameter buffers) comes after the earlier accesses to that storage and before
the later ones. An instance starts as one glue node and merges along these edges,
in graph order, unless the merge would put a boundary on a path between two of its
members. Views and allocations between glue nodes join too, but a view of a graph
input is like the input: the nodes that read it do not merge through it (every MoE
layer's aux loss reads `select(arg, 0, 0)`; merging through it would make one
instance of all layers). So:

- every glue node is in exactly one instance;
- no path leaves an instance and comes back, so an instance can become one
  `call_module` and one kernel can replace it.

The instances must also become calls together: with each instance as one node, the
graph must stay acyclic. Two instances can each be fine and still form a cycle
together, when one holds parts with no path between them. In GPT-OSS, the router
instance holds the softmax after the router GEMM and the expert sort before the
token permutation, which joined the residual add and norm before that GEMM. Each
such cycle is cut at the instance where the cut is smallest: a member the cycle
enters, and the members after it (here the token permutation).

Forward glue never merges with recompute or backward glue: that edge is a saved
activation, and fusing across it would move backward work into the forward pass.
Recompute glue merges with the backward glue it feeds.

Instances with the same topology are one region: the same ops, the same edges
among them and the same arguments, except sizes (arguments the op schema types
`SymInt`), scalar operands (numbers passed as a `Tensor` or `Scalar`) and device
indices. Module names, shapes and phases do not matter: e.g. the w13 weight
gradient of DeepSeek-V3's dense layers and of its MoE layers' shared experts are
one region. Ideal time per instance: bytes of the inputs from outside (the views'
bytes when only views read an input, e.g. one slice) plus bytes of the outputs used
outside. The profit floor is max(0.5 ms, 0.05% of GPU busy time): the gaps of
kernels written for regions track what they realize, so a region under it is not
worth a kernel.

Blind spots:
- Packing into a collective's send buffer (FSDP bucket ops) is part of the
  collective, so fusions into it are not found.
- Horizontal fusion: independent ops with no edge between them (per-layer weight
  casts) stay in separate regions.
- Instances grow greedily; another partition of the same nodes can be valid too.

## Lift passes

`register_lift_passes(capture_dir, ids=None)` registers one graph pass per region
above the profit floor (or per id), `lift_fusion_region_<id>`, and returns the
names in the order of `regions.json`, ready for `compile.extra_passes`.

A pass finds the region's instances by topology, with the code that groups them
into the region (`regions.Pattern`), not by node name. The pattern is the region's
`pattern` in `regions.json`; a match is a set of nodes with the same `match_key`s up
to node names and the same edges among them, found from each node with the op of
the pattern's last node, where a node the instance uses only inside is used only
inside the match. So the passes apply to other layer counts, shapes and ranks of
the same model code. A region whose members are joined only through a storage
(e.g. copies into slices of one buffer) has no pattern, and selecting it raises.

The passes run in the order of `regions.json`, and a node an earlier pass lifted is
not matched again: the first match wins. A small pattern can also match inside
another region's instance, so `regions.py` puts that region first (both orders
when each matches inside the other: the larger gap first); otherwise regions go by
gap. A match inside a region that is not lifted (e.g. below the profit floor) is
lifted with the smaller region.

A pass replaces match `k` with `fusion_<id>_i<k> = self.fusion_<id>_i<k>(...)`, a
submodule holding the matched nodes: the computation a kernel replaces, whose
inputs and outputs are the submodule's. The call takes the place of the match's
last node; nodes between its first and last node that depend on it, through a
tensor or a shared storage, move after the call in their order, and nothing else
moves. A submodule that writes storage is marked impure so that dead code
elimination keeps it. A pass raises if it finds no match, or if a path leaves a
match and comes back. To plug in a kernel, a later graph pass replaces the call.

## Kernel problems

`register_kernel_problems(capture_dir, problems_dir)` registers
`write_kernel_problems`, which runs after the lift passes. A problem is one
specialization of a lifted submodule: its exact code on inputs of its exact shapes,
strides and dtypes. It is keyed as `substitute_kernels` keys its kernel cache:

```
problems/
  problems.json             # the problems of this run, by gap
  check_kernel.py           # the bitwise gate, shared by all problems
  pattern_<l1>/             # level 1: the structure, ignoring literals and views
    <l2>/                   # level 2: the exact code and inputs
      problem.py            # reference(in0, ...); Model, get_inputs, get_init_inputs
      spec.json             # contract, input cases, the calls it comes from
      inputs.pt             # the integer, bool and complex inputs of each case
      kernel.py             # written by the agent: kernel_function(in0, ...)
```

The instances of a region at other sizes share `pattern_<l1>`, so a kernel written
for one problem is a start for the others. E.g. DeepSeek-V3's SwiGLU region has
one problem for the dense FFN and one for the shared experts.

- `problem.py`: `reference` is the submodule's code, the computation a kernel
  replaces; `Model`, `get_inputs` and `get_init_inputs` are the KernelBench interface
  that KernelAgent loads. Its docstring describes the problem: region, calls,
  modules, phase, measured and ideal time.
- `spec.json`: `signature`; `inputs` and `outputs` (shape, stride, dtype, device;
  symbolic sizes as `N0`, `N1`, ...); `mutated_inputs` (the inputs `reference` writes
  in place) and `output_aliases` (the input each output is a view of); `cases`
  (the call of each case, and the layout of its inputs: shapes, strides,
  storages, alignment); `source`: the
  region and each call of the step with this specialization, with its module,
  phase, and measured and ideal time when the call is an instance of the capture;
  `env`: GPU, torch, CUDA, and whether deterministic algorithms were on.
- `inputs.pt`: the integer, bool and complex inputs of each case, the first 3
  calls of the problem in the step (several layers; for symbolic sizes several
  sizes): index inputs (expert
  ids, positions, split sizes), masks and counts have the values of the run.
  Floating-point inputs are not saved: `check_kernel.build_args` generates them
  from a seed, in the case's layout. When a case is added, `reference` on the
  call's inputs must reproduce its outputs and in-place writes bit for bit, or the
  run raises.

The contract: `kernel_function` takes the inputs of `reference` and returns what it
returns, bit for bit, with the same strides and aliasing, and makes the same
in-place writes to its inputs and no others.
`python check_kernel.py <problem dir> kernel.py` checks it on each case, with the
floating-point inputs from 2 seeds, and prints `PASS` (exit 0) or the first
difference. A test that runs next to `kernel.py`, as KernelAgent runs its
tests, calls `check_kernel.main(["<problem dir>", "kernel.py"])`.

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
