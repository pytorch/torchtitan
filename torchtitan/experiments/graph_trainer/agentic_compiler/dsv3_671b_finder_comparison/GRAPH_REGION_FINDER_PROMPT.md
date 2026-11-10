# Prompt: Graph Fusion-Region Finder

You are running one side of an experiment comparing two substrates for finding
kernel-fusion opportunities in a torchtitan training step. Your substrate is
**graph**: GraphTrainer's traced joint forward+backward FX graph and its
runtime traces. A sibling agent does the same job on the eager trainer; its
results will be compared with yours. Don't look for or read its outputs.

**Goal:** find every fusion opportunity in one DeepSeek-V3 671B training
step, measure how far each is from its roofline, and report them in a
readable, comparable format. Don't write kernels or graph passes that change
the graph. The deliverable is the inventory and your observations about the
substrate.

## Fixed setup (identical for both substrates; don't change it)

- Repo: `/home/bahuang/local/torchtitan`. Follow `.claude/CLAUDE.md`,
  `.claude/rules/`, and `torchtitan/experiments/graph_trainer/AGENTS.md`.
- Model and recipe: `torchtitan_recipes.models.deepseek_v3.deepseek_v3_671b`
  with `seq_len=4096`, converted with `to_graph_trainer_config(...,
  GraphTrainerDeepSeekV3Model.Config)`.
- Fake process group (`--comm-backend fake`), `NGPU=256`,
  `data_parallel_shard_degree=256`, `expert_parallel_degree=64`.
- Tokens per microbatch per DP rank: `24 * 4096`. No gradient accumulation.
- `training.steps = 10`; profile one active step after 3 warmup steps.
- `debug.seed = 42`, `debug.deterministic = True`.
- Full activation recompute: `compile.memory_policy = "full"`.
- `model.local_compile.regions = []` (GraphTrainer requires it), so both
  substrates execute the same aten ops.
- Accelerator (CUDA) graphs off (`training.disable_accelerator_graphs = True`,
  already set by the base recipe; verify after conversion).
- GPUs:
  - `CUDA_VISIBLE_DEVICES=2` for the main capture and all timing;
  - GPU 3 is spare, for short probes only;
  - another process already holds ~17.6 GB on each GPU. Check `nvidia-smi`
    utilization before timing, and record it if the GPU is busy.
- Outputs: everything goes under `outputs/fusion_finder/graph/`
  (gitignored).
- Constraints:
  - don't modify tracked files;
  - don't commit;
  - don't upload anything (no pastry, manifold, or Perfetto uploads);
  - no GitHub activity.

`scripts/profile_dsv3_fake.sh` (`TRAINER=graph`) already implements most of
this setup. You may use it with `UPLOAD_TRACE=0 UPLOAD_MEMORY_SNAPSHOT=0
UPLOAD_TLPARSE=0 UPLOAD_LOG=0`. `TRACE_SUBDIR` is relative to `outputs/`, so
point it at `fusion_finder/graph/...`. Don't edit the script. If you need a
change, copy it into your output directory.

## Substrate rules (what makes this the graph experiment)

- **Use:** the traced FX graph after GraphTrainer's passes, graph node
  metadata (`stack_trace`, module FQN / `nn_module_stack`, custom annotations,
  memory-policy and remat tags, `val` shapes and dtypes), graph runtime
  traces, and `torchtitan/experiments/graph_trainer/` code.
- **Don't:** read model source under `torchtitan/models/` or
  `torchtitan/components/` to find candidates; the experiment measures what
  the graph shows. You may open a source line that a node's `stack_trace`
  points to, but only to name a region you already found in the graph. Don't
  profile the eager trainer.

## Steps

### 1. Capture

- **The graph.** Get the final joint graph that executes: after all passes
  (memory policy and remat, bucketing, reordering), before or without any
  Inductor compile. Either use `TORCH_TRACE` + `tlparse` artifacts (before and
  after each pass), or dump it in-process (see "Dumping Graph Modules" in
  `AGENTS.md`). The 671B graph is huge; analyze it programmatically (load the
  `GraphModule` or parse the dump).
- **Per-node attribution.** `custom_codegen_pass`
  (`graph_trainer/custom_codegen.py`) wraps every FX node in
  `_RecordFunctionFast("node_name: op (file:line)")` when the profiler is on.
  It isn't in the default pass list, so enable it for the profiled run in the
  least invasive way, e.g. a derived recipe or trainer subclass under your
  output directory. If you must edit `graph_trainer` code, keep the edit
  uncommitted, list it in the report, and revert it at the end.
- **Kineto profiler trace** of one active step with CPU and CUDA activities.
- Work out how to tell recompute nodes (forward ops duplicated before their
  backward consumers by `selective_activation_remat_pass`) from original
  forward and backward nodes. Explain the method in the report.
- Keep the raw traces, the graph dump, and every script you write under the
  output directory.

### 2. Calibrate the roofline on GPU 2

Measure, and report the numbers:

- achievable HBM bandwidth (a large device-to-device copy, and a read-only
  reduction);
- bf16 dense GEMM peak at a representative DSv3 shape.

Ideal time = `max(bytes / bandwidth, flops / peak)`. You may reuse the logic
of `torch._inductor.fx_passes.overlap_scheduling.estimate_roofline_runtime_ms`,
but use your measured numbers.

### 3. Attribute and build the node table

Map each GPU kernel to its enclosing `_RecordFunctionFast` label and from
there to its FX node. Build a node-level table:

- GPU kernel time;
- node name, op target, module FQN, and layer index;
- phase: forward, recompute, or backward;
- input and output nodes (graph edges), shapes, dtypes, and bytes
  read/written;
- flops where they apply;
- ideal time and measured/ideal ratio.

Also tag the kernels of fake collectives (`_c10d_functional` nodes and their
waits). They stand in for communication: report them but exclude them from
compute.

Coverage check: attributed kernel time must be at least 95% of GPU-busy time
in the profiled step. Also report GPU idle time.

### 4. Find regions

Do all the work yourself; don't spawn subagents. Work from the existing
artifacts; don't rerun the 671B capture unless the first one is unusable.
Short probes may use GPU 3, but all timing comes from the main trace.

The goal is to find all fusion opportunities, large or small. Don't apply
a size threshold; rank afterwards. Include opportunities that cross module
boundaries, cross layers, lie in the backward, or span recompute and
backward.

For each region, state why it can be fused (the dataflow) and what blocks
it.

Then consolidate:

- merge duplicates and overlapping regions (record `overlaps_with`);
- rank by `gap_ms_total`;
- check that every non-GEMM, non-attention, non-comm node class is in a
  region or in the `no_opportunity` list with a reason.

## Definitions

- **Phases:** `forward`, `recompute`, `backward`.
- **Scope tags:** `intra-module`, `cross-module`, `cross-layer`,
  `recompute-backward`.
- **Bitwise tier:** how hard a bitwise-identical fused kernel would be.
  - T0: pointwise, views, and casts only.
  - T1: involves reductions; the reduction order must be matched.
  - T2: order-sensitive data movement.
  - T3: needs a GEMM or attention inside.
- **Region ideal time:** external bytes only (region inputs read once, outputs
  written once), or flops/peak if larger. Gap = `(measured - ideal) *
  instances_per_step`.
- **Evidence:** `measured` if every member kernel's time comes from the trace;
  `estimated` otherwise. Say what was estimated.

## Output

### `outputs/fusion_finder/graph/regions.json`

```json
{
  "substrate": "graph",
  "setup": {"git_commit": "", "command": "", "config_deltas": {}, "gpu": "",
            "gpu_busy_during_timing": false,
            "hbm_gbps": 0, "bf16_gemm_tflops": 0},
  "step_budget": {"step_gpu_busy_ms": 0, "step_gpu_idle_ms": 0,
                  "attributed_pct": 0,
                  "by_category_ms": {"gemm": 0, "attention": 0, "fake_comm": 0,
                                     "memory_bound_glue": 0, "optimizer": 0,
                                     "other": 0}},
  "regions": [{
    "id": "G-01",
    "title": "",
    "phases": ["forward", "recompute", "backward"],
    "scope": ["intra-module"],
    "fusion_kind": "",
    "location": {"module_fqn_pattern": "", "graph_nodes_example": [],
                 "source": "file:line-line (from stack_trace)"},
    "ops": ["aten...."],
    "kernels_per_instance": 0,
    "instances_per_step": 0,
    "shapes_dtypes": "",
    "external_bytes_per_instance": 0,
    "measured_us_per_instance": 0, "measured_ms_total": 0,
    "ideal_us_per_instance": 0, "ideal_ms_total": 0,
    "gap_ms_total": 0, "gap_pct_of_step": 0,
    "bitwise_tier": "T0", "bitwise_notes": "",
    "blockers": "",
    "evidence": "measured|estimated",
    "how_found": "",
    "overlaps_with": [],
    "notes": ""
  }],
  "no_opportunity": [{"what": "", "time_ms": 0, "reason": ""}]
}
```

### `outputs/fusion_finder/graph/report.md`

1. **Setup:** command, commit, config deltas from the recipe, any temporary
   `graph_trainer` edits, GPU and its utilization, roofline numbers.
2. **Step budget:** a table by category, attribution coverage, and GPU idle
   time.
3. **Ranked regions:** all regions in one table (id, title, phases, scope,
   kind, instances, measured ms, ideal ms, gap ms, % of step, tier).
4. **Region details:** a few lines each, covering what, where (example node
   names), the dataflow that makes it fusable, blockers, and the tier with its
   reason.
5. **No opportunity:** the remaining memory-bound time, with reasons.
6. **Substrate observations**, which matter as much as the inventory:
   - what the graph substrate made easy and hard;
   - which evidence was missing or needed workarounds;
   - how you identified recompute and backward provenance;
   - false leads you discarded;
   - which opportunity types you believe you could miss;
   - effort: wall time and the scripts you wrote.
7. **Reproduction:** the scripts under the output directory and how to rerun
   them.

## Done when

- Attribution covers at least 95% of GPU-busy time.
- Every memory-bound node class is in a region or in `no_opportunity`.
- `regions.json` validates against the schema above.
- `report.md` is complete.
- Any temporary `graph_trainer` edits are reverted.

Reply with the path to the report, the five largest gaps, the number of
regions in each scope tag, and the attribution coverage.
