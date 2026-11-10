# Prompt: Eager Fusion-Region Finder

You are running one side of an experiment comparing two substrates for finding
kernel-fusion opportunities in a torchtitan training step. Your substrate is
**eager**: the eager `Trainer`, its Python model code, and its runtime traces.
A sibling agent does the same job on GraphTrainer's FX graph; its results will
be compared with yours. Don't look for or read its outputs.

**Goal:** find every fusion opportunity in one DeepSeek-V3 671B training
step, measure how far each is from its roofline, and report them in a
readable, comparable format. Don't write kernels or change model code. The
deliverable is the inventory and your observations about the substrate.

## Fixed setup (identical for both substrates; don't change it)

- Repo: `/home/bahuang/local/torchtitan`. Follow `.claude/CLAUDE.md` and
  `.claude/rules/`.
- Model and recipe: `torchtitan_recipes.models.deepseek_v3.deepseek_v3_671b`
  with `seq_len=4096`.
- Fake process group (`--comm-backend fake`), `NGPU=256`,
  `data_parallel_shard_degree=256`, `expert_parallel_degree=64`.
- Tokens per microbatch per DP rank: `24 * 4096`. No gradient accumulation.
- `training.steps = 10`; profile one active step after 3 warmup steps.
- `debug.seed = 42`, `debug.deterministic = True`.
- Full activation recompute: `activation_remat = FullActivationRematConfig()`
  (`torchtitan/distributed/activation_remat.py`).
- `model.local_compile.regions = []`, so both substrates execute the same aten
  ops. GraphTrainer requires this.
- Accelerator (CUDA) graphs off (`training.disable_accelerator_graphs = True`,
  already set by the recipe; verify).
- GPUs:
  - `CUDA_VISIBLE_DEVICES=0` for the main capture and all timing;
  - GPU 1 is spare, for short probes only;
  - another process already holds ~17.6 GB on each GPU. Check `nvidia-smi`
    utilization before timing, and record it if the GPU is busy.
- Outputs: everything goes under `outputs/fusion_finder/eager/`
  (gitignored).
- Constraints:
  - don't modify tracked files;
  - don't commit;
  - don't upload anything (no pastry, manifold, or Perfetto uploads);
  - no GitHub activity.

`scripts/profile_dsv3_fake.sh` is a reference launcher, but its eager path is
broken: it imports `torchtitan.distributed.activation_checkpoint.FullAC`,
which no longer exists. Don't edit it. Write your own derived recipe and
launcher under your output directory.

## Substrate rules (what makes this the eager experiment)

- **Use:** the model source code, the eager runtime, and eager runtime traces.
- **Don't:** trace the model into an FX graph (make_fx, torch.compile,
  torch.export, GraphTrainer, `torch.fx`), read GraphTrainer graph dumps or
  tlparse output, or use `torchtitan/experiments/graph_trainer/`. The
  experiment measures what the eager substrate can show.

## Steps

### 1. Capture

- Kineto profiler with CPU and CUDA activities, `record_shapes=True`, and
  `with_stack=True`. Tag every module call with its FQN, for example through
  forward pre/post hooks that open a `record_function`.
- An execution trace from the same run:
  `torch.profiler.ExecutionTraceObserver`, passed as
  `execution_trace_observer=` to the profiler. On this build it records every
  aten op, forward and backward, with:
  - input/output tensor and storage IDs, shapes, strides, and dtypes;
  - `seq_id`, linking each backward node to the forward op that created it;
  - `fw_parent`/`fw_tid`, marking backward-thread ops;
  - `rf_id`, joining each op to its Kineto events.
- Work out how to tell recompute (forward ops replayed during backward by
  `torch_remat`) from the original forward. Explain the method in the report.
- Keep the raw traces and every script you write under the output directory.

### 2. Calibrate the roofline on GPU 0

Measure, and report the numbers:

- achievable HBM bandwidth (a large device-to-device copy, and a read-only
  reduction);
- bf16 dense GEMM peak at a representative DSv3 shape.

Ideal time = `max(bytes / bandwidth, flops / peak)`.

### 3. Attribute and build the op table

Join the two traces into an op-level table, one row per aten op instance:

- GPU kernel time (sum of its kernels);
- source file:line, module FQN, and layer index;
- phase: forward, recompute, or backward. For backward ops, the forward op
  they came from;
- tensor IDs (for dataflow edges), shapes, dtypes, and bytes read/written;
- flops where they apply;
- ideal time and measured/ideal ratio.

Also tag kernels launched by fake collectives (parent `c10d` /
`record_param_comms` CPU ops). They stand in for communication: report them
but exclude them from compute.

Coverage check: attributed kernel time must be at least 95% of GPU-busy time
in the profiled step. Also report GPU idle time.

### 4. Find regions

Do all the work yourself; don't spawn subagents. Work from the existing
artifacts; don't rerun the 671B capture unless the first one is unusable.
Short probes may use GPU 1, but all timing comes from the main trace.

The goal is to find all fusion opportunities, large or small. Don't apply
a size threshold; rank afterwards. Include opportunities that cross module
boundaries, cross layers, lie in the backward, or span recompute and
backward.

For each region, state why it can be fused (the dataflow) and what blocks
it.

Then consolidate:

- merge duplicates and overlapping regions (record `overlaps_with`);
- rank by `gap_ms_total`;
- check that every non-GEMM, non-attention, non-comm kernel class is in a
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

### `outputs/fusion_finder/eager/regions.json`

```json
{
  "substrate": "eager",
  "setup": {"git_commit": "", "command": "", "config_deltas": {}, "gpu": "",
            "gpu_busy_during_timing": false,
            "hbm_gbps": 0, "bf16_gemm_tflops": 0},
  "step_budget": {"step_gpu_busy_ms": 0, "step_gpu_idle_ms": 0,
                  "attributed_pct": 0,
                  "by_category_ms": {"gemm": 0, "attention": 0, "fake_comm": 0,
                                     "memory_bound_glue": 0, "optimizer": 0,
                                     "other": 0}},
  "regions": [{
    "id": "E-01",
    "title": "",
    "phases": ["forward", "recompute", "backward"],
    "scope": ["intra-module"],
    "fusion_kind": "",
    "location": {"module_fqn_pattern": "", "source": "file:line-line"},
    "ops": ["aten::..."],
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

### `outputs/fusion_finder/eager/report.md`

1. **Setup:** command, commit, config deltas from the recipe, GPU and its
   utilization, roofline numbers.
2. **Step budget:** a table by category, attribution coverage, and GPU idle
   time.
3. **Ranked regions:** all regions in one table (id, title, phases, scope,
   kind, instances, measured ms, ideal ms, gap ms, % of step, tier).
4. **Region details:** a few lines each, covering what, where, the dataflow
   that makes it fusable, blockers, and the tier with its reason.
5. **No opportunity:** the remaining memory-bound time, with reasons.
6. **Substrate observations**, which matter as much as the inventory:
   - what the eager substrate made easy and hard;
   - which evidence was missing or needed workarounds;
   - how you identified recompute and backward provenance;
   - false leads you discarded;
   - which opportunity types you believe you could miss;
   - effort: wall time and the scripts you wrote.
7. **Reproduction:** the scripts under the output directory and how to rerun
   them.

## Done when

- Attribution covers at least 95% of GPU-busy time.
- Every memory-bound kernel class is in a region or in `no_opportunity`.
- `regions.json` validates against the schema above.
- `report.md` is complete.

Reply with the path to the report, the five largest gaps, the number of
regions in each scope tag, and the attribution coverage.
