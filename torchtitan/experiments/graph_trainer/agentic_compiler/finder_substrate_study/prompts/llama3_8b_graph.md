# Task: find kernel-fusion opportunities in one training step

**Goal:** find every kernel-fusion opportunity in one Llama 3 8B training step, measure how far each is
from its roofline, and report them in the format below. Don't write kernels or change any code. The
deliverables are the inventory and your observations about working from the artifacts you were given.

## What you have

- **Your directory:** `/home/bahuang/local/step_analysis/llama3_8b_graph`. Read-only inputs are in `/home/bahuang/local/step_analysis/llama3_8b_graph/artifacts/` and
  `/home/bahuang/local/step_analysis/llama3_8b_graph/hw_profile.json`; write everything you produce (scripts, tables, outputs) under `/home/bahuang/local/step_analysis/llama3_8b_graph/out/`.
  Start each shell command with `cd /home/bahuang/local/step_analysis/llama3_8b_graph` or use absolute paths.
- **Repo snapshot (read-only):** `/home/bahuang/local/step_analysis/repo`, the torchtitan code that produced the artifacts. If you import
  torchtitan from Python, put `/home/bahuang/local/step_analysis/repo` first on `PYTHONPATH`.
- **The program:** Llama 3 8B from `torchtitan_recipes.models.llama3.llama3_8b(seq_len=8192)` (BF16), laid out as FSDP 8, with one 8,192-token microbatch per step.
  - One process with a fake process group stands in for the whole job: it runs rank 0's computation and
    emulates each collective with local copies. Kernels launched under a collective CPU op (`c10d::`,
    `_c10d_functional::`, `record_param_comms`) stand in for communication: report their time, but exclude
    them from compute and from regions.
  - No activation recompute, `local_compile` off, deterministic algorithms on, CUDA graphs off. Step 10 was
    profiled after 3 warmup steps.
- **Roofline numbers:** `/home/bahuang/local/step_analysis/llama3_8b_graph/hw_profile.json` (HBM bandwidth and GEMM peaks, measured on this GPU type
  with random data). Use these numbers; don't measure your own. Ideal time = `max(bytes / hbm_copy_gbps,
  flops / GEMM peak of the matching precision)`.

## Your artifacts

The step ran as GraphTrainer's traced joint forward + backward FX graph (`memory_policy="none"`). You work
from that graph, its node metadata, its runtime trace, and the `torchtitan/experiments/graph_trainer/` code:
- `artifacts/graph_dump/final_nodes.jsonl.gz`: one JSON record per node of the graph that runs, after all
  passes: name, op, target, schema, args, kwargs, users, val (shape, stride, dtype), stack_trace,
  nn_module_stack, custom metadata, recompute tag, autograd_backward flag and seq_nr.
  `final_graph.txt.gz` is the same graph as readable code; `pre_inductor_*` is the graph before the final
  Inductor passes.
- `artifacts/traces/iteration_10/rank0_trace.json.gz`: Kineto trace of the profiled step: CPU ops, CUDA
  kernels, input shapes and Python stacks. Every graph node runs inside
  `_RecordFunctionFast("<node name>: <op> (<file:line>)")`.
- `artifacts/codegen/`: the generated code that runs the graph with those labels; `artifacts/pass_list.txt`:
  the passes applied.
- `artifacts/capture_setup.json`, `artifacts/setup.json` and `artifacts/train.log`: the capture's settings
  and log.

Substrate rules: use the graph artifacts, the node metadata and the `torchtitan/experiments/graph_trainer/`
code. Don't read model source (`torchtitan/models/`, `torchtitan/components/`, `torchtitan_recipes/`) to
find candidates. You may open a source line that a node's stack_trace points to, but only to name a region
you already found.

## Rules

- Use only `/home/bahuang/local/step_analysis/llama3_8b_graph`, the repo snapshot, and the source of installed Python packages (PyTorch, torchao,
  dist_moe, ...) to understand what an op does. Don't read anything else on this machine, don't use the
  network, don't use skills, and don't spawn subagents.
- Don't run anything on a GPU and don't rerun training: the artifacts hold everything you need. CPU-only
  Python is fine.
- Don't modify the snapshot or the artifacts. No git commits, uploads or GitHub activity.
- **Time:** run `date` first and note the time. Stop searching at 80 minutes and have every deliverable
  written by 90 minutes.

## Steps

### 1. Attribute

Map every GPU kernel of the profiled step to the op that launched it and from there to:
- module path and layer index;
- phase: `forward` or `backward`; for a backward op, the forward module whose op it differentiates;
- shapes, dtypes, bytes read and written, flops where they apply;
- ideal time and measured/ideal ratio.

Scope: the forward and backward of the profiled step. The optimizer step and anything after backward are out
of scope for regions; report their time in the step budget only.

Coverage check: attributed kernel time must be at least 95% of GPU-busy time in the forward and backward.
Also report GPU idle time.

### 2. Find regions

Find all fusion opportunities, large or small. Don't apply a size threshold; rank afterwards. All of these
kinds count; look for each:
- within one module, across modules, and across layers;
- in the forward, in the backward, and pairing a forward with its backward;
- glue around communication and sharding (copies, casts, packing and unpacking around FSDP or expert-parallel
  collectives), even when a library owns it: integration difficulty goes in `blockers`, not in
  `no_opportunity`;
- prologues and epilogues of GEMMs or attention (tier T3);
- sibling (horizontal) fusion of independent ops that read the same inputs or run back to back.

For each region, state why it can be fused (the dataflow) and what blocks it.

As soon as you identify a region, append one line to `/home/bahuang/local/step_analysis/llama3_8b_graph/out/regions_log.jsonl`:
`{"time": "<ISO time>", "title": "...", "footprint": [...]}`.

### 3. Consolidate

- Merge duplicates and overlapping regions (record `overlaps_with`).
- Rank by `gap_ms_total`.
- Check that every non-GEMM, non-attention, non-communication kernel class in scope is in a region or in the
  `no_opportunity` list with a reason.

## Definitions

- **Phases:** `forward`, `backward`.
- **Scope tags:** `intra-module`, `cross-module`, `cross-layer`, `forward-backward`, `communication-glue`,
  `gemm-adjacent`, `horizontal`.
- **Bitwise tier:** how hard a bitwise-identical fused kernel would be.
  - T0: pointwise, views and casts only.
  - T1: involves reductions; the reduction order must be matched.
  - T2: order-sensitive data movement.
  - T3: needs a GEMM or attention inside.
- **Region ideal time:** external bytes only (region inputs read once, outputs written once), or flops /
  peak if larger. Gap = `(measured - ideal) * instances_per_step`.
- **Evidence:** `measured` if every member kernel's time comes from the trace; `estimated` otherwise. Say
  what was estimated.
- **Footprint** (required for every region): the op instances the region covers, as a list of entries
  `{"phase": "forward" | "backward", "module": "<path>", "layers": [<int>, ...] | "all", "ops": ["<op>", ...]}`.
  - `module`: for a forward op, the innermost module whose forward call runs it; for a backward op, the
    module whose forward op it differentiates. Replace numeric layer indices with `*` (e.g.
    `layers.*.attention.wq`) and list the layers in `layers`. Use `""` for ops outside any module.
  - `ops`: aten op names without namespace or overload (`aten::add` and `aten.add.Tensor` are both `add`;
    in-place ops keep the underscore, `index_put_`). Ops in other namespaces are written `namespace.op`.
- **Description** (required for every region): one or two sentences in terms of the model's computation
  (modules, tensors, shapes). Don't refer to your tools or artifacts: no trace, node, graph, hook or file
  names.

## Output

### `/home/bahuang/local/step_analysis/llama3_8b_graph/out/regions.json`

```json
{
  "setup": {"command": "", "gpu": "", "hbm_gbps": 0, "gemm_tflops_used": {}},
  "step_budget": {"step_gpu_busy_ms": 0, "step_gpu_idle_ms": 0,
                  "forward_backward_busy_ms": 0, "attributed_pct": 0,
                  "by_category_ms": {"gemm": 0, "attention": 0, "communication": 0,
                                     "memory_bound_glue": 0, "optimizer": 0, "other": 0}},
  "regions": [{
    "id": "R-01",
    "title": "",
    "description": "",
    "phases": ["forward", "backward"],
    "scope": ["intra-module"],
    "fusion_kind": "",
    "footprint": [{"phase": "forward", "module": "layers.*.attention", "layers": "all", "ops": ["add"]}],
    "source": "file:line-line",
    "kernels_per_instance": 0,
    "instances_per_step": 0,
    "shapes_dtypes": "",
    "external_bytes_per_instance": 0,
    "measured_us_per_instance": 0, "measured_ms_total": 0,
    "ideal_us_per_instance": 0, "ideal_ms_total": 0,
    "gap_ms_total": 0, "gap_pct_of_step": 0,
    "bitwise_tier": "T0", "bitwise_notes": "",
    "dataflow": "",
    "blockers": "",
    "evidence": "measured|estimated",
    "overlaps_with": [],
    "notes": ""
  }],
  "no_opportunity": [{"what": "", "time_ms": 0, "reason": ""}]
}
```

### `/home/bahuang/local/step_analysis/llama3_8b_graph/out/report.md`

1. **Setup:** the program as you understood it, roofline numbers used.
2. **Step budget:** a table by category, attribution coverage, GPU idle time.
3. **Ranked regions:** all regions in one table (id, title, phases, scope, kind, instances, measured ms,
   ideal ms, gap ms, % of step, tier).
4. **Region details:** a few lines each: what, where, the dataflow that makes it fusable, blockers, the tier
   and why.
5. **No opportunity:** the remaining memory-bound time, with reasons.
6. **Observations about working from these artifacts:**
   - what they made easy and what they made hard;
   - evidence that was missing or needed workarounds;
   - how you identified phase and module for backward ops;
   - false leads you discarded;
   - kinds of opportunity you believe you could have missed;
   - effort: wall time and the scripts you wrote.
7. **Reproduction:** the scripts under `/home/bahuang/local/step_analysis/llama3_8b_graph/out/` and how to rerun them.

## Done when

- Attribution covers at least 95% of GPU-busy time in the forward and backward.
- Every in-scope memory-bound kernel class is in a region or in `no_opportunity`.
- `regions.json` follows the schema above, with a footprint and a description for every region.
- `report.md` is complete.

Reply with the path to the report, the five largest gaps, the number of regions per scope tag, the
attribution coverage, and anything you couldn't complete and why.
