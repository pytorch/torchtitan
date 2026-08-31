---
name: cuda_graph_trace_compaction
description: Make a CUDA-graph PyTorch profiler trace readable in Perfetto by packing replay streams into semantic lanes, and optionally merge per-rank pipeline-parallel traces into one aligned timeline with send/recv flow arrows. Use when a trace captured with CUDA graphs shows hundreds of one-kernel stream rows, or when the user wants to view several PP rank traces together.
---

# CUDA Graph Trace Compaction

CUDA graph replay gives each captured region its own stream, so a profiler
trace opens in Perfetto as hundreds of near-empty rows.
`scripts/compact_cuda_graph_trace.py` repacks the slices into a few semantic
lanes (compute, NCCL all-gather, reduce-scatter, ..., memcpy, memset).

**Visualization only.** Slices keep their name, `pid`, `ts`, `dur`, and `args`
-- only the display placement (`tid`/`args.stream`) changes. Hand the user the
original for measurement and the compacted copy for reading.

The script is agent-maintained, not a torchtitan API -- edit it for the trace
in front of you rather than working around it. Common tweaks: `_classify_slice`
and `GROUP_ORDER`/`GROUP_LABELS` for new lane groupings, `PP_ANNOTATION_PATTERN`
for a different annotation scheme. Keep the tests passing.

```bash
# Single trace -> <name>_compacted.json.gz beside the input
python3 .claude/skills/cuda_graph_trace_compaction/scripts/compact_cuda_graph_trace.py \
    outputs/<run>/profile_trace/iteration_10/rank0_trace.json.gz

# All rank*_trace.json[.gz] in a directory -> pp_traces_merged_compacted.json.gz
python3 .claude/skills/cuda_graph_trace_compaction/scripts/compact_cuda_graph_trace.py \
    outputs/<run>/profile_trace/iteration_10 --merge-pp-ranks
```

`-o` sets the output path, `--overwrite` replaces an existing one.

Single-rank compaction includes rank-local PP arrows (`UNSHARD -> compute`,
`RECV -> compute`, and `compute -> SEND`). Cross-rank `SEND -> RECV` arrows
are added only when merging PP ranks.

Merging aligns ranks on a common `baseTimeNanoseconds`, stacks them in
pipeline order, and draws flow arrows between the
`PP:<stage><op><microbatch>` annotations that `torch.distributed.pipelining`
emits (`SEND_F -> RECV_F` across ranks, `RECV_F -> F -> SEND_F` within one).
It also connects each `PP:<stage>UNSHARD` annotation on the first two
all-gather lanes to the first subsequent same-stage compute annotation on the
first compute lane.
No annotations means no arrows, but the merge still works.

The script prints a JSON summary and validates itself, raising if a measured
slice changed or a lane ended up with overlapping slices. Worth a look:
`original_streams` vs `compacted_lanes` for the compaction ratio, and
`pp_send_recv_unmatched` (nonzero usually means a truncated capture window,
not a pipeline bug).

Rank order comes from a `torchtitan_real_pp` group or a `mesh_pp` group
description in `distributedInfo.pg_config`; absent that it sorts by global
rank, which is only correct when PP rank order matches global rank order.

## Tests

`pytest .claude/skills/cuda_graph_trace_compaction/scripts/` -- hand-built
traces, no GPU. Outside `testpaths`, so `pytest tests/` skips it. Run it after
touching the script; a broken invariant yields a plausible-looking wrong trace
rather than an error.
