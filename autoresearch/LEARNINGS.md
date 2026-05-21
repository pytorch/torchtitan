# Autoresearch Learnings

Living document of what works, what doesn't, and how to approach kernel
fusion optimization effectively. The agent owns this file.

Re-read at the start of each loop iteration alongside `IDEAS.md` and
`EXPERIMENT_LOG.md`. Update after meaningful experiments (especially
surprising results, both positive and negative). Keep concise and
actionable — per-experiment details belong in `EXPERIMENT_LOG.md`.

## Methodology

(empty — agent populates)

## Patterns that worked

- **Identity-view elimination is a strict win** (Exp 1, +7.2% tps, -2.0 GiB).
  The AOT FX graph carries ~2k `aten.view.default(x, list(x.shape))` nodes
  from DTensor plumbing. A one-pass peephole that swaps each for its input
  is a few dozen lines, never changes numerics, and shrinks the graph by
  ~60% of all view nodes. The memory drop was the surprise — identity views
  apparently keep extra references live and delay deallocation in compiled
  graphs.

  Implication for future passes: do this **first** in the pass list so
  downstream pattern matchers see a clean graph and aren't tripped up by
  identity-view chains separating producer and consumer.

## Patterns that didn't work

- **Don't trust the pre-pass graph dump for "dead outputs" ideas** (Exp 2).
  `gm.print_readable()` output taken before any user passes shows zero-user
  `getitem` nodes from multi-output ops (e.g. `_scaled_dot_product_flash_attention`'s
  outputs 2..5, 8). These look ripe to delete. They're not — FX's
  `eliminate_dead_code()` already removes them as soon as any pass calls it
  (the producer is impure, but `getitem` itself is pure, so DCE drops it).
  When evaluating "remove dead outputs of op X" ideas, sanity-check against the
  post-DCE graph first; if it's gone, skip.

- **Pure topological "move AG to earliest legal slot" does NOT prefetch** (Exp 3).
  Each `all_gather_into_tensor` has its *input* — the `_to_copy(f32→bf16)` of
  the sharded weight, plus surrounding views — produced layer-locally. So the
  earliest legal slot for the AG is still *inside* the same layer, not above
  the previous layer's wait. Moving 130/421 AGs by ~3.5 slots each made no
  measurable difference (tps within noise). To actually overlap an AG with the
  previous layer's compute, you'd need to hoist the AG's whole producer chain
  (cast + view + AG) above the prior `wait_tensor`, which is a heavier
  transformation. Mark this as `[~]` not `[x]` — Strategy B is still worth a
  future attempt, but it's not a low-effort prefetch.

- **Functional collective targets in the AOT FX graph** are
  `torch.ops._c10d_functional.all_gather_into_tensor.default` (and
  `reduce_scatter_tensor.default`, `wait_tensor.default`). Confirmed by graph
  dump and by `target.__name__` matching at pass-run time. Useful when writing
  any pass that touches communication.

## Tooling tips

- **Dump the post-trace graph from a graph pass.** Register a temporary pass
  in `construct_default_graph_passes` that calls
  `gm.print_readable(print_output=False, include_stride=True, include_device=True, expanded_def=True)`
  and writes the result to `/tmp/autoresearch_graph.txt`. Pair with
  `bash autoresearch/scripts/run_benchmark.sh --training.steps 2`; the file
  appears as soon as the first step finishes tracing. Remove the pass before
  any benchmarking — it costs ~2.5MB of disk per call.
- **Cheap graph-shape recon (no need to read the whole 21K-line file).**
  - Op histogram: `grep -oE "torch\.ops\.[a-z_0-9]+\.[a-z_0-9]+\.[a-z_]+" file | sort | uniq -c | sort -rn`.
  - Communication tally by mesh axis: `grep -E "all_gather_into_tensor.*, N, 'M'" file | wc -l` (group size + name).
  - Dtype cast directions: `grep -oE "dtype = torch\.[a-z0-9]+" file | sort | uniq -c`.
  - Find one layer end-to-end: `grep -n "module_fqn.: 'layers\.0'" file | head` then read ~350 lines.
- **Mesh axis names in the dump are NCCL group tags, not friendly names.**
  Group `'42'` with size 4 is the FSDP `dp_shard` axis; group `'21'` with
  size 2 is the TP axis (matches `data_parallel_shard_degree=4`,
  `tensor_parallel_degree=2`). Verify this any time the parallelism config
  changes.
