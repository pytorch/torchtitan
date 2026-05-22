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

- **Identity-slice + double-transpose elimination** (Exp 7, +2.4% tps).
  Extends Exp 1 to two more peephole patterns. 100% of `aten.slice.Tensor`
  nodes (421/421) were full-range DTensor local-extraction slices. 40% of
  `aten.t.default` nodes (450/1125) were back-to-back `t(t(x))` cancelling
  pairs (mostly from backward grad accumulation). Identity casts (`_to_copy`
  with same source/target dtype) had zero matches — every `_to_copy` is
  real bf16↔fp32 traffic. Memory unchanged here (unlike Exp 1's -2.0 GiB),
  because slice/t are metadata-only ops — the win is launch/dispatch
  overhead, not deallocation.

  Tooling tip: combine multiple safe peepholes in a single `remove_identity_ops`
  pass, apply slice→double-t→identity-cast in that order so earlier
  eliminations expose later opportunities, end with a single
  `eliminate_dead_code() + lint() + recompile()`.

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

- **At FSDP=4 on NVLink, the Q/K/V AGs are bandwidth-bound, not launch-bound**
  (Exp 4). Bucketing Q/K/V (3→1 AGs/layer, 64 launches saved/step) only moved
  tps +0.7% — below noise threshold. The Q shard is 64 KB and dominates the
  collective time; K/V are 4 KB each but ride along on the same NCCL link.
  The cat+split surgery also adds CPU/launch overhead and (if not careful)
  reshape-on-non-contiguous-strided-slice copies. Implication: launch-count
  bucketing is only a clear win on truly tiny collectives (e.g. RMSNorm γ at
  ≤4 KB per shard). Bandwidth-bound bucketing needs a *coalesced* collective
  op (single launch, no cat/split overhead) to beat the current cost.

- **FX `graph.inserting_after(n)` inserts in REVERSE topological order** when
  multiple nodes are inserted with the same anchor. To insert a chain
  A→B→C after n, do `inserting_after(n) ... A`, then `inserting_after(A) ... B`,
  etc. Otherwise C appears before A in node order and crashes the runtime
  even when `lint()` passes (lint only checks SSA, not collective dependencies).

- **`numerics.log` passing does not guarantee benchmark runs cleanly.** The
  bitwise-deterministic test uses a smaller config than the benchmark, so
  some topo-order violations only surface at full scale. Always run BOTH the
  numerics test AND the benchmark before declaring keep/discard.

- **For "cross-layer" prefetch, the anchor is layer-end RS, not previous-AG-wait**
  (Exp 5). Within a Llama3 layer, weight AGs+waits (Q,K,V,wo,attn_norm,...) are
  packed tightly at the *start* of the layer. Choosing "previous AG's wait" as
  the prefetch destination yields only intra-layer movement (~10 slots, ~1
  barrier). The true layer boundary is the `wait_tensor` for the *output
  reduce_scatter* (post-MLP residual), which marks the end of compute for
  layer i. Hoisting next-layer's AG producer chain above THAT wait gives a
  full layer's worth of compute to overlap. Implementer's TODO: identify the
  RS-wait at the layer-end (group `'21'` on the activation tensor, downstream
  of `w2`'s mm), and use it as the prefetch barrier.

- **`Graph.move_before` does not exist on this PyTorch.** Use
  `prev_wait.prepend(node)` to insert before, or
  `prev_node.append(new_node)` to insert after. Walk movable subgraphs in
  topological order over `gm.graph.nodes` to preserve relative ordering.

- **NCCL group-name strings (`'42'`, `'21'`, `'19'`) are NOT stable across
  runs** (Exp 6). They depend on the order in which process groups were
  created in this process. Two consecutive benchmark runs gave different TP
  group tags. Match on **group size** (`args[1]` of the functional collective,
  e.g. world_size=4 for FSDP, 2 for TP) instead of the group-name string.

- **Per-layer RS waits in the AOT graph come in pairs** (Exp 6). Each
  Llama3 layer emits TWO TP `reduce_scatter` barriers: one after `wo`
  (mid-layer, post-attention) and one after `w2` (end-of-layer, post-FFN).
  Counting RS-waits in fwd+bwd of 32 layers gives ~130 total. For
  cross-layer prefetch you want only the **end-of-layer (post-`w2`)** ones —
  every *other* RS-wait. A naive "most recent RS-wait" anchor lands at a
  mid-layer barrier most of the time, giving only ~1 barrier of overlap,
  which is still intra-layer.

- **Hoist distance and barriers-crossed are the diagnostic** for prefetch
  passes. After moving, log avg slots and avg barriers crossed per AG.
  If barriers-crossed averages ~1.0, you're moving across a *single*
  intra-layer barrier — the hoist target is wrong. Expect 1.5-2.0+ for
  cross-layer prefetch to be meaningful.

- **TP `split→cat→reduce_scatter` cat is provably a view** (Exp 11, +1.58% tps).
  When the split source `x` is contiguous, `cat([split[0], split[1]], dim=0)`
  has byte-layout identical to `x.view([W, N, D])` for `W=2, N=split_size, D`.
  Proven element-by-element: cat-element `(r, i, j)` lands at byte
  `r*N*D + i*D + j`, which matches `x`-element `(0, r*N+i, j)` at byte
  `(r*N+i)*D + j`. Replacing the cat with a view is zero-copy, NCCL accepts
  it, and numerics are bitwise-identical. The pattern fires after every
  TP-sharded matmul (130×/step in this graph) — generalizes to any
  "split-along-inner → cat-along-outer-to-stack-for-RS" sequence-parallel
  scatter idiom. Memory floor doesn't drop because the cats are short-lived
  intermediates; the win is launch-count + memcpy bandwidth (~8.3 GiB/step
  of avoidable allocation/copy traffic). Always verify the split source's
  `meta["val"].is_contiguous()` before rewriting; skip otherwise.

- **Sub-200-ops/step metadata cleanup is below noise floor** (Exps 9, 10).
  Exp 9 removed 194 detach nodes (+0.2%). Exp 10 removed 160 nodes (96 view
  + 32 _conj + 32 clone CSE, +0.4%). Both well under +1% threshold. The
  combined removal of ~350 small ops/step from these two passes is invisible
  in tps. **Don't pursue more passes that target <500 ops/step of pure
  cleanup unless they hit a peak-memory anchor (like Exp 1 did with views)
  or unlock a bigger downstream rewrite.** Future wins need to target high-
  volume traffic, not high-count-low-cost cleanup.

- **FX-level AG reordering does NOT translate to runtime overlap on this
  graph** (Exps 3, 5, 6, 8 — four attempts, all noise-level or negative).
  Even after reaching the diagnostic target (avg ~1.44 barriers crossed
  per AG with end-of-layer-only anchor in Exp 8), tps did not improve and
  in fact regressed slightly. Either NCCL stream scheduling is already
  serialized at the runtime level (so moving the AG node earlier doesn't
  create earlier-on-stream issue), or the AOT-trace runtime doesn't issue
  collectives onto a separate non-blocking stream as expected. **Do not
  spend more iterations on AG-prefetch via FX node reordering** — change
  strategy: either reduce the *number* of AGs (bucketing with coalesced
  ops) or reduce the *cost* per AG (eliminate the fp32→bf16 cast prefix).

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
- **Finding CSE / duplicate-op opportunities cheaply.** For any op category,
  histogram the *exact argument strings* (e.g.
  `grep -oE "torch\.ops\.aten\._conj\.default\([a-z_0-9]+\)" file | sort | uniq -c | sort -rn`)
  — any line with count >1 is a duplicate-call candidate. Same trick on
  `view.default(<src>, <shape>)` finds repeated `view(view_K, [...])` from
  one source. This caught the `_conj(view_X)`×2 backward-RoPE pattern and
  the 3-way duplicate `view([8192,4096])` feeding Q/K/V matmuls.
- **Post-cleanup vs pre-cleanup op-histogram diff** is a one-line check
  that tells you which ideas from the original recon are now stale: pipe
  both `grep -oE "torch\.ops..." | sort | uniq -c | sort -rn | head -30`
  outputs through `diff`. View count drop (3358→1352) and slice drop
  (421→0) make their respective recon ideas already-explored; ops that
  *didn't* shrink (e.g. `_to_copy` 842→841, `transpose.int` 256→256) are
  the surviving structural patterns worth a fresh pass.
