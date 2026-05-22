# Autoresearch Learnings

Living document of what works, what doesn't, and how to approach kernel
fusion optimization effectively. The agent owns this file.

Re-read at the start of each loop iteration alongside `IDEAS.md` and
`EXPERIMENT_LOG.md`. Update after meaningful experiments (especially
surprising results, both positive and negative). Keep concise and
actionable — per-experiment details belong in `EXPERIMENT_LOG.md`.

## Methodology

- **Re-recon every keep that materially reshapes the graph.** Exp 11 (cat→view, 130 elisions) and Exp 13 (291 fwd-AG → 33 coalesced + 291 bwd-RS → 33 coalesced + hoisted producer chains) both invalidated prior "next-target" priorities. Post-keep, re-dump with a temporary pass at the END of `construct_default_graph_passes`, diff op histograms vs the previous dump (single-line `grep ... | sort | uniq -c | sort -rn | head` and `diff`), and update IDEAS.md before kicking off the next experiment. Keep the diff in the dump-file path: e.g. `/tmp/post_cleanup_graph.txt`, `/tmp/post_exp11_graph.txt`, `/tmp/post_exp13_graph.txt` — each dump is the post-all-passes state up to and including that experiment's accepted pass.
- **Histogram raw line-grep "x.y.default = torch.ops..." for actual call sites only.** Plain `grep -c torch.ops.X.Y` overcounts because the dump references each call multiple times (assignment line, the right-hand side, and the `... = None` self-clean line at the end of each call). Use `grep -cE "X[a-z_0-9_]* = torch\.ops"` to count assignment LHS only — gives the true call-count.

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

## Run summary (after Exp 18, 2026-05-22 ~03:20 UTC)

**Cumulative tps: 4,161 → 5,124 = +23.1%** over raw baseline, via four committed passes:
1. `remove_identity_views` (Exp 1, +7.2%) — drop ~2k identity DTensor views.
2. `remove_identity_ops` (Exp 7, +2.4%) — full-range slice + double-transpose cancellation.
3. `elide_split_cat_for_reduce_scatter` (Exp 11, +1.58%) — TP cat-as-view.
4. `bucket_fsdp_collectives` (Exp 13, +10.4%) — coalesced AG/RS + producer-chain hoist.

13 of 18 experiments discarded. The decisive structural wins (Exps 11 & 13) came from finding zero-copy / launch-collapsing rewrites grounded in byte-layout proofs (Exp 11) and placeholder-rooted hoist trees (Exp 13). Pure-cleanup bundles ceiling at +0.6% (Exp 17). Metadata peephole well is dry. Collective-coalescing without hoistable producers is dead (Exps 12/15/16). FX-level AG reordering doesn't translate to runtime overlap (Exps 3/5/6/8). Bitwise-numerics constraint forbids RoPE fp32 round-trip elimination and any precision-modifying rewrite. The next significant wins would require either (a) custom HOPs with backing kernels (out of scope), (b) FSDP/model config changes (out of scope), or (c) a structurally novel pattern not yet identified — Exp 18's open-ended search returned no_target.

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

- **Coalescing only works when producer chains are hoistable, which means tracing to placeholders** (Exps 12-16).
  Five coalesce attempts (Exp 4 cat-based, Exp 12 ws=4 RS, Exp 13 ws=4 AG+RS, Exp 15 AR, Exp 16 ws=2 TP) showed the universal pattern: critical-path-gated coalescing without hoisting yields 0-few buckets when the wait_tensor of each member is immediately consumed by intervening compute. Only Exp 13 worked because forward FSDP weight AGs have producer chains that trace back to graph placeholders — fully hoistable to the layer's start. All other collectives (TP ws=2, bwd RMSNorm γ-grad AR, bwd activation-grad RS) have producer chains that depend on deep compute outputs (e.g. `_fused_rms_norm_backward`, `mm`) and can't be hoisted across the prior compute barrier. Stop chasing more coalescing — the placeholder-rooted opportunity (291 forward AGs + 291 backward param-grad RSes) was the only one and Exp 13 captured it.

- **Coalesced AG + producer-chain hoist gives +10% — the prefetch we couldn't achieve via reordering** (Exp 13).
  Coalescing forward FSDP AGs alone yields zero buckets if you respect critical-path constraints: every AG's wait is immediately consumed by `mm`, so a "no consumer between bucket members" gate rejects all coalescing. The trick is to **hoist each AG's private producer chain (`_to_copy(placeholder)`) before the bucket's anchor** so the coalesced AG can issue at the layer's start with all 9 weights. NCCL transfers in parallel with the entire layer's compute — this IS the cross-layer prefetch that node-reordering (Exps 3/5/6/8) couldn't deliver, because here the coalesce is itself the single barrier point.
  Bundling forward AG hoist+coalesce with backward RS coalesce (Exp 12 mechanism) in one pass: tps 4,641 → 5,124 (+10.4%), mfu 27.17 → 30.00, memory unchanged. Numerics bitwise-identical. This is **the largest single win of the run**, and it works because the launch overhead + the lost-overlap of 291 separate small AGs compound — once you remove BOTH, the layer's compute critical path is no longer punctuated by 9 wait_tensor blocks.

- **`reduce_scatter_tensor_coalesced.default` exists as a functional collective** (Exp 12).
  Schema: `(Tensor[] inputs, str reduce_op, int group_size, Any group_name) -> Tensor[]`.
  Takes a list of input tensors and returns a list of output tensors in a single
  collective launch — no cat/split overhead required. Use when bucketing
  same-direction RSes. Equivalent op likely exists for AG (verify by
  `dir(torch.ops._c10d_functional)`); look for `all_gather_into_tensor_coalesced`.

- **Bandwidth-bound coalesced collectives on NVLink-FSDP=4 hit a ceiling around +1%**
  (Exp 12). 291 RSes → 33 coalesced calls (avg bucket size 8.82), 100% coverage,
  no overhead. Win: only +0.80% — just below the +1% noise threshold. NCCL
  apparently still issues per-tensor reductions internally, so the FX-level
  launch-count reduction doesn't fully translate to NCCL-level launch savings.
  Implication: **forward AG coalescing (same mechanism) likely faces the same
  ceiling**. To clear +1% via coalescing, bundle forward AG + backward RS in
  one pass so the gains compound. Don't expect a standalone forward AG coalesce
  to do better than backward RS coalesce did.

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
- **To find what dtype a collective traffics in, grep its input shape.**
  `grep -B1 "reduce_scatter_tensor.default.*'sum', 4," file | grep -oE
  '"(f32|bf16)\[[^]]+\]' | sort | uniq -c | sort -rn` instantly reveals
  the per-shape distribution (and total bytes). Caught the post-Exp11
  finding that all 291 param-grad RSes are **f32** (preceded by a bf16→f32
  cast for FSDP's reduce_dtype=f32 policy), totalling ~10 GiB/step of RS
  bandwidth — twice what bf16 would cost. The same recipe applied to
  AG-ws-2 (TP fwd activations) shows them all as bf16, confirming where
  the dtype-policy boundary sits.
