# Autoresearch Experiment Log

Cumulative log of every experiment in this run. Append-only — never
overwrite previous entries. Re-read at the start of each loop iteration to
learn from past experiments and avoid repeating failed approaches.

## Format

```markdown
## <short title> — <status> (<commit hash>)

- **Idea**: What optimization was attempted and why it was expected to help.
- **Changes**: What was actually modified (brief summary, not a full diff).
- **Result**: Perf numbers (tps, MFU, memory_gib, wall_time_s) or crash/error description.
- **Analysis**: Why it worked, why it didn't help, or why it broke.
- **Lessons**: Key takeaways — what to build on, what to avoid, what to try.
```

---

## Baseline — keep (baseline)

- **Idea**: Establish the Run 1 baseline. Llama3 8B with FSDP=4 TP=2 bs=1, `construct_default_graph_passes` returns an empty list. The agent will iterate from this raw aten-op graph.
- **Changes**: None to `passes.py`. Infrastructure-only commit.
- **Result**: tps=4,161, mfu=24.36%, memory=49.0 GiB (52% of H100), wall_time=75s; loss=9.21808, grad_norm=4.5867 (last step of 20, `c4_test` dataset).
- **Analysis**: Bitwise-identical loss/grad_norm across runs with `--debug.seed 42 --debug.deterministic`. Memory is comfortable (31 GiB headroom). There is meaningful headroom above this floor — the agent's job is to find it. Wall time ~75s per benchmark → ~1100 iterations in 24h.
- **Lessons**: FSDP=4 TP=2 bs=1 chosen over bs=2 because of larger optimization headroom and safer memory budget (31 GiB free vs 12 GiB).

---

## apply_inductor_pattern_passes — keep (pending)

- **Idea**: PyTorch's Inductor ships several FX-level pattern matchers (`joint_graph_passes`, `post_grad_passes`, etc.) that fuse common ATen sequences (mm+bias+activation, SDPA epilogues, `_to_copy` round-trips, …) by rewriting the GraphModule in place. They were designed for the same kind of joint fwd+bwd graph that `make_fx` produces, so they should apply directly.
- **Changes**: Added `apply_inductor_pattern_passes(gm, example_inputs)` that tries 5 upstream entry points in sequence, each wrapped in its own `try/except`: `joint_graph_passes`, `post_grad_passes(is_inference=False)`, `pre_grad_passes`, `binary_folding_pass`, `dedupe_symint_uses_pass`. Each logs availability and before/after node counts. Registered AFTER `remove_detach_nodes`.
- **Result**: keep. Two consecutive runs: tps=4,569 (mfu=26.75%) and tps=4,576 (mfu=26.80%); avg **tps=4,572 (+9.9% vs baseline 4,161)**. Loss=9.21808, grad_norm=4.5867 (bitwise identical). Numerics test passes. Wall time 77s.
- **Node-count trace**:
  - initial: 11,538
  - after `joint_graph_passes`: 9,106 (−2,432, −21%)
  - after `post_grad_passes(is_inference=False)`: 9,101 (−5)
  - `pre_grad_passes`: ran no-op (pre-grad targets a different graph form)
  - `binary_folding_pass`, `dedupe_symint_uses_pass`: not available in this PyTorch build
- **Analysis**: This is the first iteration to move TPS meaningfully — the −21% node reduction translates to a clean +10% TPS. Earlier topology-only FX edits (bucket/prefetch) failed to translate to TPS, but Inductor's pattern matchers are doing real *semantic* fusion (combining producer/consumer ATen ops into single fused ops, removing entire chains of casts, etc.), not just reordering. The pattern matchers also know how to fuse the `_to_copy` round-trips that iter 2 couldn't touch, which likely accounts for a large chunk of the gain.
- **Lessons**:
  - **Reuse upstream Inductor passes before writing your own.** PyTorch already ships pattern matchers that took years to tune; rewriting from scratch is wasteful.
  - The next big wins likely come from running additional Inductor pipeline stages — full **codegen via `compile_fx`** (Triton kernels for pointwise), and **CUDA graphs** for static replay.
  - `joint_graph_passes` is the right entry point for joint fwd+bwd graphs out of `make_fx`. `post_grad_passes` does small extra cleanup on top.
  - The two unavailable passes (`binary_folding_pass`, `dedupe_symint_uses_pass`) suggest the installed PyTorch version is older than the ones that exposed those names. Future iterations should be careful about API drift.

---

## prefetch_all_gathers (FX-level move) — discard (xxxxxxx)

- **Idea**: Move each `all_gather_into_tensor` node to its earliest valid FX position (right after its inputs). The launch happens on a separate NCCL stream, so intervening compute should overlap with the AG transfer; the `wait_tensor` stays at its original position.
- **Changes**: `prefetch_all_gathers(gm, example_inputs)` walks the graph, computes each AG's earliest-valid position from `A.all_input_nodes`, and moves the AG via the FX doubly-linked-list API. Distance capped at 200 to bound memory peak. Registered after `remove_detach_nodes`.
- **Result**: discard. step 20: tps=4,177 mfu=24.46% memory=47.03GiB; loss=9.21808, grad_norm=4.5867. Pass moved only **65 of 421 AGs**, average distance **7** positions, max **8**. The 200-cap never triggered. Numerics test pass.
- **Analysis**: `make_fx` already places AGs immediately after their input shards are computed. Without also hoisting the AG's **input ops** (`_to_copy(view(param))` patterns) earlier in the graph, the AG cannot move. The 7-position headroom is essentially insignificant.
- **Lessons**:
  - **FX-level reordering alone cannot prefetch FSDP all_gathers**: the AG input chain (cast + view of a placeholder param) is what pins the AG in place. To prefetch effectively we must (a) move the input ops too, OR (b) reorder at a different level (NCCL stream priorities, separate compile step), OR (c) attack the **wait_tensor side** instead — move waits LATE toward consumers so AG↔wait gap widens.
  - This also explains iter 3's bucket result: even after hoisting input ops, the shape-heterogeneity meant most AGs stayed singletons. Together they suggest FX-level scheduling of FSDP collectives has limited slack — the dominant gains live in **fusion of the compute** (kernel side) or in **whole-graph compile via inductor** (which does its own scheduling), not in shuffling existing nodes.

---

## bucket_all_gathers (dim-0 shape-grouped) — discard (xxxxxxx)

- **Idea**: Bucket consecutive `_c10d_functional.all_gather_into_tensor.default` calls (421 of them) to amortize launch overhead. Combine inputs via `aten.cat(..., 0)`, do one all_gather, then `view → slice along dim 1 → reshape` to recover per-original outputs preserving rank-major layout.
- **Changes**: Added `bucket_all_gathers` pass after `remove_detach_nodes`. Bucket size cap = 8; grouping by `(group_size, group_name, dtype, device, len(shape), shape[1:])`. Hoisting strategy: input ops (`_to_copy`, `view`, `_unsafe_view`, `t`, `transpose`, `clone`, `reshape`) of later bucket entries get moved to before the first AG so the cat input is computable in-order. Non-pure-op input chains cause flush. Slice→reshape returns non-contiguous → used `aten.reshape.default` (may copy) instead of `view`.
- **Result**: discard. step 20: tps=4,169 mfu=24.41% memory=47.03GiB; loss=9.21808, grad_norm=4.5867 (bitwise identical). Numerics test pass. 421 AGs → 64 bucketed calls (160 AGs bucketed, 261 left as singletons).
- **Analysis**: Shape heterogeneity is the limiter. FSDP unsharding emits AGs with widely varying `shape[1:]` (per-layer: `(16032,4096), (1024,), (1,4096,4096), (512,4096), (128,4096), (128,4096), (1024,2048)` etc.), so the shape-compatible grouping rule rarely finds 3+ neighbors. Average bucket size 2.5. The reduction from 421 → 64+261 = 325 collective launches (~23% fewer launches) is real but does NOT show up as tps. Two likely reasons: (i) the reshape after slice copies data (non-trivial extra work), wiping out the savings; (ii) the unbucketed AGs still dominate runtime — bucketing 38% of nodes leaves 62% untouched.
- **Lessons**:
  - **Shape-grouped dim-0 bucketing has structural limits.** To bucket aggressively across all AGs we must FLATTEN inputs to 1D before cat, then carefully reshape back. The flat path also dodges the non-contiguous slice problem (a 1D slice is contiguous).
  - **Per-original reshape can cost as much as the launch savings.** If a bucketed approach forces a copy per output, the net is ~zero. Future bucketing should arrange so each per-original output is recoverable via a contiguous view (no copy).
  - **The wins from reducing launches require also overlapping with compute.** Without prefetching, a synchronous AG+wait pair still blocks on data arrival. **Bucketing without prefetch ≠ free perf.**
  - **Next ideas to pursue:** (a) AG prefetch — move AG nodes earlier so communication overlaps with intervening compute, leaving the wait at the original position; (b) inductor regional compile on pointwise hot paths; (c) CUDA graphs.

---

## remove_redundant_to_copy — discard (xxxxxxx)

- **Idea**: 842 `aten._to_copy.default` nodes appear in the graph. If any are no-op casts (target dtype/device equals source dtype/device), removing them is safe and cheap.
- **Changes**: Added `remove_redundant_to_copy(gm, example_inputs)` that walks all `_to_copy.default` nodes, compares requested kwargs against the input's `meta["val"]`, and erases only when dtype/device/layout all match AND no user is in-place. Registered after `remove_detach_nodes`.
- **Result**: discard. 0 of 841 candidate `_to_copy` nodes qualified. tps/loss/grad_norm bitwise identical to iteration 1 (4180 tps, loss 9.21808, grad_norm 4.5867). Numerics test passes.
- **Analysis**: All `_to_copy` nodes in this graph are genuine fp32 → bf16 mixed-precision casts (or vice versa), not redundant. The conservative criterion correctly rejected all of them.
- **Lessons**: The 842 `_to_copy` count cannot be reduced by pure elimination; it must be attacked by **(a)** detecting bf16→fp32→bf16 round-trips and collapsing them, **(b)** fusing the cast into producer/consumer kernels (e.g., via inductor regional compile), or **(c)** restructuring the graph so casts happen once instead of repeatedly. Future iterations should investigate the round-trip pattern.

---

## remove_detach_nodes — keep (ae6e425)

- **Idea**: The recon counted 196 `aten.detach.default` nodes. The runtime executes the traced graph under `torch.no_grad()` (see `trainer.py: _make_fx_forward_backward_step`), so detach has no autograd effect at all. Removing them should reduce dispatcher work and free held tensor references.
- **Changes**: Added `remove_detach_nodes(gm, example_inputs)` to `passes.py`. Collects all `aten.detach.default` call_function nodes, rewires users via `replace_all_uses_with(node.args[0])`, erases, then runs `eliminate_dead_code()` / `lint()` / `recompile()`. Registered as the first pass in `construct_default_graph_passes`.
- **Result**: 196 detach nodes removed. Numerics test passes (`aot_fx_trace_vs_eager`). Two benchmark runs: tps=4,202 (mfu=24.60%) and tps=4,174 (mfu=24.44%). Both produce loss=9.21808, grad_norm=4.5867 — bitwise identical to baseline. Memory drops from 49.0 → 47.0 GiB consistently.
- **Analysis**: TPS delta is within the 1-3% noise band, so the speedup is at best marginal. The robust effect is the ~2 GiB memory drop: each detach node held a Python reference to its input via the FX graph, preventing the underlying tensor from being released during graph execution. Numerics are unaffected because detach shares storage with its input, so users that read the same data after removal are reading identical bytes.
- **Lessons**: (1) Removing FX nodes that hold references can free real memory even when the op itself is "free." (2) detach.default in joint fwd+bwd graphs is essentially always dead under run_traced's no_grad. (3) Future cleanup passes should focus on bigger node categories (e.g. 842 `_to_copy.default`, 196 (now 0) detach was the smallest cleanup target).

---

## Graph reconnaissance — discard (xxxxxxx)

- **Idea**: Before picking concrete optimizations, count FX node targets in the joint fwd+bwd graph to find the highest-leverage areas. No optimization is attempted in this iteration.
- **Changes**: Added a temporary `_recon_dump_graph_stats` pass that counts call_function targets and writes the top-40 ops, collective ops, and view-like ops to `/tmp/recon_graph_stats.txt`. Pass returns `gm` unchanged. Reverted after capturing the stats.
- **Result**: discard — no perf change attempted (single inspection run with `--training.steps 3`). Confirmed baseline still passes through unchanged.
- **Analysis**: See LEARNINGS.md "Graph shape snapshot" for the full table. Headline numbers: 11.9k nodes; 421 all_gather_into_tensor + 421 reduce_scatter_tensor + 68 all_reduce; 196 detach + 842 _to_copy; 32 SDPA flash fwd+bwd; 65 fused_rms_norm fwd+bwd. The collective counts are the obvious lever — about 13 unbucketed FSDP all-gathers per layer × 32 layers.
- **Lessons**: Pass list ordering matters: anything we add to `construct_default_graph_passes` runs after make_fx_tracer's `_remove_cpu_shadow_chains`, so the recon numbers above already exclude shadow chains. Future cleanup passes should not target CPU shadow nodes.
