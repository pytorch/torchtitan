---
name: compare-profiler-traces
description: Compare a torchtitan graph_trainer profiler trace against a Megatron-LM profiler trace for a specific ProfilerStep. Use when the user provides two PyTorch profiler Chrome traces (.json or .json.gz) and wants a per-step diff — wall time, kernels used, comm/compute overlap, stream breakdown.
---

# Compare Profiler Traces: torchtitan vs Megatron-LM

Both torchtitan's `Profiler` (`torchtitan/tools/profiler.py`) and Megatron-LM's
profiler integration export the same **PyTorch Chrome trace JSON** format
(from `torch.profiler.profile(...).export_chrome_trace`). That lets us compare
them apples-to-apples for a single `ProfilerStep#N` span.

## Stack context (call this out at the top of every report)

The "torchtitan" side of this comparison is specifically:

- **GraphTrainer** (`torchtitan/experiments/graph_trainer/`) — runs an
  AOT-traced FX graph via `--compile.mode aot_fx_trace`, not the default
  torchtitan trainer.
- **SimpleFSDP** (`torchtitan/experiments/graph_trainer/simple_fsdp.py`) —
  graph_trainer's FSDP implementation. **Not `FullyShardedDataParallel` /
  FSDP2.** SimpleFSDP is a graph-friendly FSDP that composes with AOT tracing
  and per-layer AllGather/ReduceScatter, which is why the top-kernel list
  shows many small NCCL calls (one pair per layer) rather than batched ones.

Never attribute comm patterns to "FSDP2" in the report — that would be
incorrect. If you see heavy per-layer `AllGather_RING_LL` with good overlap,
that's **SimpleFSDP + aot_fx_trace**, not FSDP2.

## What you'll receive from the user

- Path A — trace from torchtitan graph_trainer (`{dump_folder}/profile_traces/iteration_{step}/rank{N}_trace.json`)
- Path B — trace from Megatron-LM
- Optional: `--step N` to pick a specific step. If not given, use the **last**
  `ProfilerStep#` in each trace (the "active" step in a standard
  `wait+warmup+active` schedule).

If either trace is from a different rank than the user intends, mention it —
the script prints which `rank*_trace.json` it parsed.

## How to run

The skill ships with `analyze_trace.py`, a pure-Python extractor (stdlib only).
Run it on each trace to get a metrics JSON:

```bash
python "${CLAUDE_PLUGIN_ROOT:-.}/analyze_trace.py" <trace_A> --output /tmp/metrics_a.json
python "${CLAUDE_PLUGIN_ROOT:-.}/analyze_trace.py" <trace_B> --output /tmp/metrics_b.json
```

When invoking from this skill directory, use the path to `analyze_trace.py`
that sits next to `SKILL.md`. Add `--step N` if the user specified one.

Read both JSONs and produce the comparison described below.

## What the extractor gives you

Per trace, you get a JSON blob with these keys (all times in microseconds
unless suffixed `_ms`):

- `step_wall_us` / `step_wall_ms` — end-to-end wall time of the selected `ProfilerStep#`
- `gpu_span_us` / `gpu_span_ms` — first-to-last GPU kernel span within the step. Used as the denominator for all GPU-side percentages; avoids treating a CPU dispatch-then-block tail as GPU idle.
- `gpu.busy_us`, `gpu.idle_us`, `gpu.utilization_pct`, `gpu.compute_busy_us` — union of all kernel intervals vs. `gpu_span`, and the union of non-comm kernel intervals ("compute busy"). Relates to the rest by `gpu_busy = compute_busy + comm - (comm ∩ compute)`.
- `comm.total_us`, `comm.overlapped_us`, `comm.overlap_pct`, `comm.exposed_us`, `comm.exposed_pct_of_step` — NCCL time, how much of it is hidden behind compute, and how much is "exposed" (on the critical path)
- `kernel_categories` — totals per category: `comm`, `gemm`, `attention`, `reduction`, `elementwise`, `memcpy`, `other_kernel`
- `streams` — per-CUDA-stream totals (uses `thread_name` metadata)
- `top_kernels` — top 10 kernels by total time, with category/count/avg/pct

## Producing the comparison

After running the extractor on both traces, write a markdown report with
these sections. Be quantitative — show absolute values AND diffs.

### 1. Step summary

A small table. **Column order: Megatron first (left), torchtitan second (right).**
Apply this convention to every table in the report — it's easier to eyeball
diffs against Megatron as the reference.

Use a single ratio column `tt/mg` — quotient of the two numbers, formatted
with 2 decimals (e.g. `1.04×`, `0.44×`, `2.05×`). Ratios beat absolute
deltas here: they're dimension-free, scale-invariant across workloads, and
instantly convey "tt is 2× worse" vs "tt is 0.4× (i.e. >2× better)". For
percentage-point metrics (GPU util %, overlap %) still use a ratio of the
raw values. Bold ratios ≥1.5× or ≤0.67× to highlight meaningful gaps.

**Direction arrows on metric names.** Append `↑` (higher-is-better) or
`↓` (lower-is-better) to metric-row labels so the reader instantly knows
whether a ratio above 1× is good or bad. Arrows go **only on the
cross-cutting metric tables** (step summary, overlap decomposition).
Defaults:

- `↓` step wall, GPU busy, GPU idle, compute busy, comm busy, exposed comm
- `↑` GPU utilization %, comm/compute overlap %, Comm ∩ Compute (hidden)

**Do NOT add arrows to:**

- Kernel-category breakdown rows (`comm`, `attention`, `elementwise`, etc.).
  These are category labels, not metric names — the reader understands the
  columns are times.
- Themed per-work rows (`SwiGLU fwd+bwd`, `RoPE fwd+bwd`, `Norm fwd+bwd`,
  `Adam`, etc.) — same reason.
- Kernel names in per-kernel tables, stream labels, framework names, or any
  row that's descriptive metadata rather than a cross-cutting metric.

| Metric | Megatron | torchtitan | tt/mg |
|---|---|---|---|
| Step wall (ms) | ... | ... | ...× |
| GPU busy (ms) | ... | ... | ...× |
| GPU utilization (%) | ... | ... | ...× |
| Exposed comm (ms) | ... | ... | ...× |
| Comm/compute overlap (%) | ... | ... | ...× |

### 2. Kernel-category breakdown

Side-by-side category totals. Call out categories where one side spends >10%
more wall time than the other. Consider: is Megatron running fewer, fatter
GEMMs (cudagraphs/fused kernels)? Is torchtitan spending more in `elementwise`
(unfused ops)?

### 3. Notable kernel-level diffs

Do **not** emit a raw "top 10 kernels per profile" listing — it's noise and
duplicates what shows up in the themed diffs below. Instead, use the
`top_kernels` data as your source material and surface only the
comparisons that tell a story: group kernels by role (collectives, GEMM,
casts/copies, SwiGLU, RoPE, RMSNorm, optimizer) and for each group show
the head-to-head with a `tt/mg` ratio. Highlight:

- Kernels present in one but not the other (what replaced them?)
- Kernels where per-call `avg_us` differs — same kernel, different size or
  different arch path?
- Categories where the aggregate ratio is ≥1.5× or ≤0.67×.

#### Per-kernel GEMM breakdown by tile/layout

GEMM kernels on Blackwell (B200) / Hopper (H100) share a naming convention
like `nvjet_*_{tile}_{memory_layout}_{layout_suffix}`:

- `tile` — e.g. `128x256`, `256x256`
- `memory_layout` — `h_bz` (horizontal) vs `v_bz` (vertical)
- `layout_suffix` — `NNT` / `NTT` / `TNT`, encoding the A/B/C transpose pattern
- dtype suffix family — bf16 uses `..._tst_*`; fp8 uses `..._qqtst_*`
  (quantized–quantized) or `..._qrtst_*` (quantized–rowwise)

When GEMM is a top category, always tabulate per-call `avg_us` grouped by
`{tile, memory_layout, layout_suffix}` on both sides. This is what cleanly
separates "kernel quality is fine" from "plumbing around the kernel is
broken" — if per-call GEMM avg µs is competitive but the framework is
still slower, the cost is outside the matmul.

| Tile / layout | Calls (A / B) | A avg | B avg | B/A |
|---|---|---|---|---|
| `128x256 v_bz_TNT` | 97 / 96 | 714.3 µs | 324.4 µs | **0.45×** |

**Layout-suffix collapse is a red flag.** If one side emits a mix of
`NNT`/`NTT`/`TNT` suffixes but the other collapses to a single suffix
(commonly `TNT`), that side is inserting a pre-matmul layout transform.
Pair this observation with the cast/copy diagnostic — the extra transpose
usually shows up as a `direct_copy_kernel_cuda` surge.

#### Themed per-op signatures to probe

Graph-tracer anti-patterns have reproducible kernel fingerprints. When
comparing GraphTrainer against an eager baseline, probe specifically for:

| Op | Healthy (fused) signature | Unhealthy (unfused) signature |
|---|---|---|
| SwiGLU | single Triton fused fwd+bwd kernel | separate `mul` / `silu` / `clone` / grad kernels |
| RoPE | single fused bf16 kernel (TE-style) | f32 complex path: `ComplexMulFunctor`, `cat`, `slice_backward` |
| RMSNorm | single fused fwd+bwd, γβ reduced inline | separate `layer_norm_fwd`, `layer_norm_bwd`, `γβ_reduce` kernels |
| Adam/AdamW | single fused multi-tensor kernel | multi-tensor apply + separate scale/copy kernels |
| Cast/copy | ≤2 per transformer block | 10+ per block → dtype juggling between ops |

When the unhealthy signature dominates on the GraphTrainer side, that's a
direct pointer to a missing fusion pass — regional Inductor coverage, a
Triton-fused op, or an upstream torchao/TE-equivalent kernel.

### 4. Comm / compute overlap

Use the full overlap decomposition, not just "exposed comm". The table
below shows how compute, comm, and their intersection add up to GPU busy,
so the reader can see where the step wall is actually going.

| Metric | Megatron (X ms step) | torchtitan (Y ms step) |
|---|---|---|
| GPU busy (union of all streams) | busy_us — util_pct% | busy_us — util_pct% |
| GPU idle in step | idle_us | idle_us |
| Compute busy | compute_busy_us | compute_busy_us |
| Comm busy | comm.total_us | comm.total_us |
| Comm ∩ Compute (hidden) | comm.overlapped_us | comm.overlapped_us |
| Exposed comm (not hidden) | comm.exposed_us (x% of step) | comm.exposed_us (x% of step) |
| % of comm hidden by compute | comm.overlap_pct% | comm.overlap_pct% |

After the table, comment on what the decomposition shows: is torchtitan's
longer step coming from more comm, more compute, or both? If exposed comm
is low on both sides, the gap is compute-bound. If one side has notably
higher GPU idle, call it out — that's the signal for launch overhead or
a synchronous stall.

### 5. Stream breakdown

Compare the `streams` list. Most PT runs have:
- one default compute stream
- one or more NCCL comm streams
- an extra stream for AllGather/prefetch when FSDP2 is used

If the stream topology differs (e.g. Megatron uses a dedicated copy stream
torchtitan doesn't), surface it.

### 6. Bottom line

In 3–5 bullets: what's the most important thing this diff tells you? Where
is torchtitan leaving performance on the table vs Megatron (or vice versa)?
Keep it actionable — point at specific categories/kernels.

## FP8-specific diagnostics

When either run uses FP8, the most common failure mode is an **un-fused
scaling pipeline**: `amax → scale → quantize → matmul → dequantize` emitted
as separate graph nodes instead of fused into the GEMM prologue/epilogue
(as TransformerEngine does). This can make FP8 *slower* than BF16 even
when the FP8 GEMM kernel itself is faster per call. If you see
elementwise time explode on the FP8 side relative to BF16, walk through
this checklist before blaming the GEMM:

1. **amax pipeline as independent kernels.** Large launch counts of
   `AbsFunctor<BFloat16>` and `reduce_kernel<…BFloat16…>` — expect on the
   order of `n_layers × (W + A + G) × (fwd + bwd)` launches each. On a
   32-layer model that's ≈1,300 per kernel. In TE-based Megatron these
   never appear standalone — they're folded into the GEMM wrapper.
2. **Standalone `ConvertToFloat8E4M3fnOp<float>` kernels.** If the fp8
   cast fires as its own kernel once per tensor per step, fusion is
   absent. TE collapses this into the GEMM epilogue.
3. **`direct_copy_kernel_cuda` count explosion.** Compare launch count
   bf16-vs-fp8 *within the same framework*. 8–10× more `direct_copy`
   launches on fp8 means every `fp8_quant → matmul → fp8_dequant` is
   being marshalled through explicit dtype casts instead of staying
   in-kernel.
4. **Layout suffix collapse** (see the GEMM tile/layout table in §3).
   A bf16→fp8 path that collapses `NNT`/`NTT` into `TNT` is inserting a
   pre-matmul layout transform, which pairs with (3).
5. **Scale-factor broadcast kernels.** Watch for `elementwise_kernel<128,
   2, …nocast>` (two-input) and `elementwise_kernel<128, 4, …nocast>`
   (four-input) near the GEMM site — these are tensor × scale broadcasts
   and result rescaling that TE folds into the GEMM.

When multiple signatures are present, the recommendation writes itself:
fix the FP8 lowering so amax/scale/quantize/dequantize collapses into a
single fused FP8 GEMM call with scale factors passed as kernel args. Don't
waste time on kernel-level tuning until the plumbing is fused — per-call
FP8 GEMMs are usually already competitive.

## Pitfalls to watch

- **Megatron `--use-nccl-ub` hides AllGather in peer-copy events.** NCCL
  user-buffer registration + symmetric memory bypasses the traditional
  ring/tree AllGather kernel and issues direct `cudaMemcpyAsync` peer
  copies between ranks. In the Chrome trace these appear as
  `Memcpy PtoP (Device -> Device)` under the `gpu_memcpy` category — not
  as `ncclDevKernel_AllGather_*`. Rough sanity check: one coalesced
  AllGather on 8 ranks produces ~7 peer copies, so `n_AG × 7` should
  approximately match the PtoP event count (e.g. 132 AG × 7 ≈ 900 PtoP).
  `analyze_trace.py` reclassifies `Memcpy PtoP` as comm so `comm.total_us`
  and `comm.overlap_pct` stay correct, but if you see `ncclDevKernel_*`
  counts that look absurdly low (e.g. only 2 per step on the Megatron
  side) call it out — the AllGathers are still happening, just under a
  different mechanism. Non-peer-to-peer memcpys (HtoD / DtoH / DtoD
  within a device) stay in the memcpy category.
- **Different batch sizes / parallelism configs make the comparison meaningless.**
  Before diving into the diff, skim step wall and top GEMM sizes. If they're
  wildly apart, flag it to the user rather than pretending the numbers are
  comparable.
- **Rank 0 vs average**: the user almost always gives rank 0. Communication
  kernels look different on non-root ranks (ReduceScatter vs AllGather timing).
  If they hand you multiple ranks, compare like-for-like.
- **NCCL stream naming varies by PyTorch version** — the extractor merges by
  category, so this shouldn't affect numbers, but the per-stream breakdown
  might look different.
- **Warmup contamination**: the first `ProfilerStep#` in an active schedule
  is usually clean, but if the user picks step 0, they may be measuring
  compile/cudagraph-capture overhead. Warn if `step_wall_ms` on either side
  is >2× the other.
