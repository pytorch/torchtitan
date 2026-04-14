# Learnings

High-level guide for autonomous graph optimization experiments.
Updated as experiments reveal patterns and principles.

## Methodology

- Always establish a baseline first before making changes.
- Profile the baseline to understand where time is spent before optimizing blindly.
- Verify numerics after every change with the bitwise deterministic test.
- TPS variance is ~1-3% between runs; re-run to confirm borderline results.
- Dump and inspect the FX graph before designing passes.
- Pre-existing test failure: `test_eager_self_deterministic` for flex_attn variants fails regardless of changes (hash mismatch). Use `-k "not FlexAttn"` or skip those 2 tests.
- `str(node.target)` in FX graphs returns `'aten.op.default'`, NOT `'torch.ops.aten.op.default'`. Use substring matching for target classification.

## What Works

- **Autobucketing reordering** (`schedule_overlap_bucketing` with `collective_bucketing=True`): +3.9% TPS. Reorders collectives relative to compute for better comm/compute overlap. Modest memory increase (+2 GiB). Works on the raw make_fx flat graph. Default parameters are well-tuned — don't over-tune.

## What Doesn't Work

- **Regional inductor for all compute ops** (tag every non-collective node): No improvement when cudagraph is already enabled. Inductor kernel fusion saves launch overhead, but cudagraph already eliminates that.
- **Regional inductor for elementwise-only ops** (skip matmul/SDPA/RMSNorm): No improvement. Collectives fragment the graph into tiny regions (2-3 ops) too small for inductor to optimize.
- **Regional inductor for silu+mul** (MLP activation fusion): +1.1% TPS for Llama3 BUT breaks DSv3 numerics. Inductor's Triton silu kernel doesn't match eager CUDA silu bitwise. `eager_numerics.use_pytorch_libdevice` insufficient.
- **Regional inductor for RoPE** (view_as_complex/real chains): Breaks numerics. Inductor decomposes complex multiplication differently than eager.
- **Removing cudagraph**: -30% TPS. Kernel launch overhead for ~11K+ ops per step is catastrophic.
- **Separate FSDP PG for AG/RS overlap**: No improvement. Autobucketing already handles scheduling.
- **Aggressive autobucketing params**: Worse than default. Over-scheduling causes contention.
- **Dead code elimination**: No benefit. Existing cleanup passes (remove_detach, remove_identity_view, remove_identity_slice) already handle dead code.

## Key Insights

- **Comm/compute overlap is the primary bottleneck** for Llama3 8B with FSDP(4)+TP(2). The baseline graph has 421 AG + 421 RS + 68 AR collectives with no overlap optimization.
- **Cudagraph is essential and non-negotiable**: Without it, kernel launch overhead dominates.
- **Regional inductor + cudagraph is largely redundant**: With cudagraph eliminating launch overhead, fusion provides negligible benefit for this workload.
- **Regional inductor breaks bitwise determinism for transcendental functions**: `exp`, `sigmoid` (silu), complex multiplication all produce different rounding in Triton vs eager CUDA. No inductor config fully fixes this. This limits regional_inductor to ops that are purely algebraic (add, mul, sub — but these are too small to benefit from fusion).
- **Graph structure**: The Llama3 8B graph has ~21.5K lines, 675 matmuls, 910 collectives. Model uses SDPA (not flex_attention), so flex_attn passes are no-ops. Collectives fragment the graph at very fine granularity, preventing useful fusion regions.
- **Autobucketing reduces collective count**: From 1820 to 1382 ops (24% reduction through collective bucketing), in addition to reordering for overlap.
