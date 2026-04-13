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

## What Works

- **Autobucketing reordering** (`schedule_overlap_bucketing` with `collective_bucketing=True`): +3.9% TPS. Reorders collectives relative to compute for better comm/compute overlap. Modest memory increase (+2 GiB). Works on the raw make_fx flat graph.

## What Doesn't Work

- **Regional inductor for all compute ops** (tag every non-collective node for inductor compilation): No improvement when cudagraph is already enabled. Inductor kernel fusion saves launch overhead, but cudagraph already eliminates that.

## Key Insights

- **Comm/compute overlap is the primary bottleneck** for Llama3 8B with FSDP(4)+TP(2). The baseline graph has 421 AG + 421 RS + 68 AR collectives with no overlap optimization.
- **Cudagraph + kernel fusion is redundant**: With cudagraph, there's no kernel launch overhead to save, so inductor fusion provides negligible benefit.
- **Graph structure**: The Llama3 8B graph has ~21.5K lines, 675 matmuls, 910 collectives. Model uses SDPA (not flex_attention), so the flex_attn annotation pass is a no-op. The graph is pure eager ops wrapped in cudagraph.
- **Weight dtype pattern**: Parameters are stored in fp32, cast to bf16 (_to_copy), then all-gathered. 842 _to_copy ops total.
