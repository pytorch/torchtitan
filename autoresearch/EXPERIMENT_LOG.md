# Experiment Log

Cumulative log of all experiments. Never overwrite previous entries.

## Baseline — keep (4463e48)

- **Idea**: Establish baseline performance for Llama3 8B with aot_fx_trace, FSDP(4)+TP(2) on 8 GPUs.
- **Changes**: No changes. Default graph passes: tlparse logging, remove_detach, remove_identity_view, remove_identity_slice, annotate_flex_attention_for_regional_inductor, regional_inductor, cudagraph.
- **Result**: TPS=6938, MFU=40.63%, Memory=46.90GiB
- **Analysis**: This is the reference point. The graph already has regional_inductor compiling flex attention regions, and cudagraph wrapping the entire graph.
- **Lessons**: Starting point for all future experiments. Need to profile to understand where time is spent.

## Regional Inductor for all compute ops — discard (xxxxxxx)

- **Idea**: Tag all non-collective call_function nodes with `compile_with_inductor` so regional_inductor compiles them into fused Inductor kernels. The baseline had no inductor compilation for the 8B model (it uses SDPA, not flex_attention, so the flex_attn annotation pass is a no-op). Expected benefit: reduced kernel count and memory bandwidth through op fusion.
- **Changes**: Added `annotate_all_compute_for_regional_inductor_pass` that tags all `torch.ops.aten.*` nodes (excluding `_c10d_functional`, `_dtensor`, `device_mesh` ops) with `compile_with_inductor`. Inserted before `regional_inductor_pass` in the pass list.
- **Result**: TPS=6927 vs baseline 6938 (-0.16%), Memory=46.90GiB, MFU=40.57%. Numerics pass (all 4 aot_fx_trace_vs_eager tests pass).
- **Analysis**: No meaningful improvement. cudagraph already eliminates kernel launch overhead, so fusing ops into fewer kernels doesn't help the launch path. The fused kernels might save memory bandwidth for small ops, but the overall computation is dominated by large matmuls and SDPA which inductor doesn't improve.
- **Lessons**: With cudagraph enabled, kernel fusion via regional_inductor provides minimal benefit since launch overhead is already amortized. The bottleneck must be elsewhere — likely in the compute/communication overlap (or lack thereof). Need to profile to identify actual bottlenecks. Pre-existing test failure: `test_eager_self_deterministic` for flex_attn variants is broken regardless of our changes (hash mismatch in expected values).

## Autobucketing reordering pass — keep (5b1ae19)

- **Idea**: Add `autobucketing_reordering_pass` (which calls `schedule_overlap_bucketing` with `collective_bucketing=True`) to improve comm/compute overlap. The flat fwd+bwd graph has 421 all-gathers + 421 reduce-scatters + 68 all-reduces — substantial communication that could overlap with compute.
- **Changes**: Added `autobucketing_reordering_pass` to the default pass list after `regional_inductor_pass` and before `cudagraph_pass`.
- **Result**: TPS=7214/7198 (two runs, avg ~7206) vs baseline 6938 (+3.9%), Memory=48.95GiB (+2.05GiB), MFU=42.24%. Numerics: all 4 aot_fx_trace_vs_eager tests pass.
- **Analysis**: The bucketing pass reorders collectives and compute ops to maximize overlap. The +3.9% improvement confirms that comm/compute overlap was a significant bottleneck in the baseline. The memory increase is modest and acceptable.
- **Lessons**: Comm/compute overlap is the primary optimization lever for this workload. The autobucketing pass from inductor's overlap scheduling infrastructure works well on the make_fx flat graph. Future experiments should explore whether more aggressive overlap strategies (manual bucketing, separate PGs for AG/RS) can stack on top.

