# Measurement

Most of the mistakes in this work were measurement mistakes, not code mistakes. This file is the procedure plus every trap we fell into.

## Ideal time

For a memory-bound op: ideal = (bytes of every input read once + every output written once) / measured bandwidth.

- Measure bandwidth once with a large copy (GB300: ~7.1 TB/s; H100: ~3.35 TB/s). Spec-sheet numbers overstate what kernels reach.
- Count fwd and bwd separately. A pointwise op's backward reads the grad and saved inputs and writes input grads.
- Example: multiplying `[32768, 7168]` bf16 by a per-row score reads 470 MB and writes 470 MB: 940 MB / 7.1 TB/s = 132 µs. The eager broadcast kernel took 507 µs (3.8x), which made it the first target.
- For GEMMs and attention use a FLOP floor and the achieved rate of the best kernel you know (GEMMs reach ~1.8-2.0 PFLOP/s bf16 on GB300).
- "Gap to ideal × share of the layer" ranks candidates. A 3x gap on 1% of the layer matters less than 1.3x on 15%.

## Correctness

- Rel L2 of the output and every grad vs eager, run once in bf16 and once in fp32, plus an fp64 reference. Do the fp32 and fp64 checks at op level: an fp32 layer-level run with flex autotuning took over 30 min.
- Tests that fail before a perf change: region-vs-eager parity at op level (within ~1 bf16 ulp); traced placeholder shapes are concrete when a dim is meant to be static (`mark_static`); graph-count assertions over the shapes the model uses.
- MoE layers: routing flips make raw comparisons noisy. Replay the eager routing in the compiled run (patch the model's own router class: DSv4 overrides `_select_experts`, so patching the base class did nothing) and compare against a noise band from a 1-ulp perturbation of the input.
- Don't dump full MoE grads for comparison (53 GB per dump on DSv4); compare per-tensor norms, samples or hashes.
- Op-level first (isolates the kernel), then layer-level (neighbors, host time, memory).

## Traps

| Trap | What happened | Fix |
|---|---|---|
| Detached step inputs | the residual stack needed no grad, so one form looked 2 ms faster in eager when it was 5.75 ms slower | inputs require grad as in training; treat any `.detach()` on step inputs as suspect |
| Grads accumulate across steps | bf16 accumulation adds, ~1.3 ms per DSv3 MoE layer, inflated times and "% of layer" | set grads to None every step |
| Unseeded `torch.empty` buffer | cross-process accuracy comparisons were meaningless | seed before building inputs and weights |
| Summing kernels by name | two 7 µs scatters showed as one 14 µs entry | check kernel counts before attributing |
| "fp64 reference" calls `.float()` inside | silently computed in fp32 | check the reference's dtypes |
| Routing replay keyed by call count | broke under AC recompute (the router runs again) | key by layer and step |
| fp32 matmul precision differs from the trainer's | fp32 router/gate GEMMs ran as SIMT kernels; wrong baseline | set `torch.backends.cuda.matmul.fp32_precision` to what this tree's trainer sets |
| Another job hits mid-run | a run that started clean showed a 2x outlier wall; start/end `nvidia-smi` checks miss short jobs (one landed inside a timed loop) | poll `nvidia-smi` during the run; report the median of processes within ~1% of the best, with n shown |
| A crashed run reusing the last result | the harness printed the previous RESULT line after a crash | write a FAILED marker per run |
| Order effects | a variant always run second carried warm caches or a slower host | rotate order or use fresh processes |
| Pooled batches | pooling a harness batch with a PR-tree batch hid a 1.4% vs 0.3% disagreement | report batches separately |
| Timing a harness copy | the shipped code computed an extra in-graph inverse permutation (~18 µs) | time the code you ship |
| World-1 smoke before a cluster job | a config check that fires only at EP > 1 killed every rank | build at the target world size first |

## Profiling and trace diffs

- Profile eager, regions and block compile on the same inputs. Group kernels by family (GEMM, attention, norm, cat/copy, pointwise, reduction, scatter/gather, sort).
- The regions-vs-block diff is your list of glue that a region could still take. Typical leftovers: cat kernels (`CatArrayBatchedCopy`), broadcast muls, `index_put`/scatter backward, casts, standalone adds.
- Look for host syncs: `torch.cuda.set_sync_debug_mode("warn")`, `cudaStreamSynchronize` and DtoH memcpy in the trace. Boolean-mask indexing, `.item()`, `.tolist()`, `repeat_interleave` without `output_size`, and blocking copies from pageable or pinned memory all sync.
- Under CUDA graphs, measure replay time too: replay ≈ kernel time + a few µs per kernel, so kernel-count cuts still help a little.

## Replicates and hygiene

- **On a shared GPU,** wrap each fresh-process run in a wait-for-quiet-GPU + retry loop (poll `nvidia-smi`, rerun contaminated rows); that gave clean numbers. When the GPU never goes quiet, interleaved A/B inside one process still gave correct relative deltas under 2.3x contention.

- **Screen in one process, confirm in three.** In-process A/B (variants interleaved) is good for ranking. Cited numbers are medians of 3 fresh processes.
- **Busy shared GPU and many branches:** a full fresh-process matrix per branch may not fit. A fallback we used: clean 3-process medians for main vs the final stack, plus single-run per-branch deltas labelled "indicative".
- **Host-bound regimes.** At 4k tokens per rank, eager plus many small kernels can be host-bound; walls swung 1.5-3x with other agents' CPU load. Report GPU kernel time alongside wall, and an interleaved in-process A/B to show a region's calls pay for their host cost.
- **Record per row:** wall and kernel time, kernel count, peak memory, graph count, foreign GPU processes, commit hash.
- **Label everything:** shape, fwd+bwd or fwd, wall or kernel, process count, GPU. "13% faster" without these is not a claim.

## Graph budget

- Count graphs for every shape the model calls the region with (all layers, routed vs shared experts, different head counts), for every training token count, and for a `no_grad` pass (grad mode is a guard, so validation adds a graph per shape class).
- `recompile_limit=8` with `fullgraph=True` is a hard failure, not a fallback. Headroom helps (we aimed for ≤ 6 of 8). Raising the limit (we used 16 in several models) costs more compiles and more cached graphs, and how to do it depends on the tree:
  - newer upstream main: a per-region `recompile_limit` where the region supports it (e.g. `fused_binary_activation`);
  - trees that pass `local_compile` kwargs straight to `torch.compile`: there is no such argument there;
  - otherwise: `torch._dynamo.config.recompile_limit`, globally.
- Graph count also tells you whether a symbolic size crept in (a static graph followed by a dynamic one).
- Every new call signature costs one graph per grad mode.
- A region shared by call sites of different rank or layout multiplies graphs; give call sites stable layouts or separate regions.
- `triton.mix_order_reduction_non_strict_mode=True` stops a scheduler heuristic from installing guards on symbolic sizes (worst case 8 -> 6 graphs, time-neutral).
- Keep fixed dims static (`mark_static` inside the region) and mark the token dim dynamic in eager before the region (patterns.md 5).
- How to count: harness.md.

## Cold compile

- Measure with caches disabled (`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`), per graph. Autotuning options (coordinate descent) can double compile time; an unrolled loop can add minutes (a per-entry Sinkhorn rewrite added ~3 min cold).
- Report it next to the win, as a trade-off ("~11 s more cold compile for 0.5% of step"; DSv4 per-entry unrolling: ~+42 s for 2% of step, stack total 48.8 -> 159.7 s) and let the owner decide.
