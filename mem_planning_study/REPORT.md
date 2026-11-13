# Is global memory planning worth it for GraphTrainer? A directional study

Workload: DeepSeek V3 671B, MXFP8 linears, Dist-MoE experts, GraphTrainer (whole-step
joint graph), seq 4096, one microbatch per DP rank, AdamW with bf16 moments.
Parallelism: FSDP256 x EP64 (SPMD), plus PP2 x VPP4 x FSDP128 x EP64 for the pipeline
study. A smaller model, Qwen3 30B-A3B (FSDP8 x EP8, 8 microbatches per step), is in
section 5. All numbers are logical rank 0 (SPMD) or PP rank 0 / 1, measured with the
fake distributed backend on one GB300 (288 GB) per process, steady state (step >= 2).

Reference: STWeaver / STAlloc, "Reducing GPU Memory Fragmentation via Spatio-Temporal
Planning for Efficient Large-Scale Model Training" (arXiv 2507.16274), code at
github.com/infinigence/STAlloc.

## TL;DR

1. **Fragmentation is small on this workload.** The PyTorch caching allocator wastes
   1.7-4.1 GiB (memory efficiency E = live peak / reserved = 96.1-98.4%).
   `expandable_segments` (already the default in `run_train.sh`) cuts that to
   1.2-2.4 GiB (E = 97.6-99.2%). The paper's 43% waste does not reproduce here.
2. **Planning with the paper's algorithm** packs the per-step transient memory to
   0-7.4% above the liveness lower bound (greedy-by-size: 0-0.5%). End to end, the
   measured saving is -1.3 to +2.7 GiB vs the caching allocator and -1.8 to +1.0 GiB vs
   expandable segments. Even with persistent tensors perfectly packed (projection), the
   best case is ~1-2.4 GiB (1-2.3% of peak) vs expandable segments.
3. **Peak memory is dominated by live bytes, not fragmentation.** Of the 101 GiB SAC
   peak, 62.1 GiB is one Dist-MoE buffer that Dist-MoE already plans statically,
   ~29 GiB is model state, and only 10-20 GiB is per-step transient. Planning cannot
   reduce live bytes.
4. **The paper's recompute observation reproduces only weakly**: SAC (partial
   recompute) has the lowest efficiency (96.1%), but no-recompute and full recompute
   are both ~98.4%. In the paper's own setting, pipeline training (PP2 x VPP4, 8
   microbatches), E is 97.0-99.5% with SAC or full recompute (paper: ~90% / ~60%).
   STAlloc packs the PP transient memory to 0.13-3.6% over the lower bound. The best
   projected saving vs expandable segments is 0.9-4.2 GiB (0.5-3.0% of peak).
5. **A smaller model does not change the picture.** On Qwen3 30B-A3B (FSDP8 x EP8,
   8 microbatches per step, the size class the paper evaluates), E is 97.8-98.9%
   with the caching allocator and 98.4-99.2% with expandable segments. STAlloc
   hits the lower bound exactly for none and SAC, and the best projected saving vs
   expandable segments is 2.0-2.8 GiB (0.8-1.5% of peak). Of the 250.8 GiB
   no-recompute peak, 123.5 GiB is the Dist-MoE buffer.
6. **Numerics and throughput are unchanged.** A real 2-GPU run with the planned
   allocator active matches the caching allocator bitwise (loss and grad_norm, 20
   steps, both ranks); step time 232.0 ms vs 237.5 ms.
7. **Recommendation:** an allocator-level planner is not worth productionizing for
   this setup. The integration cost is high (see "Technical challenges") and the
   achievable win is 0.5-3% of peak across all three setups. Higher-ROI memory work is in live bytes: the
   end-of-backward gradient double buffer (9.8 GiB in the full-recompute peak), the
   full-vocab fp32 log-softmax (~6 GiB, chunked loss), and saved attention outputs
   under SAC (7.7 GiB).

## 1. Without planning: peak memory and fragmentation

| memory policy | live peak | caching allocator: reserved (E, waste) | expandable segments: reserved (E, waste) |
|---|---|---|---|
| none (no recompute) | 179.6 GiB | 182.8 GiB (98.3%, 3.2 GiB) | 180.9 GiB (99.2%, 1.5 GiB) |
| default (SAC) | 101.3 GiB | 105.5 GiB (96.1%, 4.1 GiB) | 103.8 GiB (97.6%, 2.5 GiB) |
| full recompute | 101.2 GiB | 102.9 GiB (98.4%, 1.7 GiB) | 102.4 GiB (98.8%, 1.2 GiB) |

Production setting (SAC, CUDA graphs on, caching allocator): the step runs from the
graph's private pool; reserved 105.6 GiB for the same ~101.3 GiB live peak (4.3 GiB,
E = 95.9%).

Where the memory is (SAC, start of a step: 91.45 GiB live):

- 62.1 GiB: one Dist-MoE device buffer (57.1 GiB MoE activation slots + 5.1 GiB
  scratch), planned by the Dist-MoE annex memory planner; fragmentation-free.
- ~29.3 GiB: parameters, gradients of the previous step (9.8 GiB, freed by
  `zero_grad`), AdamW moments.
- Transient per step: peak 19.6 GiB (SAC), 19.5 GiB (full), 97.8 GiB (none).

What sets the transient peak (attributed by FX node, from the allocator trace):

- **none**: end of forward; 97.8 GiB of saved activations (MXFP8 GEMM outputs and
  quantized copies 34 GiB, MLA q/k rope 23 GiB, Dist-MoE weight prep 10 GiB,
  attention 7.7 GiB, norms 7.5 GiB, ...).
- **SAC**: end of forward; attention outputs 7.7 GiB, residual adds 3.3 GiB,
  full-vocab fp32 log-softmax + NLL backward ~5.9 GiB (this recipe uses plain
  CrossEntropyLoss), Dist-MoE 1.4 GiB.
- **full**: end of backward; activations are negligible, the peak is the gradient
  handoff: 9.8 GiB in-graph `_to_copy` (gradient cast) + 9.8 GiB graph outputs.

So SAC and full recompute have nearly the same peak for different reasons, and the
full-recompute peak is not an activation problem at all.

## 2. With planning (the paper's algorithm)

Prototype pipeline (all in `mem_planning_study/`, nothing in torchtitan core changed):

1. **Trace** (`memplan_alloc.so` in trace mode): a `CUDAPluggableAllocator` records
   the exact alloc/free sequence of two consecutive steady-state steps.
2. **Plan** (`planner.py`): one train step is the planning window. Blocks freed in the
   next step (gradients, freed by `zero_grad`) wrap around the step boundary, which
   replaces STAlloc's cross-iteration interval shifting. Planners:
   - `stalloc`: port of STAlloc's default synthesizer path
     (`allocate_group(None, ..., group=False)`): exact-size clusters in descending
     size; each cluster first fills free time/space of larger layers (skyline), after
     STAlloc's failed-first reordering trials; the rest form new layers with the
     paper's Algorithm 1.
   - `greedy`: greedy-by-size best fit (reference).
   Every plan is validated for time x address overlap before use.
3. **Execute** (`memplan_alloc.so` in plan mode): one preallocated arena; the k-th
   allocation of a step gets its planned offset after checking size, stream, thread
   and a live-range overlap guard; anything else (persistent tensors, mismatches)
   goes to `cudaMallocAsync`.

Planner results (transient memory of one step):

| policy | plannable blocks/step | wrapping (grads) | distinct sizes | liveness lower bound | STAlloc arena (overhead) | greedy arena (overhead) | plan time |
|---|---|---|---|---|---|---|---|
| none | 25,551 | 850 (9.8 GiB) | 99 | 97.80 GiB | 97.80 GiB (0.0%) | 97.80 GiB (0.0%) | 17 s / 1.4 s |
| SAC | 33,700 | 850 (9.8 GiB) | 99 | 19.64 GiB | 21.09 GiB (7.4%) | 19.73 GiB (0.4%) | 1.0 s / 1.3 s |
| full | 34,182 | 850 (9.8 GiB) | 99 | 19.53 GiB | 20.77 GiB (6.3%) | 19.53 GiB (0.0%) | 5.1 s / 1.2 s |

End-to-end, measured with the plan active (reserved = arena + fallback pool peak):

| policy | caching | expandable | planned STAlloc (measured) | planned greedy (measured) | projected STAlloc / greedy (persistent perfectly packed) |
|---|---|---|---|---|---|
| none | 182.8 (98.3%) | 180.9 (99.2%) | 182.4 GiB (98.4%) | - | 179.5 / 179.5 GiB (100%) |
| SAC | 105.5 (96.1%) | 103.8 (97.6%) | 104.1 GiB (97.3%) | 102.7 GiB (98.6%) | 102.8 (98.6%) / 101.4 GiB (99.9%) |
| full | 102.9 (98.4%) | 102.4 (98.8%) | 104.2 GiB (97.1%) | - | 102.5 (98.8%) / 101.2 GiB (100%) |

- Plan hit rate: 33,699-34,182 hits per step, 0 misses; the overlap guard redirected
  one allocation every other step to the fallback pool (benign, safety net working).
- The measured residual splits into STAlloc packing overhead (0-1.45 GiB) and
  fragmentation of the prototype's fallback pool (1.3-2.9 GiB): persistent tensors
  (optimizer state created lazily in step 1) were interleaved with step-1
  temporaries in the `cudaMallocAsync` pool, and trimming cannot release those
  pages. STAlloc avoids that by also laying out weights, gradients and optimizer
  state linearly; the "projected" column assumes that.
- Against the default in `run_train.sh` (expandable segments), the best projected
  saving is 1.4 GiB (none), 2.4 GiB (SAC, greedy) and 1.2 GiB (full).
- STAlloc's layering is worse than plain greedy-by-size whenever wrap-around
  (gradient) blocks are present: they fragment its same-size layers.

## 3. Recompute vs efficiency (the paper's observation)

The paper reports ~90% efficiency for plain pipeline training and ~60% with
recompute. On DSV3 SPMD (caching allocator):

| policy | E (caching) | E (expandable) |
|---|---|---|
| none | 98.3% | 99.2% |
| SAC (partial recompute) | 96.1% | 97.6% |
| full recompute | 98.4% | 98.8% |

The direction holds for partial recompute (SAC interleaves short-lived recomputed
tensors with long-lived saved ones), but the magnitude is ~10x smaller. Three reasons:
GraphTrainer frees every intermediate at its last use in a fixed graph order (no
Python/autograd timing noise); Dist-MoE takes the bulk of activation memory out of
the caching allocator; and full recompute leaves too few long-lived activations to
fragment around.

## 4. Pipeline (PP2 x VPP4)

Setup: PP2 x VPP4 (Interleaved1F1B, 8 virtual stages of 8 layers) x FSDP128 x EP64,
8 microbatches of 4096 tokens per step, `real_pp_fake_spmd` (one process per PP rank,
real PP send/recv, SPMD axes fake). This is the paper's main setting: Megatron-style
VPP with and without recompute.

Where the memory is at the peak (SAC, PP rank 0, 162.8 GiB live): 75.9 GiB Dist-MoE
buffer (9 per-microbatch activation slots of 7.9 GiB + 5.1 GiB scratch; rank 1: 7 slots,
60.2 GiB), ~45 GiB model state, ~41.6 GiB per-step transient.

| PP rank, policy | live peak | caching (E, waste) | expandable (E, waste) | planned STAlloc (E) | planned greedy (E) | projected STAlloc / greedy |
|---|---|---|---|---|---|---|
| rank 0, SAC | 162.8 | 165.2 (98.6%, 2.3) | 163.7 (99.5%, 0.9) | 165.1 (98.6%) | - | 162.8 / 162.8 (100%) |
| rank 1, SAC | 143.8 | 147.5 (97.5%, 3.6) | 147.7 (97.3%, 3.9) | 146.0 (98.5%) | - | 143.8 / 143.8 (100%) |
| rank 0, full | 153.8 | 157.0 (98.0%, 3.1) | 155.8 (98.7%, 2.0) | 156.7 (98.1%) | 156.0 (98.5%) | 154.9 (99.2%) / 153.9 (99.9%) |
| rank 1, full | 137.0 | 140.4 (97.6%, 3.4) | 141.1 (97.0%, 4.2) | 139.2 (98.4%) | 138.7 (98.7%) | 137.5 (99.6%) / 136.9 (100%) |

All values in GiB.

Planner results (transient memory of one step, 8 microbatches):

| PP rank, policy | plannable blocks/step | wrapping (grads) | distinct sizes | lower bound | STAlloc (overhead) | greedy (overhead) | plan time STAlloc / greedy |
|---|---|---|---|---|---|---|---|
| rank 0, SAC | 96,990 | 428 (9.5 GiB) | 88 | 41.61 GiB | 41.66 (0.13%) | 41.61 (0.003%) | 29 s / 8 s |
| rank 1, SAC | 98,085 | 424 (10.1 GiB) | 85 | 38.84 GiB | 38.90 (0.14%) | 38.85 (0.004%) | 28 s / 8 s |
| rank 0, full | 114,566 | 428 (9.5 GiB) | 92 | 32.61 GiB | 33.78 (3.6%) | 32.72 (0.33%) | 28 s / 10 s |
| rank 1, full | 115,621 | 424 (10.1 GiB) | 89 | 31.97 GiB | 32.53 (1.8%) | 31.97 (0.00%) | 24 s / 10 s |

- **Efficiency is 97-99.5% with or without recompute**, not the paper's ~90% / ~60%.
  Recompute does not lower E here: full recompute 97.0-98.7%, SAC 97.3-99.5%.
- **STAlloc packs the PP SAC step almost perfectly (0.13% over the lower bound)**.
  Eight microbatches repeat the same sizes in the same order, which is the case
  Algorithm 1 handles well. Full recompute adds short-lived recomputed tensors in
  backward, and STAlloc's overhead rises to 1.8-3.6%; greedy stays at 0-0.3%.
- Measured savings vs the caching allocator: 0.0 / 1.5 GiB (SAC, rank 0 / 1),
  0.3 / 1.2 GiB (full, STAlloc), 0.9 / 1.7 GiB (full, greedy). The fallback pool's
  fragmentation (1.7-2.3 GiB) again absorbs most of the arena's gain.
- Best projected saving vs expandable segments: 0.9 GiB (rank 0, SAC), 3.9 GiB
  (rank 1, SAC), 1.9 GiB (rank 0, full), 4.2 GiB (rank 1, full). That is 0.5-3.0% of
  peak. Rank 1 gains more because expandable segments does worse there (E 97.0-97.3%).
- Plan hit rate: every planned allocation hit (96,990-115,621 per step, 0 misses,
  0-1 overlap fallbacks per step). Step time: SAC 7.8 s caching vs 7.1 s planned;
  full 11.6 s vs 11.1 s (STAlloc) / 11.8 s (greedy). These are noisy fake-PG timings
  from one host.
- PP rank plots have no fwd/bwd colors: the graph-info dump was off for PP runs, so
  the transient tensors show as "other".

## 5. Smaller model: Qwen3 30B-A3B

Question: does the paper see 60-90% efficiency because its models are small? The paper
evaluates Llama2-7B, Qwen1.5-MoE-A2.7B (14B total), Qwen2.5 7B-72B and Llama-3.1
8B / 70B on Megatron-LM, DeepSpeed and ColossalAI. Qwen3 30B-A3B is close to its MoE case.

Setup: port of `graph_trainer_qwen3_30b_a3b_dist_moe_mxfp8_8gpu` (agentic_compiler
branch, `torchtitan_recipes/graph_trainer/megatron_bridge_benchmarks.py`, matched to
Megatron-Bridge's GB300 MXFP8 recipe). It has 48 MoE layers and MXFP8 Dist-MoE experts
(Dist-MoE fuses SwiGLU into its expert GEMMs), MXFP8 attention linears, a BF16 LM head
with chunked loss, and BF16 Adam moments. Parallelism is FSDP8 x EP8 with round-robin
routing. GBS 512 x 4096 and MBS 8 x 4096 give 8 accumulated microbatches per step per
rank. Fake PG with NGPU=8, logical rank 0. Differences from the source recipe:

- CUDA graphs are off so every allocation goes through the allocator under study.
  Turning them on fails config validation on `main`: the recipe still carries an
  `AllToAllTokenDispatcher` config, which accelerator graphs reject.
- The recipe's recompute is "none". SAC and full recompute were added for the paper's
  recompute comparison.

Where the memory is (no recompute, 250.8 GiB live peak):

- 123.5 GiB: one Dist-MoE device buffer. One 32K-token microbatch's expert
  activations at capacity factor 1 take 118.5 GiB, plus 5.0 GiB scratch.
- ~28.5 GiB: parameters and AdamW moments. EP spans all 8 ranks, so FSDP cannot
  shard the experts further.
- 14.2 GiB: accumulated gradients (wrap around the step boundary).
- Transient peak of the step: 98.8 GiB (none), 50.3 GiB (SAC), 40.3 GiB (full).

| policy | live peak | caching (E, waste) | expandable (E, waste) | planned STAlloc (E) | planned greedy (E) | projected STAlloc / greedy |
|---|---|---|---|---|---|---|
| none | 250.8 | 253.7 (98.9%, 2.9) | 252.7 (99.2%, 2.0) | 254.4 (98.6%) | - | 250.8 / 250.8 (100%) |
| SAC | 202.3 | 206.8 (97.8%, 4.5) | 205.1 (98.6%, 2.8) | 204.5 (98.9%) | - | 202.2 / 202.2 (100%) |
| full | 192.3 | 196.6 (97.8%, 4.3) | 195.4 (98.4%, 3.2) | 195.4 (98.4%) | 194.7 (98.7%) | 193.2 (99.5%) / 192.6 (99.8%) |

All values in GiB.

| policy | plannable blocks/step | wrapping (grads) | distinct sizes | lower bound | STAlloc (overhead) | greedy (overhead) | plan time STAlloc / greedy |
|---|---|---|---|---|---|---|---|
| none | 100,903 | 437 (14.2 GiB) | 58 | 98.80 GiB | 98.80 (0.00%) | 98.80 (0.00%) | 27 s / 8 s |
| SAC | 125,495 | 437 (14.2 GiB) | 58 | 50.27 GiB | 50.27 (0.00%) | 50.27 (0.00%) | 14 s / 13 s |
| full | 142,166 | 437 (14.2 GiB) | 60 | 40.28 GiB | 41.21 (2.3%) | 40.59 (0.76%) | 16 s / 13 s |

- **Model size is not the explanation.** On this smaller MoE model the caching
  allocator reaches E = 97.8-98.9% and expandable segments 98.4-99.2%. Recompute
  lowers E by about one point (98.9% -> 97.8%), not 90% -> 60%.
- **The 8-microbatch step is the easy case for STAlloc**: the same sizes repeat in the
  same order 8 times, and the plan is exactly at the lower bound for none and SAC.
- **Measured gains are again eaten by the fallback pool**: 2.2-3.6 GiB of `cudaMallocAsync`
  fragmentation around the 152 GiB of persistent tensors. Projected (persistent tensors
  packed), the best saving vs expandable segments is 2.0 GiB (none, 0.8% of peak),
  2.8 GiB (SAC, 1.4%) and 2.8 GiB (full, greedy, 1.5%).
- Plan hit rate: 100,902-142,166 hits per step, 0 misses, at most 1 overlap fallback
  per step. Step time (4 concurrent fake runs, noisy): none 8.1 s caching vs 7.4 s
  planned; SAC 8.6 s vs 8.6 s; full 11.5 s / 9.1 s expandable / 9.3 s planned.
- Measured with the plan active: SAC is the one setting where the prototype beats
  both baselines (204.5 GiB vs 206.8 caching / 205.1 expandable). For none, the
  3.6 GiB of fallback-pool waste makes it 0.7 GiB worse than caching.
- What remains different from the paper: (1) routing is round-robin and Dist-MoE
  buffers are capacity-sized, so expert token counts never vary. The paper names
  variable MoE token counts ("dynamic layers") as a main fragmentation source.
  Learned routing under fake PG is not usable (it ties), so this needs a real
  multi-GPU run with a dispatcher whose buffers follow the routed token count.
  (2) GraphTrainer frees each tensor at its last use in a fixed order, where
  Megatron's eager autograd, communication buckets and Python-side tensors make
  the order irregular.

## 6. Visualizations

Address (y) x time (x) for every allocation of one train step. x is the allocation
ordinal within the step (identical sequence in both runs). Left: caching allocator,
with all `cudaMalloc` ranges stacked (persistent-heavy ranges at the bottom), so the
plot's height is reserved memory and light grey is reserved-but-unused. Right:
planned layout; the bottom band is the fallback pool (persistent tensors, drawn as a
filled live-bytes area), the top band is the arena with exact planned offsets. The
strips underneath compare live bytes with reserved.

- SAC, STAlloc plan: `plots/sac_stalloc.png`
- SAC, greedy plan: `plots/sac_greedy.png`
- No recompute, STAlloc plan: `plots/no_recompute_stalloc.png`
- Full recompute, STAlloc plan: `plots/full_recompute_stalloc.png`
- PP2 x VPP4, SAC, STAlloc plan: `plots/pp2vpp4_sac_rank0_stalloc.png`,
  `plots/pp2vpp4_sac_rank1_stalloc.png`
- PP2 x VPP4, full recompute, STAlloc / greedy plan:
  `plots/pp2vpp4_full_rank{0,1}_{stalloc,greedy}.png`
- Qwen3 30B-A3B, no recompute (the recipe's setting), STAlloc plan:
  `plots/qwen3_30b_none_stalloc.png`. The 8 accumulated microbatches are 8 forward
  triangles; the accumulated gradients are the red bands that live across the step.
- Qwen3 30B-A3B, SAC, STAlloc plan: `plots/qwen3_30b_sac_stalloc.png`
- Qwen3 30B-A3B, full recompute, STAlloc / greedy plan:
  `plots/qwen3_30b_full_{stalloc,greedy}.png`

What they show: the caching allocator scatters same-size activations across many
segments (the jagged "V"); the plan stacks them into dense per-size triangles that
grow through forward and unwind through backward. The visual difference is large;
the reserved-height difference is 0.4-2.7 GiB because the caching allocator already
reuses that memory well.

## 7. Numerics and throughput

Real 2-GPU run (not fake): DSV3 debugmodel, MXFP8, Dist-MoE, FSDP2 x EP2 (the
GraphTrainer Dist-MoE integration-test layout), learned router, seed 42,
`debug.deterministic = True`, 20 steps, plan active from step 3:

- loss and grad_norm (compared as exact float hex) are **bitwise identical** on both
  ranks for all 20 steps across the caching allocator, the planned allocator
  without a plan, and the planned allocator with the plan (3,424 planned allocations
  per step, 0 misses).
- Median step time: caching 237.5 ms, planned 232.0 ms. `cudaMallocAsync` alone
  (trace mode) is slower, 259.5 ms; serving from the arena avoids driver calls.
- 671B fake runs (noisy: four CPU-bound runs shared the host): SAC 3.07 s caching vs
  2.6-3.0 s planned; none 2.48 s vs 2.36 s; full 2.63 s vs 2.69 s.

## 8. What a real implementation would look like

The prototype replays a trace by allocation ordinal (like STAlloc). With
GraphTrainer, the plan can instead come from the graph:

1. A graph pass after memory policy, remat and bucketing (the `extra_passes` slot)
   computes storage-level lifetimes (`GraphAliasTracker` in
   `torch/_inductor/fx_passes/memory_estimator.py` already does alias-aware
   liveness), adds stream-aware ends for collectives (until `wait_tensor`) and
   wrap-around ends for step outputs (gradients), and runs the packer.
2. Each allocating node gets an arena offset; non-graph allocations (optimizer,
   data, Dist-MoE kernels' internal temporaries) stay on the caching allocator.
3. Execution: either a planned pool inside the CUDA caching allocator (allocate the
   k-th request of a capture-like region at a fixed offset), or codegen that writes
   node outputs into arena views (`out=` variants), as Inductor's `memory_planning`
   does within a compiled region. Fixed addresses also make CUDA-graph capture
   trivial.

## 9. Technical challenges found

1. **Replacing the whole allocator breaks PyTorch features.** `CUDAPluggableAllocator`
   does not support `MemPool` (symmetric memory: Dist-MoE needed
   `TORCH_SYMMMEM_IMPLICIT_POOL=0`), allocator snapshots / NCCL registration hooks
   (the default `nccl2` process group aborts; needed `TORCH_DIST_USE_NCCL2=0`), device
   stats (torchtitan's memory monitor), or CUDA-graph private pools. A production
   planner must live inside the caching allocator or in codegen, not beside it.
2. **Persistent memory must be planned too.** Leaving parameters and optimizer state
   on an unplanned allocator costs 1.3-2.9 GiB, as much as the whole gain.
3. **Cross-step lifetimes.** Gradients outlive the step; without wrap-around
   planning the arena cannot reuse their memory and the plan is worse than the
   caching allocator (first prototype: +9.8 GiB).
4. **Determinism of the allocation sequence.** Ordinal matching assumes an identical
   sequence every step; first steps (tracing, lazy optimizer init) and data-dependent
   shapes differ. Graph-keyed planning removes this, but needs every allocation
   (including custom-op internals) visible in the graph.
5. **Streams.** Reuse is planned in host order; safe here because collectives hold
   their buffers until `wait_tensor`, but a general planner must model side-stream
   lifetimes (STAlloc ignores streams entirely).
6. **The paper's layering is not robust.** With wrap-around blocks it lost to
   greedy-by-size (7.4% vs 0.4% over the lower bound).
7. **Fake-PG caveats.** Fake all-gathers leave non-local shards unset, so routing was
   forced round-robin (allocation sizes are capacity-bound, so memory is unaffected);
   pure fake PG cannot run PP (needed `real_pp_fake_spmd`); numerics need real runs.

## Reproduce

Scratch package `mem_planning_study/` (untracked): `recipes.py` (configs),
`run.py` (entry point + measurement), `memplan_alloc.cpp` (allocator),
`planner.py`, `plot_alloc.py`, `summarize.py`, `compare_numerics.py`, launchers
`launch_fake.sh`, `launch_pp.sh`, `launch_real.sh`. Raw outputs:
`outputs/memplan/<run>/` (`memstat.jsonl`, traces, plans).

Build the allocator first:

```bash
cd mem_planning_study && g++ -O2 -std=c++17 -shared -fPIC memplan_alloc.cpp \
  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -o memplan_alloc.so
```

Configs: `dsv3_671b_fp8_distmoe_eager_alloc` (SPMD), `dsv3_671b_fp8_distmoe_pp2_vpp4`
(PP, via `launch_pp.sh`), `qwen3_30b_a3b_fp8_distmoe_fsdp8_ep8` (pass `NGPU=8`),
`dsv3_debug_fp8_distmoe_fsdp2_ep2_real` (real 2-GPU numerics). Memory policy via
`MEMPLAN_MEMORY_POLICY=none|default|full`. Example (Qwen3 30B, SAC):

```bash
C=qwen3_30b_a3b_fp8_distmoe_fsdp8_ep8; E="NGPU=8 MEMPLAN_MEMORY_POLICY=default"
mem_planning_study/launch_fake.sh qwen_sac_caching $C 0 $E MEMPLAN_TRACE_STEP=3
mem_planning_study/launch_fake.sh qwen_sac_trace $C 1 $E MEMPLAN_ALLOCATOR=trace \
  MEMPLAN_TRACE_STEP=3 MEMPLAN_DUMP_GRAPH=0
T=outputs/memplan/qwen_sac_trace
python mem_planning_study/planner.py $T/window_trace.txt --planner stalloc \
  --plan-out $T/plan_stalloc.txt --summary-out $T/summary_stalloc.json
mem_planning_study/launch_fake.sh qwen_sac_planned $C 0 $E MEMPLAN_ALLOCATOR=plan \
  MEMPLAN_PLAN=$PWD/$T/plan_stalloc.txt MEMPLAN_PLAN_START_STEP=3 MEMPLAN_STEPS=7 \
  MEMPLAN_DUMP_GRAPH=0
mem_planning_study/make_plots.sh qwen3_30b_sac qwen_sac_caching qwen_sac_trace \
  qwen_sac_planned stalloc
```
