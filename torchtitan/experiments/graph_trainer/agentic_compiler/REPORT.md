# Agentic compiler for training: agent-found fusions, agent-written kernels, bitwise-identical speedups

Status: draft for a post at the end of the week of 2026-11-09. Every number comes from the report
directories in this folder and was measured on one GB300 with the fake process group, unless stated
otherwise. `[TODO]` marks gaps to fill before posting.

## TL;DR

- On TorchTitan MoE training recipes, an agent loop finds fusion regions in GraphTrainer's joint graph,
  writes CuTe DSL kernels for them, and swaps them in through a kernel cache. Loss and grad_norm stay
  **bitwise identical** to the baseline.
- **The headroom is in the glue.** GEMMs mostly run near their compute roofline, while the memory-bound
  glue between them (norms, RoPE, casts, quantization, FSDP packing) takes **46-57%** of GPU time in the
  Dist-MoE recipes, which run their GEMMs in MXFP8. The loop's kernels removed 24-63% of that glue time.
- GPU time per training step: **-13.8%** on DSv3 671B, **-29.8%** on Qwen3 30B-A3B and **-15.7%** on GPT-OSS
  120B. Tokens/s: +13.5%, +41.8% and +24.0%. These are one-GPU, fake-process-group numbers, so they are
  upper bounds. On 2 real GB300 nodes, Qwen3 30B-A3B's kernels saved the same 2.24 s per step as on one GPU
  in the same benchmark: **-27.6%** step time instead of -32.6%. Over 300 steps there, with deterministic
  attention, GraphTrainer with the kernels stayed bitwise identical to eager.
- **Integration: a graph pass instead of a model rewrite.** The patterns are glue between GEMMs, attention
  and collectives, and most of the saving crosses module boundaries (59-87%). Eager reached **93-101%** of
  the graph's saving, bitwise, by calling the same kernels from autograd.Functions, but only by rewriting the
  transformer blocks. The graph ships them as a kernel pack, a pip-installable graph pass outside torchtitan
  ([graph-trainer-kernel-packs](https://github.com/SherlockNoMad/graph-trainer-kernel-packs)) that swaps
  only the subgraphs its matchers find and leaves the model code alone.
- **Eager's own compiler regions already collect part of it.** The recipes' `local_compile` (Inductor)
  regions save **76%** of the graph's saving on Qwen3 30B-A3B, mostly RoPE and the loss, and **12%** on
  DSv3 671B, whose hand-written fused kernels leave them little. Unlike the loop's kernels, these regions
  are not bitwise.
- **Warm start:** seeding the loop with a sibling model's verified kernels cut the wall clock from ~13 h to
  **2 h 41 min** (Qwen3 30B-A3B -> 235B-A22B). Every kernel was accepted on its first trial, but no kernel
  file ran unchanged: they all hard-code their shapes. A controlled study with a cold arm (DSv3 671B,
  microbatch doubled, 3 runs per arm) reached 90% of the best gain **2.7x sooner** warm, with a third fewer
  GPU-hours, and the warm kernels ended as fast or faster.
- **The hard parts are time and trust.** Verification is about 75% of a warm-start loop. Bitwise gates can
  be passed by kernels that silently fall back to the reference, and per-kernel speedups can be inflated
  by harness settings and slow baselines.

## 1. Background

- **Agents optimizing models end to end:** Together Compiler [8] is a continual optimization loop in which
  agents generate optimizations for an inference engine and hot-reload candidate forward passes into the
  running engine. AnyFast `[TODO: link and one line]`, ...
- **Agents writing GPU kernels:** KernelAgent [1], CUDA Agent [2], Cursor's multi-agent system [3], CAKE [4],
  TIRx Harness [5], KDAgent [6], EAT [7]. This loop's solve step (sec. 2, step 4) belongs to that family.
  Its setting is a training step: the problems are regions found in a real training graph, each kernel
  must match the exact aten subgraph it replaces byte for byte, and the kernels are swapped into a full
  training run.
- **Kernel verification:** ptxv `[link]`, Ed's `[TODO: name, link]`, ... Verification is what makes
  agent-written kernels usable in training.
- **Model + shape specialization is the industry trend:** FlashInfer [9] adapts its attention kernels to
  each model configuration through a customizable template and JIT compilation; ... `[TODO]`.
- **RSI is picking up.** Hypothesis: an RSI cluster spends about 50% of its compute on Tier 3 and smaller
  models `[TODO: source or reasoning]`. A fixed set of models and shapes running for a long time is what
  makes per-model, per-shape kernels pay off.

**The question:** can agents produce such specializations for a training step end to end, from finding
what to fuse to verified kernels, with exactly unchanged numerics? And how much of it depends on having a
graph?

## 2. Method

```
recipe --> 1 capture --> 2 plan --> 3 run <-------------------+ --> 5 package
           profile +     agent:     lift regions,             |     kernel pack:
           graph dump    fusion     substitute kernels        |     one graph pass per
                         regions    | misses + real inputs    |     model x layout x GPU
                                    v                         |
                                  4 solve --------------------+
                                  agent swarm: writer -> fresh verifier -> promote on PASS
```

1. **Capture.** A profiled training step of GraphTrainer, with every FX node of the joint graph
   (forward, recompute and backward) labelled in the trace.
2. **Plan** (fusion_finder skill). Ranks candidate regions by their gap to a measured roofline.
   For each region it generates a topology matcher and a graph pass that outlines every site of the region
   into a `call_module`. Matchers wildcard sizes, scalars and device indices, so they keep matching on
   other shapes and ranks.
3. **Run** (`substitute_kernels`, #5838). Each lifted call is keyed by the code of its subgraph (`l1`) and
   its input specialization (`l2`). A hit calls the cached kernel; a miss runs the original aten ops and
   writes a problem (`reference.py` plus the exact input/output contract). Lifting alone never changes
   numerics.
4. **Solve** (autokernel skill, #5839). A coordinator agent runs a swarm. Writers write CuTe DSL kernels,
   and every candidate goes to a freshly spawned verifier that never wrote it. The gate checks every output
   byte against `reference.py`, on random seeds and on real captured inputs, with output memory poisoned,
   under compute-sanitizer, with CUDA-graph replay and adversarial inputs. Only a PASS is promoted into
   the cache.
5. **Package.** The regions, the verified kernels and the expected call counts become a pip-installable
   kernel pack ([graph-trainer-kernel-packs](https://github.com/SherlockNoMad/graph-trainer-kernel-packs)).
   The pack checks the environment, the kernel hashes and every call count, and raises on any mismatch
   (sec. 5d).

**Rules for every loop:**
- Bitwise: a run passes only if loss, grad_norm and the MoE load-balance loss are bitwise equal to the
  baseline at every logged step (TensorBoard at full precision).
- Deterministic mode, seed 42. The recipe's parallel layout is emulated with the fake process group on one
  GB300, as SPMD rank 0 (except the real-cluster run in sec. 4).
- GPU time per step excludes the kernels that only emulate collectives.

## 3. Where the time goes: GEMMs near their roofline, glue far from it

A kernel's time has a floor set by what it runs out of first: compute (FLOPs / peak FLOP/s) or memory
(bytes / HBM bandwidth). On a GB300 (`hw_profile.json`), GEMMs peak at 1,930 TFLOP/s in BF16 and 3,615
TFLOP/s in MXFP8, and copies run at 6.9 TB/s.
- **GEMMs and attention** do hundreds of FLOPs per byte, so they are compute-bound, and library kernels
  (cuBLAS / nvjet, FA4) implement them.
- **Glue** is everything between them: norms, RoPE, casts, MXFP8 quantization and scale swizzles, router
  math, the loss, copies and FSDP bucket packing. It does a few FLOPs per byte, so its floor is bytes /
  bandwidth. Run as separate kernels, each piece reads and writes whole activations again. A fused kernel
  reads its inputs once and writes its outputs once; that difference is what the loop goes after.

**Measured split.** One profiled step per loop setup, without the loop's kernels, on one GB300 with the
fake process group; the copies that emulate collectives are excluded. Source: `glue_roofline/`.

| GPU time per step | DSv3 671B | Qwen3 30B-A3B | GPT-OSS 120B (per microbatch) | Qwen3 235B-A22B |
|---|---:|---:|---:|---:|
| Step | 583.9 ms | 967.6 ms | 2002.7 ms | 1031.0 ms |
| GEMM | 160.2 ms (27%) | 318.1 ms (33%) | 249.0 ms (12%) | 389.2 ms (38%) |
| Attention | 88.8 ms (15%) | 155.4 ms (16%) | 72.6 ms (4%) | 169.8 ms (16%) |
| **Glue** | **334.9 ms (57%)** | **494.1 ms (51%)** | **1681.1 ms (84%)** | **472.0 ms (46%)** |
| Glue without the optimizer step and deterministic index ops | 284.9 ms (49%) | 428.4 ms (44%) | 564.3 ms (28%) | 457.0 ms (44%) |

- **Glue is about half of the step** in the three Dist-MoE recipes, whose GEMMs run in MXFP8. Faster
  GEMMs leave a larger share to glue, and MXFP8 adds glue of its own: quantizing activations, gradients and
  weights, and swizzling their scales. In GPT-OSS's deterministic recipe, the sorts behind `index_add` /
  `index_put` alone take 1,093 ms (sec. 4).
- **GEMMs mostly run near their roofline.** From the traces of a separate study of the same kind of step
  (`finder_substrate_study/`):
  - Llama3 8B's GEMMs averaged 1,961 TFLOP/s, at the BF16 peak.
  - DSv3's MXFP8 GEMMs ran near peak (wo at 0.97x of its ideal), except small ones such as wkv_a
    (N = 576) at 2x its ideal.
  - Qwen3's Dist-MoE grouped GEMMs with fused dispatch or combine ran at 20-45% of the MXFP8 peak, and their
    weight-gradient GEMMs at 85-95%.
  - Attention efficiency wasn't measured. In Qwen3, FA4's deterministic backward took 4.3x its forward.

  These are library kernels, and the loop leaves every GEMM and attention kernel unchanged.

**The bound, and how much the loop captured.** If every glue kernel took no time, the step would shrink by
the glue share. That bounds what fusing glue can save; a fused kernel still has to move its inputs and
outputs once.

| | DSv3 671B | Qwen3 30B-A3B | GPT-OSS 120B | Qwen3 235B-A22B |
|---|---:|---:|---:|---:|
| Bound: step reduction if glue took no time | 57.3% | 51.1% | 83.9% | 45.8% |
| Achieved by the loop's kernels | 14.0% | 28.7% | 16.3% | 28.8% |
| Share of the glue time removed | 24% | 56% | 19% | 63% |
| The same, without the optimizer step | 28% | 64% | 20% | 64% |

Without its deterministic index ops and the optimizer step, GPT-OSS's glue is 28% of the step, and the loop
removed 56% of it. These runs repeat the loops' measurements on 2026-11-09 and land within about one point
of the reductions in sec. 4.

**Unfused glue runs far from its floor.** The loop's fused kernels do the work of about 118, 333, 403 and
347 ms of glue in 36, 55, 71 and 47 ms (DSv3, Qwen3 30B, GPT-OSS, Qwen3 235B): 3.3x, 6.1x, 5.7x and 7.3x
less time.

**Caveats:**
- Times are summed kernel durations; overlaps from programmatic dependent launch make the sum slightly
  larger than busy time. CUDA graphs were off.
- The fake process group's data is low-entropy, which lets GEMMs run at or above the random-data peak. On
  real data the GEMMs likely take longer, so the glue share is likely smaller.

## 4. Result 1: speedups on TorchTitan models

All rows ran on one GB300 with the fake process group: collectives are emulated by local copies, excluded
from GPU time, and nothing overlaps or waits on another rank. The gains are therefore upper bounds for a
real cluster; the last part of this section measures one model on real hardware.

| Model, layout | Kernels (regions) | GPU time per step | Tokens/s | Numerics |
|---|---|---|---|---|
| DSv3 671B DistMoE MXFP8, FSDP 256 x EP 64 | 20 (13) | 583.9 -> 503.4 ms (**-13.8%**) | 6,141 -> 6,968 (**+13.5%**) | bitwise |
| Qwen3 30B-A3B DistMoE MXFP8, FSDP 8 x EP 8 | 15 (15) | 967.9 -> 679.6 ms (**-29.8%**) | 34,702 -> 49,214 (**+41.8%**) | bitwise |
| GPT-OSS 120B MXFP8, FSDP 64 x EP 16 | 16 (14) | 1992.3 -> 1679.2 ms per microbatch (**-15.7%**) | 7,638 -> 9,468 (**+24.0%**) | bitwise |
| Qwen3 235B-A22B DistMoE MXFP8, FSDP 256 x EP 32 (warm start, sec. 6a) | 19 | 1036.4 -> 737.7 ms (**-28.8%**) | 7,698 -> 10,631 (**+38.1%**) | bitwise |

Sources: `dsv3_671b_distmoe_mxfp8/`, `qwen3_30b_a3b_distmoe_mxfp8/`, `gpt_oss_120b_mxfp8_64gpu/`,
`qwen3_235b_a22b_distmoe_mxfp8/`, and the packs' READMEs. The DSv3 and Qwen3 30B tokens/s come from the eager
studies' graph runs (step 20, CUDA graphs on).

**GPT-OSS needs a note on baselines.** In deterministic mode, torch's `index_add` / `index_put` sort the
indices and then accumulate in order. That took 1094 of the 2000 ms per microbatch. The loop's first pack
replaced the `index_add` of the expert-bias gradients with kernels more than 100x faster, and reported 3.26x
tokens/s over the deterministic recipe. That gain says more about the slow deterministic baseline than about fusion,
so the published pack no longer contains the `index_add` and `deterministic_scatter_add` kernels. The pack
above is what remains. The deterministic recipe with this pack is still about 0.6x of the non-deterministic
recipe (about 15,000 tokens/s). In a non-deterministic timing run without the `index_add` / `index_put`
kernels, the remaining kernels cut GPU time per microbatch from 976.5 to 658.8 ms (1.48x).

### What the kernels fuse

The regions are glue between the GEMMs and attention (sec. 3). A few examples from the packs' READMEs; time
per call is the verifier's measurement of the original aten ops against the kernel:

| Model, region | What one kernel does | Scope | Calls per step | Time per call, ops -> kernel | Saving per step |
|---|---|---|---:|---|---:|
| Qwen3 30B-A3B, G-02 | splits wqkv's output, makes q, k and v contiguous, applies the q/k RMSNorms and RoPE with its position gather | cross-module (`qkv_linear`, `q_norm`, `k_norm`, `rope`); forward | 48 | 2252 -> 157 us (14.3x) | 100.6 ms |
| Qwen3 30B-A3B, G-01 | RoPE backward and q/k RMSNorm backward (with their weight gradients), concatenation with v's gradient, MXFP8 quantization for wqkv's backward GEMMs | cross-module, into torchao's linear backward; **backward** | 48 | 2481 -> 185 us (13.4x) | 110.2 ms |
| DSv3 671B, G-001 | the router: group-limited top-8 selection and the aux loss | cross-module (router, aux loss, the experts' token counts); forward | 58 | 245 -> 6.2 us (39x) | 13.8 ms |
| GPT-OSS 120B, G-007 | backward of the experts' clamped SwiGLU | one module; **backward** | 36 | 3301 -> 534 us (6.2x) | 99.6 ms |
| Qwen3 30B-A3B, G-07 | cross-entropy backward of one loss chunk, fp32 logits -> bf16 gradient | one function (the loss); **backward** | 8 | 2556 -> 519 us (4.9x) | 16.3 ms |
| Qwen3 235B-A22B, G-105 | packs the expert weight gradients into FSDP's reduce-scatter bucket | crosses into FSDP's gradient packing; **backward** | 94 | 384 -> 47 us (8.3x) | 31.7 ms |
| DSv3 671B, G-047 | attention_norm's RMSNorm plus the MXFP8 input quantization of the two linears it feeds | cross-module, into torchao's linear forward; forward | 61 | 52 -> 18 us (2.9x) | 2.1 ms |
| DSv3 671B, G-049 | layer L's attention_norm backward and residual-gradient add, plus the gradient quantization for layer L-1's shared-expert down projection | **cross-layer**; **backward** | 61 | 93 -> 57 us (1.6x) | 2.2 ms (4.4 ms with its second variant) |

- **The biggest wins replace long chains of small kernels.** G-02 alone replaces 34 aten ops, about 20 of
  which launch kernels. Qwen3's two attention-glue kernels (G-01, G-02) save 211 of the loop's 282 ms. The
  router is many tiny kernels per layer, so one kernel runs 34-39x faster per call (forward and backward);
  in DSv3 the router's forward, backward and recompute save 30 of 76 ms.
- **Norm and quantization glue around the MXFP8 GEMMs gains less per call** (1.6-3x), because the original
  is only a few kernels. It still adds up, since it runs in every layer.
- **Most cross a boundary:** 6 of the 8 cross a module, library or layer boundary, and 5 run in the
  backward.
- **The same families recur across models** (sec. 6a): router and aux loss, RMSNorm with add and
  quantization, the loss, RoPE glue and MXFP8 weight preparation.

### Qwen3 30B-A3B on a real cluster

Two separate runs took the Qwen3 30B-A3B loop's 16 lift passes and its kernels (18 kernels, cache keys
remapped to this benchmark) to 2 GB300 nodes (8 GPUs, FSDP 8 x EP 8) with real communication. Both use the
same benchmark, which differs from the table above: 8 microbatches of 32K tokens per step (2,097,152 tokens).
The first measured throughput over 50 steps; the second trained 300 steps to compare loss with eager.

**Throughput, 50 steps.** Median step time over steps 11-50. The same benchmark also ran on one GB300 with
the fake process group. Source:
[Qwen3 30B-A3B: eager vs GraphTrainer + kernels](https://artifacts.internalmeta.com/a/1158072533227800).

| Step time, median | 1 GB300, fake process group | 2 GB300 nodes (8 GPUs), real communication |
|---|---:|---:|
| Eager recipe, with its `local_compile` regions | 5.405 s | 6.629 s |
| Plain GraphTrainer | 6.869 s | 8.123 s |
| GraphTrainer + lift passes + kernels | 4.630 s | 5.881 s |
| **Saving vs plain GraphTrainer** | **2.239 s (-32.6%)** | **2.242 s (-27.6%)** |
| Tokens/s per GPU, plain GraphTrainer -> with kernels | 37,931 -> 56,057 (+47.8%) | 32,039 -> 44,111 (+37.7%) |
| With kernels vs the eager recipe | -14.3% | -11.3% |

- **The kernels saved the same absolute time with real communication.** The glue they remove was not
  hidden behind communication on this layout. The relative gain falls from 32.6% to 27.6% only because
  communication adds time that no kernel touches.
- **Numerics held.** Every substituted call matched its reference byte for byte on real training inputs.
  Loss and grad_norm were bitwise equal to plain GraphTrainer over 5 steps, using FA4's deterministic
  attention for that comparison. Rank 0 had 1,474 cache hits and 0 misses.
- **Running it took five fixes:**
  - GraphTrainer's gradient-accumulation graph extraction dropped lifted regions;
  - FSDP bucketing, deferred until after the lifts, called kernels on FakeTensors;
  - kernel files inherited `from __future__ import annotations`;
  - quack and cutlass-dsl versions mismatched;
  - four lift passes contained bucketing glue that doesn't exist before deferred bucketing (R2 in `TODO.md`
    item 1).

**Loss parity with eager, 300 steps.** Seed 0 and the same data order for every run. Source:
[Qwen3 30B-A3B: eager vs GraphTrainer are bitwise identical for 300
steps](https://artifacts.internalmeta.com/a/1151430804545666). Eager is unchanged. GraphTrainer carries three
fixes from a parity investigation, which are not in this branch yet:
- **Gradient accumulation.** Eager reduce-scatters every microbatch and accumulates the shards in float32.
  GraphTrainer accumulated unsharded bf16 gradients and reduced once. Fix: reduce every microbatch (8
  reduce-scatters per step) into float32 accumulators.
- **Chunked-loss LM head.** Eager sums the chunks' unsharded weight gradients in place (`mm`, then
  `addmm(out=grad)`) and reduce-scatters once; GraphTrainer reduce-scattered each chunk. Fix: merge the
  multi-use reduce-scatters, with an in-place `addmm_`.
- **Logged loss.** `torch.sum` over a stack vs a sequential add. Fix: a sequential sum.

| | Loss at step 300, deterministic attention | Step time (s), deterministic attention | Step time (s), cuDNN attention | Tokens/s per GPU, cuDNN attention |
|---|---|---:|---:|---:|
| Eager | 2.15474 | 8.895 | 8.684 | 30,186 |
| GraphTrainer | 2.15474, bitwise equal to eager at all 300 steps | 8.656 | 8.465 | 30,968 |
| GraphTrainer + kernels | 2.15474, bitwise equal to eager at all 300 steps | 6.417 | 6.221 | 42,142 |
| Eager + `local_compile` | 2.00271, differs from step 1 | 7.189 | 6.980 | 37,555 |

- **Bitwise for 300 steps, against eager.** With FA4's deterministic attention backward, eager,
  GraphTrainer and GraphTrainer + kernels have the same loss and grad_norm at every step. All 1,472 region
  calls ran on the kernels. The investigation found no other numerical difference.
- **The saving held:** 2.239 s per step with deterministic attention (-25.9%) and 2.244 s with cuDNN
  attention (-26.5%), the same 2.24 s as the 50-step run. With cuDNN attention, GraphTrainer + kernels is
  10.9% faster per step than eager with `local_compile`. The three configurations that also ran in the
  50-step run are each 0.34-0.35 s slower here. That includes eager with `local_compile`, which the parity
  fixes don't touch, so the shift comes from the measurement (this run reports a mean and logs the loss
  every step), not from the fixes.
- **With production attention, loss curves cannot validate kernels.** Two runs of the identical eager
  configuration with cuDNN attention matched at step 1, diverged from step 2, and ended 0.22 apart at step
  300 (2.1964 vs 1.9762; largest per-step gap 0.454). The cuDNN and flash attention backward kernels
  accumulate dQ with atomic adds whose order varies from run to run, even under
  `torch.use_deterministic_algorithms`. All four configurations' curves fall inside that band (about
  0.2-0.25 at step 300), so a curve comparison only catches errors larger than the band. `local_compile`
  shows how much training amplifies a small difference: it is deterministic and differs from eager by
  compiled-region rounding (about 6e-6 relative at step 1), yet ends 0.15 lower at step 300.

## 5. Result 2: integrating the kernels, a graph pass vs an eager rewrite

The loop's kernels are specialized to one model, layout, GPU and torch build; each hard-codes its shapes
(sec. 6a). They still have to get into the training step, and there are two ways:
- **eager:** rewrite model code so it calls the kernels from autograd.Functions;
- **graph:** a pass that swaps the matched subgraphs for the kernels.

Rewriting model code puts agent-written code with a single consumer into shared model files. Every reviewer
has to trust its numerics, and every later model change has to keep it working. Hillclimb artifacts are
hard to land that way. This section measures how far eager gets, what it costs, and how the graph's kernel
packs avoid the rewrite.

### a. Where the patterns sit

Share of each loop's verified saving, by what eager would need to do (classification of every kernel in the
packs, `outputs/agentic_compiler/PATTERN_AUDIT.md`; GPT-OSS without the removed index kernels):

| Eager level | DSv3 671B | GPT-OSS 120B | Qwen3 30B-A3B |
|---|---:|---:|---:|
| op-level: a better implementation of one op fixes eager too | 13% | 14% | 1% |
| one autograd.Function within one module | 33% | 86% | 56% |
| restructure: crosses an autograd boundary the model doesn't own (torchao linear, custom op) | 28% | - | 43% |
| "compiler-only": nodes created by graph passes (bucketing, remat), or crossing layers | 25% | - | - |

By scope and phase (the two eager studies, step 1):

| | Qwen3 30B-A3B | DSv3 671B |
|---|---:|---:|
| Cross-module within a layer | 87% | 59% |
| Cross-layer | 0% | 3% |
| One module, function or linear's backward | 12% | 3% |
| FSDP parameter / gradient paths | 1% | 35% |
| Backward / recompute | 54% / - | 56% / 16% |

The cross-layer region is DSv3's G-049: layer L's attention_norm backward + residual-gradient add, fused
with the gradient quantization of layer L-1's FFN down projection.

### b. What eager already fuses by hand

| | Qwen3 30B-A3B | DSv3 671B |
|---|---|---|
| Eager's `local_compile` (Inductor) regions that save time | RoPE, loss, gate-gradient split | aux-loss statistics, loss, gate-gradient split; the RoPE, MLA and SwiGLU regions are bypassed by the recipe's fused overrides |
| Already fused in both eager runs (not counted) | FA4, Dist-MoE's grouped GEMMs with fused SwiGLU | FA4, Dist-MoE's grouped GEMMs, the fused MLA RoPE and SwiGLU overrides, quack RMSNorm |
| `local_compile` saving, vs the graph's | 216.8 ms (**76%** of the graph's 286.7 ms) | 9.6 ms (**12%** of 78.8 ms) |
| Numerics | not bitwise (the RoPE and loss regions) | not bitwise (the loss region) |
| Graph regions with no eager counterpart | router counts and softmax, norm + quantization (fwd and bwd), add + norm, weight prep | almost all: cross-module glue, recompute-specific regions, FSDP bucket glue |

**What differs:**
- **Eager fuses within a function.** Each `local_compile` region and fused override covers one function or
  module. In Qwen3, most of eager's 76% is two such functions (RoPE 187 ms, loss 28 ms).
- **The graph fuses across the boundaries eager fixes in code:** module boundaries (attention_norm + the
  input quantization of the next MXFP8 linear), autograd boundaries (a module's backward + the gradient
  quantization inside torchao's linear backward), the memory policy's recompute (a region exists only at its
  recompute site), and FSDP's bucket packing.
- **When eager already has a strong fused kernel** (DSv3's MLA and SwiGLU overrides), there is little left for
  Inductor regions, and the graph's regions move to the glue between those kernels.

### c. Via eager: the same kernels from autograd.Functions

Both studies call the loop's verified kernels unchanged, from autograd.Functions activated by a config
`@override` of the transformer block. Sources: `qwen3_30b_eager_study/`, `dsv3_671b_eager_study/`.

| | Qwen3 30B-A3B | DSv3 671B |
|---|---|---|
| Kernels used | 15 of 15 | 16 of 20 |
| GPU-time saving, eager vs graph | 290.2 vs 286.7 ms (**101%**) | 73.4 vs 78.8 ms (**93%**) |
| Tokens/s gain, eager vs graph | +14,232 vs +14,512 | +831 vs +827 |
| Numerics | bitwise vs stock eager | bitwise vs stock eager |
| Port effort, after the loop | ~18 min, 5 Functions | ~50 min, 6 Functions + FSDP2 copy-in hook + weight-prep patch |

**What made every region reachable:**
- A Function owns every forward op whose backward is in a region, and returns a shared tensor once per
  consumer, so the autograd engine's accumulation adds land inside the Function's backward.
- Saved tensors are exactly the kernel's contract. Where the graph recomputes, eager's forward runs the
  recompute kernel, because eager saves what the graph recomputes.
- The cross-layer region needs a side channel: layer L's backward stores the quantized gradient keyed by the
  gradient tensor it returns, and layer L-1 uses it when it receives that same tensor (60 of 61 hits).
- FSDP bucket glue goes through FSDP2's experimental `set_reduce_scatter_input_fn` and the Dist-MoE weight
  hook. The pattern audit had marked these and the cross-layer region "compiler-only"; in practice only the
  router's recompute-site kernel and one bucket-pack kernel whose layout differs from FSDP2's stayed out of
  reach.

**What eager costs:**
- The Functions are model rewrites, not passes. They cross module boundaries, hard-code one layout and
  raise otherwise. The graph's matchers fall back to aten ops when a pattern changes.
- The port depended on the loop for kernels with exact contracts and a bitwise reference.

### d. Via the graph: kernel packs

**What a pack is.** A pip-installable package that lives outside torchtitan:
[github.com/SherlockNoMad/graph-trainer-kernel-packs](https://github.com/SherlockNoMad/graph-trainer-kernel-packs).
Each pack is one registered graph pass for one model x layout x GPU, enabled by name:

```python
from graph_trainer_kernel_packs.dsv3_671b_distmoe_mxfp8_fsdp256_ep64_sm103 import (
    dsv3_671b_distmoe_mxfp8_fsdp256_ep64_sm103,  # importing registers the pass
)

config.compile.extra_passes = [dsv3_671b_distmoe_mxfp8_fsdp256_ep64_sm103.__name__]
```

A pack holds:
- `regions/`: the topology matchers, in lift order;
- `kernels/`: the verified kernels with their `reference.py`, keyed like the `substitute_kernels` cache;
- `manifest.json`: the target environment, and each kernel's sha256 and expected call count;
- `verification/` and the recipe it was verified with.

**How it runs**, on the joint graph after GraphTrainer's own passes and before Inductor:

```
environment == manifest target? ------------------------------- no  -> raise
for each region: a matcher finds a subgraph? ------------------ no  -> graph unchanged there
                 yes -> lift it into call_module fusion_G<id>          (numerics unchanged)
for each lifted call: a kernel stored under its (code, inputs) key?
                 no  -> the call runs its original aten ops
                 yes -> sha256 matches the manifest? --------------- no  -> raise
                        yes -> call the kernel instead
each kernel called exactly as often as in the verified run? ---- no  -> raise
```

**Why it lands:**
- **Nothing in torchtitan changes per model.** Model code, recipes and core stay as they are. torchtitan
  only needs the generic hooks: registered passes and `substitute_kernels` (#5838, not yet in `main`). One
  config line enables a pack; deleting it removes the pack.
- **The change is local by construction.** A pack is pattern matching. Where no matcher matches, the graph
  is untouched, and lifting alone never changes numerics. A lifted call without a kernel keeps running its
  aten ops. A pack can only affect the subgraphs it matched.
- **Each replacement has a checked contract.** A kernel replaces exactly one lifted subgraph: it takes that
  subgraph's inputs and returns its outputs. It was verified byte for byte against the subgraph's
  `reference.py`, and the whole run is checked bitwise on loss and grad_norm.
- **Mismatches fail loudly.** On top of the matching, the published packs check the exact environment
  (torch build, CUDA, GPU, cutlass-dsl), each kernel's sha256 and each kernel's call count. On another graph
  or environment the pass raises at compile time instead of applying in part, so a pack never runs silently
  slower or different.
- **Matchers survive changes around their regions.** They accept any size, scalar operand and device index,
  and neither cache key depends on the device index, so every rank should lift the same regions and find the
  same kernels. On graph dumps they kept matching across sequence length, microbatch size, determinism and
  model size within a family. In a training run, 13 of 16 Qwen3 30B-A3B matchers lifted unchanged on 235B.
  Known breaks are layout-dependent parameter-producer chains, bucket packing, list arity and decomposition
  variants (`TODO.md` item 1).

**Limits:**
- A pack hits only with the pack's deterministic and Inductor settings: for static shapes, the input key is
  the AOTAutograd cache key, which includes them (sec. 7).
- Kernels are pinned to one torch build and GPU (sec. 6b). A torch upgrade means re-verifying or regenerating
  the pack.
- Until `substitute_kernels` lands, packs need the `agentic_compiler` branch.

### e. One pattern, two formats

Qwen3 30B-A3B's G-02 is the forward glue between the wqkv GEMM and
attention: the split of wqkv's output, three contiguous copies, the q/k RMSNorms, and RoPE with its position
gather. One kernel replaces it and saves 100.6 ms per step.

*In eager*, the kernel cannot replace any one module:
- its ops belong to four modules: `qkv_linear`, `q_norm`, `k_norm` and `rope`;
- its backward partner G-01 reaches into the wqkv gradient quantization inside torchao's MXFP8 linear
  backward;
- the forward has to save exactly what G-01 reads.

So the port wraps attention_norm -> wqkv -> split -> q/k norm -> RoPE in one autograd.Function that calls two
of the loop's kernels and a GEMM in each direction. It then rewrites the whole transformer block's `forward`
to call it. Trimmed from
`qwen3_30b_eager_study/graph_kernels.py` and `override_qwen3.py`:

```python
class AttentionInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_TD, attn_norm_w, weight, w_fprop, w_sfprop, ..., q_norm_w, k_norm_w, cache, positions_T, meta):
        normed_TD, rstd, x_q, x_s = kernel("kbxm")(x_TD, attn_norm_w)      # G-41: attention_norm + quant
        qkv = scaled_mm(x_q, w_fprop, x_s, w_sfprop)                       # the unchanged GEMM
        xq, xk, xv, rq, rk, cos, sin, q, k = kernel("4dzo")(               # G-02
            qkv, q_norm_w, k_norm_w, cache, positions_T)
        ctx.save_for_backward(x_TD, rstd, normed_TD, ..., xq, xk, rq, rk, cos, sin, q_norm_w, k_norm_w)
        return q, k, xv, x_TD.view_as(x_TD)

    @staticmethod
    def backward(ctx, gq, gk, gv, gx_res):
        ... = kernel("qnfe")(...)    # G-01: RoPE + q/k norm backward, cat, quant
        g_w = wgrad(...)             # the unchanged WGRAD GEMM
        ... = kernel("tl5t")(...)    # G-15: DGRAD GEMM + attention_norm backward + residual add
        return g_x, g_attn_norm_w, g_w, None, None, None, None, g_qw, g_kw, None, None, None


class GraphKernelQwen3Block(Qwen3TransformerBlock):   # the whole block, rewritten
    def forward(self, x, attention_metadata, positions=None, *, padding_mask=None, aux_loss_denominator=None):
        attn, moe = self.attention, self.feed_forward
        xq, xk, xv, x_res = AttentionInput.apply(x, self.attention_norm.weight, ..., positions, ...)
        out_TD = attn.inner_attention(xq, xk, xv, ...).contiguous().view(...)
        h, x_router, x_experts = AttentionOutput.apply(out_TD, x_res, ...)   # wo -> add -> ffn_norm
        scores_TK, ids_TK, routing_map_TE = moe.router(x_router, ...)
        return torch.add(h, moe.routed_experts(x_experts, scores_TK, ids_TK, routing_map_TE, ...))


@override(target=Qwen3TransformerBlock.Config, description="Qwen3 block calling the graph loop's kernels")
def graph_kernel_block(cfg, *, attention_input=True, attention_output=True, router=False):
    return derive(cfg, GraphKernelQwen3Block.Config, ...)
```

A recipe enables it with `config.override.imports = [("override_qwen3.graph_kernel_block", {...})]`.

*In the kernel pack*, the same kernel is data next to an unchanged model:

```
regions/g02_k_norm.py              the generated matcher for G-02 (below)
kernels/pat_4dzo.../reference.py   the 34 aten ops the kernel replaces: its input/output contract
kernels/pat_4dzo.../<l2>.py        the verified CuTe DSL kernel, run(...)
manifest.json                      "pat_4dzo.../<l2>.py": {"sha256": "bbc34dfe...", "calls": 48}
```

```python
REGION_ID = "G-02"

def match_v0(n0):
    """One instance, anchored at its aten._to_copy.default node: (members, external inputs) or None."""
    if not is_op(n0, 'aten._to_copy.default', [N], {'dtype': 'torch.bfloat16', ..., 'device': D('cuda')}):
        return None
    n1 = n0.args[0]
    if not is_op(n1, 'aten.add.Tensor', [N, N]):
        return None
    ...    # 41 more nodes; S matches any size, D any device index
    return members, external_inputs

MATCHERS = [('aten._to_copy.default', match_v0)]
```

| | Eager override | Kernel pack |
|---|---|---|
| Unit of change | the transformer block's `forward`, rewritten by hand | one generated region file per pattern, plus its kernel |
| What one pattern pulls in | a Function spanning every module that the pattern and its backward partner touch | one matcher; forward and backward patterns are separate regions |
| Hand-written code (Qwen3) | a 146-line block override, a 41-line loss override, and 540 lines of Functions, MXFP8 operand helpers and a patch of the expert weights' FSDP hooks | the pack's `__init__.py`, `recipe.py` and README; regions and manifest are generated |
| Granularity | three flags, each switching a stretch of several patterns | one region per pattern |
| The contract | implicit in what each Function saves and returns | `reference.py`, checked byte for byte |
| Outside the measured layout | raises (no TP/CP, no shared experts, no remat) | a site that doesn't match keeps its aten ops; the published packs raise on a call-count mismatch |
| Reading it | ordinary model code | the matcher is generated and not meant to be read; the region's docstring and `reference.py` say what it covers |

In short: eager asks where in the model code a pattern lives and rewrites that code. The pack asks where in
the graph the pattern occurs and swaps only those nodes.

**Answer:** for these models, almost all of the graph's kernels can be integrated in eager too, but only by
rewriting the transformer blocks around them. The graph integrates them as a pass that leaves the model code
alone, with each kernel's contract written down as `reference.py`.

## 6. Hard problems

### a. Optimization time and reuse

**Why retuning is the common case.** Today's kernels are specialized to their shapes: each hard-codes its
sizes, grid splits and reduction order. Any change of model variant (hidden size, number of experts) or of
sharding (FSDP or EP degree, which sets each rank's local shapes and bucket layout) needs the kernels
tuned again. Even the microbatch size changes most input shapes: doubling it changed 25 of DSv3's 29 kernel
specs (the controlled study below). Warm start makes that retuning cheaper by seeding it with the kernels
already verified for a nearby configuration.

#### Warm start: Qwen3 30B-A3B -> 235B-A22B

**Setup.** The 235B loop started from everything the 30B loop produced: its regions, its plan, and each
kernel with its design notes and verifier checklist in the writer's and verifier's prompts. Source:
`qwen3_235b_a22b_distmoe_mxfp8/reuse_report.md`. There was no cold 235B run, so the ~5x shorter wall clock
also includes a more mature process; the controlled study below isolates the warm-start effect.

| | Qwen3 30B-A3B, cold | Qwen3 235B-A22B, warm |
|---|---|---|
| Wall clock to a validated result | ~13 h | **2 h 41 min** |
| Plans / training iterations | 4 / 7 | 2 / 3 |
| Kernels accepted | 15 | 19 (15 ported, 4 new) |
| Trials per accepted kernel | typically 2-14, up to 30 | **1 for every kernel** |
| Verifier FAILs | at least 2 | **0** |
| GPU time saved per step | -29.8% | -28.8% |

**How much was reused:**

| Artifact | Reused? |
|---|---|
| Region matchers | 13 of 16 lifted the expected sites unchanged; 3 broke on a decomposition change or a different parameter-producer chain |
| Patterns (`l1` keys) | 6 of 13 identical to 30B; only the input specialization differed |
| Kernel files as is | **0 of 13**: every kernel hard-codes its shapes |
| Kernel source | 15 of 19 kernels ported, at 0.76-1.00 source similarity (near-identical patterns rewritten from scratch in earlier loops: 0.30-0.39) |
| Kernel speed | kept: e.g. attention forward glue 14.3x -> 14.2x, router backward 31.7x -> 36x; launch-bound kernels lose some with 4x fewer tokens per call |

**Where reuse stops:**
- The finder still had to run on the new model: one layout-specific region (expert-gradient bucket pack)
  saved 31.7 ms/step, as much as the six ported router and loss kernels together.
- The real porting cost was reduction orders that depend on shape. A bitwise kernel must add in the same
  order as the backend, and quack's RMSNorm and torch's reductions choose their order from the sizes.
- Verification did not get cheaper (below).

Across all three packs, the same families recur: router and aux loss (all three models), RMSNorm with add and
MXFP8 quantization (fused forward and backward), cross-entropy loss (GPT-OSS and Qwen3 share an identical
`reference.py`), RoPE glue, and MXFP8 weight preparation.

#### Controlled: DSv3 671B at twice the microbatch

An independent study by Yidi measured the warm-start effect alone, with a cold arm and repeats. Source:
[Whole-config warm start: re-tuning every kernel for mbs8k](https://artifacts.internalmeta.com/a/2198296197707549).

**Setup:**
- **Configs:** DSv3 671B, FSDP 256 x EP 64. The microbatch goes from 4,096 to 8,192 tokens per rank, which
  changes the input shapes of 25 of the 29 kernel specs (21 patterns).
- **Workers:** one kernel-tuning worker per pattern, each on one GB300 with a 120-minute active budget.
- **Arms:**
  - **cold:** every pattern from scratch;
  - **warm:** the 3 specs whose shapes didn't change reuse the old kernel as is; every other pattern starts
    from the old config's kernels for it, or for a near-duplicate pattern.
- **Repeats:** 3 runs per arm.
- **Scoring:** every promoted kernel is re-gated on held-out seeds and re-timed against `reference.py`, with
  both arms of a pattern on the same GPU.
- **Per-step kernel time:** the sum over specs of calls per step x kernel time, using each run's best exact
  kernel at that point. With no kernels it is 291.7 ms; with the best kernel of any run for every spec,
  169.4 ms (1.72x).

| | Warm, 3 runs | Cold, 3 runs |
|---|---|---|
| Patterns tuned | 18 (+3 specs reused) | 21 |
| Median time to a first verified kernel, per spec | **12.3 min** | 32.3 min |
| Time to 90% of the best config-level gain | **16 / 25 / 28 min** | 59 / 67 / 95 min |
| Per-step kernel time at 30 min | 178.1-180.6 ms | 220.8-255.9 ms |
| Per-step kernel time at 120 min | 173.0-175.4 ms | 178.9-181.0 ms |
| GPU-hours | 15.0-17.0 | 22.5-25.8 |
| Output tokens | 1.03-1.15M | 1.58-1.66M |

- **Warm reached 90% of the gain about 2.7x sooner** (median 25 vs 67 min), with about a third fewer
  GPU-hours and tokens.
- **Warm kernels ended as fast or faster.**
  - Warm set the best kernel on 17 of 29 specs. On the other 12, cold's best is within 0-3.8% of warm's,
    which is run-to-run noise.
  - Warm's largest leads were 34%, 19%, 13% and 11%: starting from the old kernel found faster kernels
    than 120 minutes from scratch.
- **Where warm didn't help:** GEMM-bound patterns with little headroom (best 1.04-1.25x) sit at noise in
  both arms, and the two patterns with no old kernel ran cold in both.
- **Isolation needed more than the prompt.** Three runs leaked: agents searched the machine for earlier
  work, read another pattern's tuned cache, and guessed a sibling run's path. Those runs restarted from
  scratch, and the folders holding answers were made unreadable.

This complements the Qwen3 235B warm start, which had no cold arm. The two are different transfers: a shape
change within one model here, a larger model of the same family there.

#### What stays hard

- **Verification is the critical path.** In the 235B warm start, writers handed off 18 of 19 candidates
  within 26 min; verification took about 2 h. Each verifier ran the full suite (40-100 min), even for ports
  that changed only constants (4-25 lines). About 40 min were lost to coordination: a GPU lock held through
  JIT compiles, and a cap on concurrent agents.
- **Kernels are not reusable as files.** They encode sizes, grid splits and reduction orders. What would
  help:
  - shape-generic kernels with explicit raises;
  - a port manifest recording the backend configuration a kernel's reduction order copies;
  - delta verification for constants-only ports;
  - retrieval of verified kernels by pattern.
- **Matchers are mostly robust**, but break on decomposition changes and on parameter producers that depend
  on the parallel layout (`TODO.md` item 1).

### b. Kernel verification

What the gate caught, or what got past a weaker gate, across the four loops:

| Failure | How it was found |
|---|---|
| Kernels silently falling back to the reference ops for other shapes passed the bitwise gate with no speedup | timing check during the 235B port; kernels must now raise on unsupported inputs |
| A kernel with its zero-padding stores removed passed, because fresh memory reads as zero | a writer; the harness now poisons output memory |
| PDL ordering: a kernel triggered its dependent before waiting on its predecessor (aux loss wrong 20/20) | verifier, with an early-triggering predecessor |
| Non-coherent loads of data written earlier in the same PDL chain | verifiers |
| -0 where the reference gives +0 (a missing +0 seed in a sum) | verifier |
| Grid-barrier word left dirty under CUDA-graph capture (207 s hang); a two-thread race in the fix | verifiers (3 candidates for one region) |
| Timing claims that held only with L2-resident inputs | verifiers re-timing with rotating inputs |
| References timed with `fill_uninitialized_memory=True`, inflating per-kernel speedups (346 vs 199 us) | harness audit |
| cuDNN varlen attention backward is nondeterministic, so the baseline itself was not reproducible | first lifted Qwen3 run; moved to FA4 |
| A captured reference whose output is mostly uninitialized (99.6% of a `torch.empty` bucket buffer is never written) fails a bytewise check against itself; runs split on whether to gate only the written slice | the controlled warm-start study; the capture has to be fixed |

**Still open:**
- **Passing on one GPU is not passing everywhere.** Four GPT-OSS kernels use whole-GPU cooperative launches,
  which can serialize or deadlock next to real NCCL kernels. One Qwen3 kernel hard-codes 152 SMs. A gate on
  one fake-PG GPU does not see either.
- **Bitwise means "bitwise for this torch build and GPU".** Kernels copy the backend's reduction order (quack
  configurations, torch's reduction splits). An environment variable such as `PYTORCH_SUM_INNER_TREE` breaks
  parity, so packs pin the torch build, CUDA, GPU and cutlass-dsl versions.
- **Correct is not the same as fair.** The GPT-OSS `index_add` kernel was verified and exact, but its speedup
  measured a slow deterministic baseline (sec. 4). Report the non-deterministic baseline and attribute the
  gain per kernel.

## 7. Caveats and next steps

- **Fake process group, one GPU, rank 0, for every loop.** No communication overlap and no other rank's
  graph. The real-cluster runs so far (Qwen3 30B-A3B, 2 GB300 nodes, sec. 4) kept the full absolute saving
  and, over 300 steps, bitwise parity with eager. Larger layouts are untested: next, GPT-OSS on 64 GPUs, then DSv3 on 256 (`TODO.md` item 4).
- **Deterministic mode only:** packs hit only in deterministic mode today, because one cache key includes the
  deterministic flag.
- **Workflow:** the loop is three tools plus per-model glue (`TODO.md` item 3, `WORKFLOW.md`).

## References

1. KernelAgent: hardware-guided GPU kernel optimization via multi-agent orchestration.
   https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/
2. CUDA Agent. https://arxiv.org/abs/2602.24286
3. Cursor's multi-agent system for kernels. https://prod.cursor.com/blog/multi-agent-kernels
4. CAKE. https://arxiv.org/abs/2608.12629
5. TIRx Harness: an open compiler harness for agentic GPU programming.
   https://blog.mlc.ai/2026/09/29/tirx-harness-an-open-compiler-harness-for-agentic-gpu-programming
6. KDAgent. https://nvlabs.github.io/kda/
7. EAT. https://arxiv.org/abs/2610.36848
8. Together Compiler: continual compilation for AI inference.
   https://www.together.ai/blog/together-compiler-continual-compilation-for-ai-inference
9. FlashInfer: efficient and customizable attention engine for LLM inference serving.
   https://arxiv.org/abs/2501.01005, https://github.com/flashinfer-ai/flashinfer

## Appendix: artifacts

| What | Where |
|---|---|
| Loop reports (plans, regions, kernel caches, audit reports) | `dsv3_671b_distmoe_mxfp8/`, `gpt_oss_120b_mxfp8_64gpu/`, `qwen3_30b_a3b_distmoe_mxfp8/`, `qwen3_235b_a22b_distmoe_mxfp8/` |
| GEMM vs glue split and glue roofline (sec. 3) | `glue_roofline/` |
| Warm-start report | `qwen3_235b_a22b_distmoe_mxfp8/reuse_report.md` |
| Controlled warm-start study (DSv3 671B, sec. 6a) | [artifact 2198296197707549](https://artifacts.internalmeta.com/a/2198296197707549) |
| Real-cluster throughput run (Qwen3 30B-A3B, sec. 4) | [artifact 1158072533227800](https://artifacts.internalmeta.com/a/1158072533227800) |
| Real-cluster 300-step loss parity with eager (Qwen3 30B-A3B, sec. 4) | [artifact 1151430804545666](https://artifacts.internalmeta.com/a/1151430804545666) |
| Eager studies | `qwen3_30b_eager_study/`, `dsv3_671b_eager_study/` |
| Eager vs graph finder study (source of the GEMM efficiency numbers in sec. 3) | `finder_substrate_study/` (protocol, prompts, scoring, both arms' regions and reports, judgements) |
| Kernel packs and their runtime | [github.com/SherlockNoMad/graph-trainer-kernel-packs](https://github.com/SherlockNoMad/graph-trainer-kernel-packs) |
| Skills | `../.claude/skills/fusion_finder/`, `../.claude/skills/autokernel/` |
