# MoE decode: PyTorch, Inductor and CUTLASS asks from TorchTitan's RL generator

[Felipe Mello](mailto:felipemello@meta.com)

## TL;DR

- TorchTitan's RL generator runs the TorchTitan model inside vLLM. At decode, its MoE layer is slower than vLLM's, which runs FlashInfer's fused TRT-LLM MoE kernels.
- One Qwen3.5-35B-A3B MoE layer's routed experts at T=34 (34 sequences decoding), CUDA-graph replay:
  - ATen `torch._grouped_mm` today: 252 us.
  - With a small-group tile fix in ATen (ask 1): 205.6 us.
  - FlashInfer: 167.5 us.
- The remaining 38 us:
  - **GEMM side, 21 us:** w13 +4.9, w2 +7.5, 2 grouped-GEMM setup kernels +4.1, launch gaps ~4.
  - **Glue, 17 us:** dispatch 16.2 vs 5.6 us (8 kernels vs 1, plus a cast); SwiGLU 2.0 us as its own kernel (FlashInfer fuses it into its first GEMM); the rest is launch gaps (our 14 kernels without PDL vs FlashInfer's 5 with it). The combine is the exception: ours is faster (2.8 vs 6.3 us).
  - These walls come from a process that also ran the profiler, which adds ~0.3 us per kernel to later CUDA-graph replays (see Known behaviors), so part of the launch-gap share is the profiler.
- 12 asks, each with numbers at real shapes and a repro in `repros/`. 8 are pure torch; ask 2 needs out-of-tree CUTLASS builds and ask 7 needs cutlass-dsl 4.8.
  - Decode: small-group grouped-GEMM tiles (48 us/layer), a ragged MoE GEMM path without per-group descriptors (~10), dispatch codegen (~7-10).
  - Prefill: bounds masks under dynamic shapes (~20 us/layer at 4096-token chunks), and dynamic kernels tuned at their first size (up to ~110 us/layer when that size is small).
- One GB300, torch 2.16.0.dev20261009 (git 146a5a2f73); every repro run on 2026-10-15.

**Terms.**
- Routed experts: the MoE part after the router. Dispatch sorts the T x 8 routes by expert and gathers their token rows; w13 (gate+up) and w2 (down) are grouped GEMMs over 256 experts; SwiGLU sits between them; the combine is each token's score-weighted sum of its 8 rows.
- Rows per expert: routed rows / 256. At T=34 it is ~1, so each expert's GEMM reads 4 MiB of weights for about one row: decode is bound by reading weights.
- Roofline: bytes of the weights of every expert that got at least one row, at 8 TB/s.
- Region: a function TorchTitan compiles on its own (`torch.compile(fn, fullgraph=True)`), as in the parent doc. The generator compiles each region static at its first size and dynamic at the second.
- FlashInfer: its TRT-LLM bf16 routed MoE, 5 kernels: routing, bmm1 (gather + w13 + SwiGLU), bmm2 (w2), finalize (combine). The 5 kernels overlap through programmatic dependent launch (PDL).

**Scale.** Qwen3.5-35B-A3B has 40 MoE layers. A decode step with 34 sequences takes ~23 ms in vLLM with the TorchTitan model, so 1 us per MoE layer is ~0.17% of a decode step. Prefill numbers use 4096-token chunks.

```text
#   kind               ask                                                              impact (per MoE layer; share of a ~23 ms decode step)
A. grouped GEMM (ATen / CUTLASS)
1   ATen kernel        small-group tiles for torch._grouped_mm (2D x 3D)                48 us at T=34; ~8%
2   ATen + CUTLASS     ragged MoE path: one weight descriptor, offs on device, no setup ~10 us at T=34; ~1.7%
3   ATen API           gather rows by index (A prologue) + gated-activation epilogue     ~4-6 us decode; ~49 us at T=4096
4   launch             PDL by default for Inductor kernels and the grouped GEMM          -1.8 us at T=34; +34 us at T=4096 today
B. Inductor
5   fusion             no prologue/epilogue fusion into the grouped_mm Triton template  enables ask 3 under compile
6   template configs   Triton grouped_mm template has no small-M configs                1.4x slower than ATen at T=1
7   template launcher  CuteDSL grouped_mm template: D2H sync, 1539 kernels, no capture  unusable at decode
8   autotuning         dynamic kernel keeps the config tuned at its first (small) size  1.8x on SwiGLU; up to ~110 us at T=4096
9   codegen            bounds masks under symbolic T slow a per-token gather-sum 1.7x   ~20 us at T=4096
10  codegen            dispatch: scan -> scatter -> gather = 5 kernels; looped scan      ~7-10 us at T=34; ~1.2-1.7%
11  fusion             router: per-row softmax, top-k, gather, normalize = 3 kernels      ~7 us at T=34; ~1.2%
12  codegen            split reduction chosen at the size hint, never at runtime         ~3-5 us at T=34; ~0.6-0.9%
```

Repros: `./run_all.sh gpu` (pure torch) and `./run_all.sh ext` (asks 2 and 8, see `run_all.sh`). Raw outputs: `logs/`.

---

## A. Grouped GEMM (ATen / CUTLASS)

### 1. Small-group tiles for `torch._grouped_mm`

**Ask:** pick a tile for 2D x 3D grouped GEMMs from the rows per group. Swap A and B (`out_g^T = W_g @ x_g^T`) when groups have <= 96 rows, so the weight rows fill a 256-row tile and the tokens go to a 32- or 128-wide N tile.

**Impact:** 48 us per MoE layer at T=34 (w13 + w2 kernels 216 -> 168 us), ~1.9 ms per decode step (~8%).

```text
torch._grouped_mm, Qwen3.5 experts, uniform top-8 routing   (logs/ask01)
w13 T=1    24.8 us   17% of roofline        w2 T=1    21.3 us   10%
w13 T=34  151.7 us   60%                    w2 T=34   93.7 us   49%
w13 T=128 209.9 us   64%                    w2 T=128 121.3 us   55%
FlashInfer bmm1 / bmm2 at T=34: 88% / 84% of the same roofline
```

**Why:** ATen's sm100 grouped GEMM has two fixed configs: a 128x256x64 1SM tile when the average rows per group is <= 128 or N <= 128, otherwise 256x256x64 2SM, both with a 2x1 cluster. With ~1 row per expert, each 128-row tile holds one real row and the cluster's second CTA idles.

**Workaround:** a draft GroupMM.cu change (swap-AB below 24 / 96 rows per group, a 32- or 128-wide N tile, 2SM). 1.3-1.5x on 9 MoE shapes, never slower than ATen where it applies, bitwise equal to `torch._grouped_mm`. Over 33 tile configs, the best is only 2% better than this 2-config rule.

**Repro:** `repros/ask01_grouped_mm_small_groups.py`.

### 2. A ragged MoE grouped-GEMM path without per-group descriptors

**Ask:** a 2D x 3D path for `torch._grouped_mm` built on CUTLASS's MoE kernel (examples/92: `MoEProblemShape`): one batched TMA descriptor for the weights, cp.async for the token rows, `offs` read on device, no setup kernel. CUTLASS's version takes tokens padded per expert; this needs it for ragged rows (offsets into one contiguous buffer).

**Impact:** ~10 us per MoE layer at T=34 (w13 -2.2, w2 -3.4, setup kernels -4.1), ~1.7%.

```text
same weights and rows, CUDA graph, min of 2 runs   (logs/ask02)
                ptr-array (today, with ask 1's tiles)   CUTLASS MoE kernel    bitwise equal
w2  T=8          27.9 us   54% of roofline                24.1 us   63%          True
w2  T=34         68.9 us   67%                            65.5 us   70%          True
w13 T=34        118.4 us   77%                           116.2 us   79%          True
w13 T=128       165.6 us   81%                           169.4 us   79%          True
+ prepare_grouped_gemm_data: ~2 us per GEMM call, not needed by the MoE kernel
```

**Why:**
1. ATen's kernel is CUTLASS's ptr-array grouped GEMM: it rewrites a TMA descriptor for each group (171 groups at T=34), and a setup kernel (`prepare_grouped_gemm_data`) builds the per-group pointers before each call.
2. FlashInfer and CUTLASS's MoE kernel index all experts through one descriptor. The gap is largest where groups are smallest: w2 at T=8 is 14% faster.

**Workaround:** none in PyTorch.

**Repro:** `repros/ask02_ptr_array_vs_moe_kernel.py` (needs out-of-tree CUTLASS builds, see its docstring).

### 3. Gather rows by index and a gated-activation epilogue in `torch._grouped_mm`

**Ask:**
- (a) an optional row-index tensor for A, so the GEMM gathers token rows in its load (cp.async) instead of reading a pre-gathered copy;
- (b) a gated-activation epilogue on w13 (`act(acc[:, :F]) * acc[:, F:]`), or a generic epilogue hook (CUTLASS EVT).

**Impact:** ~4-6 us per MoE layer at decode (2 kernels and their launch gaps), ~49 us at T=4096.

```text
separate kernels next to w13, compiled, CUDA graph, kernel time   (logs/ask03)
T=34     gather  1.8 us + w13 144.2 us + SwiGLU  1.2 us
T=256    gather  6.0 us + w13 204.2 us + SwiGLU  2.5 us
T=4096   gather 34.7 us + w13 226.8 us + SwiGLU 14.7 us
FlashInfer's bmm1 does the gather in its cp.async load and SwiGLU in its epilogue
```

**Why:** at decode the intermediates are small (2-3 MB at T=34); the cost is two extra launches. At T=4096 the gathered copy is 268 MB written and read again, and w13's output is 134 MB.

**Workaround:** none in PyTorch; vLLM uses FlashInfer.

**Repro:** `repros/ask03_gather_and_swiglu.py`.

### 4. PDL for Inductor's Triton kernels and for the grouped GEMM

**Ask:**
- (a) find why `TORCHINDUCTOR_ENABLE_PDL=1` makes a prefill-size MoE block 5% slower, then turn PDL on by default under CUDA graphs;
- (b) launch ATen's CUTLASS grouped GEMM with PDL (CUTLASS supports it through the GDC define and the launch attribute).

**Impact:** small today. For (a) the whole MoE block gains 1.8-3 us at decode (bitwise equal) but loses 34 us at T=4096, so we don't turn it on. (b) is not measured (est. 1-2 us per GEMM). FlashInfer's 5 kernels overlap through PDL.

```text
MoE dispatch region (5 Inductor kernels), CUDA graph   (logs/ask04)
TORCHINDUCTOR_ENABLE_PDL=0:  T=1 11.8 us | T=8 13.3 | T=34 12.2 | T=256 19.7     bitwise vs eager: True
TORCHINDUCTOR_ENABLE_PDL=1:  T=1  8.3 us | T=8 10.4 | T=34 11.6 | T=256 17.9     bitwise vs eager: True
whole Qwen3.5 MoE block, generator compile order, no profiler, min of 2 processes per arm; outputs bitwise equal:
  T=1 84.4 -> 81.3 us | T=34 284.4 -> 282.6 | T=64 343.8 -> 342.6 | T=256 401.6 -> 399.6 | T=4096 632.0 -> 665.8
```

**Workaround:** none (we leave it off).

**Repro:** `repros/ask04_pdl.py` (`run_all.sh` runs it with PDL off and on).

---

## B. Inductor

### 5. No prologue or epilogue fusion into the grouped_mm Triton template

**Ask:**
- (a) allow prologue inputs on the grouped_mm template's A operand, including an indexed gather;
- (b) a paired-column epilogue (`act(acc[:, f]) * acc[:, F + f]`), e.g. a template option that tiles N so a tile holds both the gate and the up columns.

The combine is a per-token reduction over 8 rows and needs no ask; FlashInfer keeps it separate too.

**Impact:** this is how ask 3 would arrive under compile: the gather and SwiGLU kernels (1.8 + 1.2 us at T=34, 6.0 + 2.5 at T=256) and their launches. With `_grouped_mm` lowered to the extern ATen kernel (the default), Inductor fuses nothing into it.

```text
one compiled function: dispatch -> gather -> _grouped_mm -> SwiGLU -> _grouped_mm -> gather-sum   (logs/ask05)
default (extern ATen GEMM):         T=34  11 kernels, graph 237.2 us (GEMM 210.6, glue 18.1)
max-autotune, TRITON template only: T=34   9 kernels, graph 225.1 us (GEMM 200.1, glue 15.0)
  gathered rows and the SwiGLU output are still separate kernels and buffers; fusion log:
    gather -> w13 template:      template has no allowed prologue inputs
    SwiGLU -> w2 template:       template has no allowed prologue inputs
    SwiGLU as w13 epilogue:      memory deps did not match
    combine as w2 epilogue:      template epilogue not satisfied
```

**Why:** the template allows no prologue inputs, so no producer fuses into it. SwiGLU reads two output columns per element (gate f and up F+f), so it is not pointwise over the template's output index.

**Workaround:** none.

**Repro:** `repros/ask05_grouped_mm_template_fusion.py` (`default` and `triton` modes; the triton run prints the fusion-log reasons).

### 6. The Triton grouped_mm template has no small-M configs

**Ask:** add swapped or narrow-N configs (rows per group <= 32) to the template's search space, or let max-autotune pick ATen's CUTLASS kernel with ask 1's tiles.

**Impact:** with 1-8 rows per expert the template is 1.4x slower than ATen at T=1, and slower than ATen with ask 1's tiles at every decode size.

```text
w13, max-autotune with TRITON only, CUDA graph   (logs/ask06)
T=1     ATen  24.1 us | Triton template  33.1 us
T=34    ATen 151.3 us | Triton template 135.8 us   (ATen with ask 1's tiles: ~107-112 us)
T=128   ATen 209.5 us | Triton template 186.7 us
```

**Workaround:** none needed (we use ATen).

**Repro:** `repros/ask06_triton_template_small_m.py`.

### 7. The CuteDSL grouped_mm template isn't usable at decode

**Ask:** compute the cluster count on device (or use an upper bound) instead of `int(compute_total_num_clusters(...))`, build the per-group metadata in one kernel, and make the call capturable in a CUDA graph.

**Impact:** unusable: its GEMM kernel is ATen-level, but the call is ~20x slower and can't run in vLLM's CUDA graphs.

```text
w13 T=34, profiled eagerly (capture fails)   (logs/ask07)
GEMM kernel 142.6 us | all device work per call 2807 us | 1539 kernels per call | 1 D2H sync per call
CUDA graph capture fails: cudaErrorStreamCaptureInvalidated
```

**Why:** the launcher builds the per-group problem metadata with ~1500 small torch ops on the group sizes, then reads a scalar back to the host (`aten::_local_scalar_dense`), which breaks capture.

**Workaround:** none needed (we use ATen).

**Repro:** `repros/ask07_cutedsl_template.py` (needs cutlass-dsl 4.8; 4.7.1 fails).

### 8. A dynamic kernel keeps the launch config tuned at its first dynamic call

**Ask:** re-tune dynamic pointwise and reduction kernels per size bucket (e.g. powers of 2), key the saved `.best_config` by bucket, and don't commit a pick made on inputs where every config ties.

**Impact:** prefill only, and only when a region's second call is small: ~110 us per MoE layer at 4096-token chunks (SwiGLU 67 vs 19 us static, per-expert counts 51 vs ~10, shared-expert gate 34 vs ~10). vLLM's generator calls the regions at 16,384 tokens first, so it is not hit; a trainer whose first steps or evaluation use small batches is.

```text
SwiGLU on [R, 2, 512] routed rows; us at R = 131,072 / 524,288; one fresh process per line   (logs/ask08)
decode first (272, then 512 rows):     110.6 / 422.3    109.8 / 422.1    110.4 / 422.0
large first (131072, then 114688):      65.4 / 237.6     63.7 / 236.5
static compile at each size:            63.6 / 235.1
```

**Why:** Inductor compiles the first size static and the second dynamic. The pointwise autotuner picks XBLOCK and num_warps at the dynamic kernel's first launch and keeps them for every later size. At 272-512 rows every config ties, so the pick is arbitrary and is then reused at 524,288 rows. The pick also persists in `.best_config` on disk, so it leaks into later processes.

**Workaround:** compile and tune small and large inputs as two separate compiled copies (a size split in TorchTitan's region wrapper, in progress).

**Repro:** `repros/ask08_dynamic_autotune_first_size.py`.

### 9. Bounds masks under a symbolic token count slow a per-token gather-sum

**Ask:** choose 2D tiles automatically when the inner dim is static and the outer dim is symbolic, or drop the mask when the static inner dim is a multiple of the block.

**Impact:** the MoE combine is 1.7-1.8x slower than static: ~20 us per MoE layer at 4096-token prefill chunks, ~285 us at 65,536 tokens. This is a different kernel from the parent doc's ask 12 (masked loads on SM103), with the same root: a mask Inductor can't prove unnecessary.

```text
combine: per-token score-weighted sum of 8 rows of 2048, compiled large-first; us at T = 4096 / 16384 / 65536   (logs/ask09)
dynamic                       49.5 / 158.5 / 636.4
dynamic + prefer_nd_tiling    29.1 /  94.5 / 397.1
static                        28.9 /  92.1 / 351.0
```

**Why:** with a symbolic T, numel = 2048 T is not a known multiple of the largest XBLOCK, so the 1D kernel keeps `xindex < xnumel` on every load. Compiled large-first, so this is not ask 8.

**Workaround:** `triton.prefer_nd_tiling` on this region only (it slows SwiGLU by ~6% and costs +13% on the static combine at 64k).

**Repro:** `repros/ask09_dynamic_bounds_mask.py`.

### 10. Dispatch codegen: a scan chain becomes 5 latency-bound kernels

**Ask:**
- (a) for tiny problems, emit the chain one-hot cumsum -> scatter -> gather as one kernel (one CTA, or a grid sync);
- (b) under dynamic shapes, keep small scans persistent (`triton_per`): multi-kernel doesn't cover scans today.

**Impact:** ~7-10 us per MoE layer at decode (~1.2-1.7%). FlashInfer's routing kernel does the same expert sort and permutation, plus softmax and top-k, in one 2.7-3.9 us kernel.

```text
TorchTitan's counting-sort dispatch, compiled, CUDA graph   (logs/ask10)
(a) static:  5 kernels at every size; T=34 kernel sum 8.5 us, graph 18.5 us; T=1 6.8 / 12.3 us
(b) dynamic (compiled at 2048, then 1024 tokens): one-hot cumsum is a looped reduction (triton_red)
    T=34: 4.9 us vs 2.1 us static (triton_per); T=64: 5.4 vs 2.4
TORCHINDUCTOR_MULTI_KERNEL=1, with or without multi_kernel_hints: no change
```

**Why:** each scan (the one-hot cumsum, then the per-expert offsets) and the `index_put` that the gathers read end an Inductor kernel. At decode every kernel is 1.2-2.1 us of launch-bound work.

**Workaround:** rank the routes without a scan. Each route's row is the number of routes with a smaller `(expert, route)` key: one `(N, N)` compare-and-count, which Inductor fuses with the sort's `index_put` into one kernel. Dispatch at T=34: 5 kernels (9.1 us) -> 3 (5.1 us). It is O(N^2), so TorchTitan uses it up to 2048 routes (256 tokens) and a radix sort above. The ask stands for scans in general.

**Repro:** `repros/ask10_dispatch_codegen.py`.


### 11. The router's per-row work becomes 3 kernels

**Ask:** fuse a chain of row reductions over one token's scores into one persistent kernel, even when a later step reduces over a different length (the k=8 picks after the E=256 argmaxes) or reads a `cat` of earlier reductions.

**Impact:** ~7 us per MoE layer at T=34 (~1.2%): the graph takes 14.2 us for 7.5 us of kernels.

```text
router after the gate GEMM, compiled in the generator's order, CUDA graph   (logs/ask11)
gather/scatter (TorchTitan today)   T=34: graph 14.2 us, 3 kernels 4.6 + 0.8 + 2.1
row ops (masked sums, OR of masks)  T=34: graph 19.5 us, 3 kernels 4.8 + 5.6 + 1.3
```

**Why:** the softmax and the 8 argmax rounds fuse into one persistent kernel, which writes the ids (`cat`). The pick gather and the map scatter start a second kernel; the zeros for the map is a third. Written as row ops, the second kernel recomputes the softmax, and the normalization over the 8 picks (a different reduction length) splits off a third.

**Workaround:** none in TorchTitan; a hand-written routing kernel would replace the region.

**Repro:** `repros/ask11_router_row_gather.py`.


### 12. A split reduction is chosen from the size hint, never at runtime

**Ask:** decide split vs single-kernel reductions per call (as multi-kernel does for persistent vs looped), or let a region pass a runtime threshold.

**Impact:** ~3-5 us per MoE layer at T=34 (~0.6-0.9%): tokens per expert costs 2 kernels and a launch gap where 1 kernel takes 1.2 us.

```text
tokens per expert = (T, E) bool map .sum(dim=0), compiled dynamic in the generator's order, kernel sums   (logs/ask12)
                 T=34          T=256        T=4096        T=16384
default          4.5 (2 krn)   4.4 (2)      4.9 (2)       5.7 (2)
no split         1.2 (1)       1.6 (1)      6.1 (1)       19.9 (1)
```

**Why:** with a large size hint the column sum is split into a partial-sum kernel and a final kernel. The choice is baked into the dynamic kernel, so decode sizes pay for 2 launches. Turning splits off per region fixes decode and makes prefill 3.5x slower.

**Workaround:** none clean. Per-region `split_reductions=False` trades prefill for decode; computing the counts inside the consuming kernel removes the reduction.

**Repro:** `repros/ask12_split_reduction_size_hint.py`.

---

## Not asks (TorchTitan work)

- **Shared expert on a side stream** under `no_grad`, as vLLM does: -7.6 us per MoE layer at T=34 (measured).
- **Skip the router's routing map, per-expert counts and eager offsets cumsum** at inference; the dispatch recomputes the counts: ~-10 us per layer at T=34 (6 kernels, estimate).
- **Size-split region compile and 2D tiles:** the workaround for asks 8 and 9.
- **Sort narrow keys for prefill:** argsort of int64 expert ids is a radix sort with 8 passes; on uint8 ids it is 1 pass, same order. 32,768 routes (T=4096): 69 -> 24 us.
- **Opt-in FlashInfer routed experts in the generator:** -74 us per layer at T=34. The generator's MoE output would then differ from the trainer's by ~5e-3 relative L2, instead of matching within ~5e-5.

## Known behaviors (one line each)

- **vLLM's compile mode traces once at 16,384 tokens and drops guards**, so a Python branch on the token count is fixed at trace time: TorchTitan's dispatch (counting sort up to 16,384 routes, argsort above) takes the argsort path at every decode size, ~23 us per MoE layer of radix sort. This is vLLM's integration, not Inductor.
- **A torch.profiler session slows later CUDA-graph replays in the same process,** by ~0.3 us per kernel (16 tiny kernels: 24.9 -> 29.9 us). Static and dynamic Inductor kernels replay at the same speed before any profiling (24.9 vs 24.5 us). Measure walls before profiling, or in a separate process. Repro `repros/kb_profiler_slows_graph_replay.py`.
- **No graph breaks and one static-to-dynamic recompile per region** from T=1 to 4096 in the MoE block (`TORCH_LOGS=graph_breaks,recompiles`).

## How we measured

- One GB300 (SM103), shared with other jobs. torch 2.16.0.dev20261009+cu132 (git 146a5a2f73), Triton 3.9.0, CUDA 13.2, CUTLASS 4.8.0. FlashInfer 0.7.0.post1 and vLLM 0.31.0 for the comparison.
- Qwen3.5-35B-A3B routed experts: E=256, D=2048, F=512, top-8; bf16 weights; uniform random routing in the repros, the model's router in the layer numbers.
- Times are CUDA-graph replay: the min over 3 rounds of the median of 100-200 replays. Kernel times come from the profiler, as the min over 20 replays at the same position in the graph.
- The headline breakdown and the whole-block numbers come from TorchTitan's MoE layer harness (same weights, router ids and scores on both sides). Repro numbers: `logs/`, one run per script on 2026-10-15 (ask 2: min of 2 runs).
- Per-step shares multiply a per-layer saving by 40 layers and divide by ~23 ms, the measured decode step (34 sequences) of the TorchTitan model in vLLM.
