# LLEP Performance Optimization Report — PR008

**Model:** deepseek_v3 mini_kimi_k2_llep — 34.3B total params (1.5B active)
**Config:** 256 experts, top_k=8, EP=8, dim=3072, hidden=2048, seq_len=8192, lbs=6
**Hardware:** 8×NVIDIA B200 (single node, NVLink)
**Branch:** `phuc/kimi_k2_with_autotune_llep_optimized_llep`
**Date:** 2026-02-17 → 2026-02-19

---

## 1. Starting Point

LLEP (Least-Loaded Expert Parallelism) is an alternative to standard EP that dynamically redistributes expert workloads across GPUs to handle token imbalance. The original implementation in `torchtitan/distributed/llep.py` was a functional port from the Salesforce GPT-OSS codebase, adapted for SwiGLU (w1/w2/w3, no bias).

**Baseline E2E:**
- Standard EP (no LLEP): **2365 TPS**
- LLEP original: **1470 TPS** (−37.8% vs standard EP)

The goal was to close this gap by optimizing all LLEP-specific code paths.

---

## 2. Files

| File | Role |
|------|------|
| `torchtitan/distributed/llep.py` | Main LLEP implementation (all optimizations applied here) |
| `torchtitan/distributed/llep_kernels.py` | Triton kernels + vectorized numpy helpers (new file) |
| `torchtitan/distributed/llep_original.py` | Pre-optimization snapshot for A/B comparison |
| `torchtitan/distributed/llep_v2_optimized.py` | Latest optimized snapshot |
| `tests/unit_tests/test_llep.py` | Original unit tests (unchanged) |
| `tests/unit_tests/test_llep_correctness.py` | Distributed correctness tests (2-GPU torchrun) |
| `tests/unit_tests/test_llep_bench.py` | Benchmark + correctness suite with OLD reference impls |
| `tests/unit_tests/test_new_kernels.py` | Correctness + benchmark for pad/unpad/send_matrix |
| `tests/unit_tests/profile_llep_components.py` | Isolated profiling of every LLEP function |
| `profiling_output/llep_nsys.nsys-rep` | NSys full trace (no NVTX) |
| `profiling_output/llep_nvtx.nsys-rep` | NSys trace with NVTX phase annotations |

---

## 3. Methodology

1. **Profile** every function inside `llep_moe_forward` at training sizes (393K tokens, 256 experts, dim=3072) using `torch.cuda.synchronize()` + `time.perf_counter()`.
2. **Write kernel/optimization**, verify correctness (bitwise match for data movement ops, relative tolerance for compute ops).
3. **Benchmark** each kernel in isolation vs the PyTorch baseline.
4. **Integrate** into `llep.py` with lazy-cached imports and fallback paths.
5. **E2E benchmark** on mini_kimi_k2_llep_ep8 (25 steps, 8×B200).
6. **NSys profile** the actual training to identify remaining bottlenecks.

---

## 4. Optimization Inventory

### 4.1 Triton `_pad_for_grouped_mm` — 7.6ms → 1.6ms (4.6×)

**Problem:** Each expert's token group must be padded to multiples of 8 for `torch._grouped_mm`. The original vectorized PyTorch path used `cumsum → arange(max_len) broadcast → masked_select → indexed scatter`. With zipf-distributed token counts, `max_len` can be 10–100× the average count, creating huge intermediate tensors (e.g., 256 × 240K = 61M elements for the broadcast).

**What failed:** First Triton attempt used one program per expert with a sequential loop over rows. With 256 experts averaging ~1536 rows each, this was 90ms — 12× slower than PyTorch. The GPU has thousands of SMs but only 256 programs, each doing 1536 serial iterations.

**What worked:** Row-parallel strategy:
1. Precompute `dst_index` for each source row using `torch.repeat_interleave` (single CUDA kernel, O(N)).
2. Triton `_copy_rows_kernel`: one program per (row, col_block). Grid = (393K, ceil(3072/4096)) = 393K programs, massive parallelism.
3. Zero-initialization via `torch.zeros` handles the padding region.

Also tried pure PyTorch with `repeat_interleave` + `x_padded[dst_idx] = x_sorted` (no Triton). This was 6.0ms for pad — faster than the original 7.6ms but slower than Triton's 1.6ms because PyTorch's indexed scatter is less efficient than Triton's contiguous-per-row writes.

**File:** `llep_kernels.py:437-534` (kernel + wrapper), `llep.py:1039-1051` (integration with lazy cache + threshold dispatch)

### 4.2 Triton `_unpad_output` — 7.0ms → 1.0ms (6.5×)

Same strategy as pad but reversed: `_gather_rows_kernel` reads from `src_index[row]` and writes to contiguous `dst[row]`. Slightly faster than pad because gather (contiguous write, random read) has better memory access patterns than scatter (random write, contiguous read).

**File:** `llep_kernels.py:537-636`, `llep.py:1092-1103`

### 4.3 Triton `fused_silu_gate` — 1.35ms → 0.69ms (1.9×)

**Problem:** `F.silu(x1) * x3` materializes 2 intermediate tensors (silu output + multiply output). For 393K × 2048 in bf16, that's 2 × 1.6GB = 3.2GB of unnecessary memory traffic.

**What worked:** Fused kernel: one program per row, computes `x1 * sigmoid(x1) * x3` in fp32 and writes bf16. Includes autograd backward kernel with correct `silu'(x)` gradient. Reduces memory traffic from 5 passes (3 reads + 2 writes) to 3 passes (2 reads + 1 write).

**File:** `llep_kernels.py:32-211` (fwd + bwd kernels + autograd wrapper)

### 4.4 Triton `imbalance_ratio` — 0.07ms → 0.04ms (1.7×)

**Problem:** `counts.view(ep,nle).sum(1).float() → mean → max → div → .item()` = 6 PyTorch op launches.

**What worked:** Single-program kernel: loads all 256 expert counts, computes per-GPU loads via segmented sum (static loop over 8 GPUs), computes max/mean/ratio in one pass. The constexprs (NUM_EXPERTS, NUM_LOCAL) must be powers of 2 for `tl.arange`.

**File:** `llep_kernels.py:221-279`, `llep.py:334-351` (lazy-cached dispatch)

### 4.5 Triton `assign_tokens` — 3.6× vs PyTorch

**Problem:** Python loop over LPT plan + per-GPU scatter in `assign_tokens_to_gpus`. O(num_experts × ep_size) Python iterations.

**What failed first round:** The Triton kernel itself was fast, but the plan encoding was slow. Original approach allocated 4 GPU tensors element-by-element from Python:
```python
for eid, assignments in lpt_plan.items():
    plan_gpu_ids[eid, j] = gpu_id  # GPU tensor write per element!
```
This was 301μs (hundreds of tiny H2D transfers). The Triton kernel was only ~50μs. So the "optimization" was net negative in E2E (-1.9% TPS).

**What fixed it:** Build numpy arrays on CPU, single H2D transfer:
```python
gpu_ids_np = np.full((num_experts, max_assign), -1, dtype=np.int32)
# ... fill numpy arrays ...
plan_gpu_ids = torch.from_numpy(gpu_ids_np).to(device, non_blocking=True)
```
Plan encoding dropped from 301μs → 24.5μs (12×).

**Kernel design:** Grid of (N / BLOCK_SIZE) programs, each processes BLOCK_SIZE tokens. For each token: load expert ID → load plan_count → if has plan, binary search through assignments via `tl.static_range(MAX_ASSIGN)` loop → write target GPU.

**File:** `llep_kernels.py:290-424`, `llep.py:797-808` (integration)

### 4.6 Vectorized `send_matrix` — 2.2ms → 1.2ms (1.9×)

**Problem:** Nested Python loops: `for eid in range(256): for src_rank in range(8): for (dst, start, end) in assignments:`. ~2048 iterations for 256 LPT experts × 8 ranks.

**What worked:** Two-phase numpy vectorization:
1. Non-LPT experts: `np.add.at` with owner mask (vectorized across all non-LPT experts at once).
2. LPT experts: per-expert vectorized overlap computation using `np.maximum/np.minimum` across all 8 src_ranks simultaneously (eliminates inner src_rank loop).

**File:** `llep_kernels.py:645-695`, `llep.py:853-890` (lazy-cached dispatch with fallback)

### 4.7 Selective `_pack_expert_weights` — eliminated `torch.where` double-materialization

**Problem:** Original code used `torch.where(is_native, w1_local[all_indices], foreign_w1[all_indices])`. This materializes BOTH branches (256 × 2048 × 3072 × 2 bytes = 3.1GB EACH), then selects. Total: ~6.3GB memory traffic for w1 alone, ×3 for w1/w2/w3 = 18.9GB.

**What worked:** Selective indexing — only gather from the relevant source for each expert:
```python
native_pos = is_native.nonzero(...)
w1_packed[native_pos] = w1_local[native_local_indices[native_pos]]  # only native
foreign_pos = is_valid_foreign.nonzero(...)
w1_packed[foreign_pos] = foreign_w1_stacked[safe_stacked_idx]  # only foreign
```
Halves memory traffic since each expert's weights are read from exactly one source.

Note: the isolated benchmark showed no improvement because it tested the native-only path (no foreign experts). The improvement only manifests in actual training where foreign experts exist. The 5.4ms total is still dominated by raw HBM bandwidth for reading 1.2GB of weight data.

**File:** `llep.py:934-965`

### 4.8 Batched D2H sync

**Problem:** Original code called `.item()` per element in `_pack_expert_weights` and `assign_tokens_to_gpus`, causing N separate GPU→CPU synchronization points.

**What worked:** Single `torch.stack(all_expert_counts).cpu().numpy()` instead of per-rank `.item()` calls. Also `.cpu().tolist()` for expert counts in `compute_llep_lpt_plan`.

### 4.9 Dispatch threshold for pad/unpad

**Problem:** Vectorized PyTorch pad/unpad was slower than for-loop for small expert counts (<32) due to GPU index-building overhead.

**What worked:** `_PAD_VECTORIZE_THRESHOLD = 32`. Below 32 experts: simple Python for-loop with slice copies. Above 32: Triton kernel. Above threshold was determined empirically on B200.

### 4.10 Cached imports and env vars

**Problem:** `try: import triton_kernel; except: fallback` on every `llep_moe_forward` call = import overhead × 7 layers × 16 calls/step.

**What worked:** Module-level lazy caches:
```python
_fused_silu_gate_fn = None  # set once on first call
_triton_pad_fn = None
_triton_unpad_fn = None
_send_matrix_fn = None
```
Env vars (`EP_MAX_TOKENS_FACTOR`, etc.) cached at import time, not re-read per call.

---

## 5. E2E Results

All measurements on the same node, same config (mini_kimi_k2_llep_ep8, 8×B200).

| Version | Mean TPS (steps 10-25) | vs Original | vs No-LLEP |
|---------|----------------------|-------------|------------|
| No-LLEP (standard EP) | 2365 | — | baseline |
| LLEP original | 1470 | baseline | −37.8% |
| **LLEP optimized** | **1625** | **+10.6%** | **−31.3%** |

Optimization recovered 17.3% of the LLEP-vs-standard-EP gap.

---

## 6. NSys Profiling — Where Time Actually Goes

### 6.1 First pass: kernel-level stats (no NVTX)

Profiled 5 training steps with `nsys profile --trace=cuda,nvtx,osrt,cublas`.

**Trap:** The kernel-level aggregation showed AllToAll as 64% of GPU time. This was misleading — the mean was dominated by step 1 warmup (Triton compilation, CUDA graph setup). The median AllToAll time was only 0.4ms/call.

**File:** `profiling_output/llep_nsys.nsys-rep`

### 6.2 Second pass: NVTX-annotated phases

Added `torch.cuda.nvtx.range_push/pop` around each phase of `llep_moe_forward` and re-profiled.

**Files:** `profiling_output/llep_nvtx.nsys-rep` (phase-level), `profiling_output/llep_deep.nsys-rep` (sub-phase)

**Per-call breakdown (median, steady-state):**

| Phase | Median (ms) | % of LLEP |
|-------|------------|-----------|
| FSDP weight unwrap | 109.20 | 64.6% |
| SwiGLU FFN (total) | 40.10 | 23.7% |
| Imbalance check | 14.18 | 8.4% |
| Assign tokens | 2.67 | 1.6% |
| AllGather counts | 0.80 | 0.5% |
| Barrier | 0.62 | 0.4% |
| AllToAll dispatch | 0.42 | 0.2% |
| LPT plan (CPU) | 0.41 | 0.2% |
| P2P weight transfer | 0.35 | 0.2% |
| AllToAll combine | 0.25 | 0.1% |
| Unsort + aggregate | 0.14 | 0.1% |
| Wait P2P | 0.00 | 0.0% |
| **TOTAL** | **169.14** | 112 calls/step → **18.9s/step** |

### 6.3 Deep FFN breakdown

The 40.1ms SwiGLU FFN phase breaks down into:

| Sub-phase | Median (ms) | % of FFN | Notes |
|-----------|------------|----------|-------|
| Sort + unique_consecutive | 18.90 | 48.8% | Includes async GEMM wait (see 6.4) |
| Unpad (Triton) | 13.83 | 35.7% | Includes async GEMM wait (see 6.4) |
| Pack weights | 3.52 | 9.1% | Real cost: 1.2GB indexed gather |
| Pad (Triton) | 1.93 | 5.0% | Close to isolated benchmark (1.6ms) |
| silu_gate (Triton) | 0.15 | 0.4% | Fast |
| grouped_mm w1 | 0.11 | 0.3% | **Async launch only** |
| grouped_mm w2 | 0.07 | 0.2% | **Async launch only** |
| grouped_mm w3 | 0.07 | 0.2% | **Async launch only** |
| Postprocess | 0.17 | 0.5% | Cast + argsort |

### 6.4 Critical insight: async GEMM timing

The grouped_mm medians (0.07–0.11ms) are **CPU-side launch times only**. The actual GPU compute (~10ms per GEMM, ~30ms total for 3 GEMMs) is asynchronous and appears in the **next synchronizing operation**:

- `sort_tokens` (18.9ms) = ~2ms real sort + ~17ms waiting for backward GEMMs from prior layer
- `unpad` (13.8ms) = ~1ms real unpad + ~13ms waiting for grouped_mm_w2 forward compute

**Corrected real-cost picture:**

```
Per llep_moe_forward call (~169ms median wall-clock):
├─ FSDP weight all-gather:     109.2ms  (65%) — DTensor redistribute + to_local
├─ grouped_mm compute:          ~30ms   (18%) — 3 CUTLASS GEMMs (hidden in sort/unpad timings)
├─ Imbalance sync wait:         14.2ms  ( 8%) — waiting for async AllGather to finish
├─ Pack weights:                 3.5ms  ( 2%) — indexed gather, 1.2GB bandwidth-bound
├─ Pad (Triton):                 1.9ms  ( 1%) — row-parallel scatter
├─ Token sort + gather:          ~2ms   ( 1%) — radix sort + 2.4GB gather (real, minus async wait)
├─ Assign tokens:                2.7ms  ( 2%) — Triton kernel + send_matrix + argsort
├─ Unpad (Triton):               ~1ms   ( 1%) — row-parallel gather (real, minus async wait)
├─ AllToAll dispatch+combine:    0.7ms  (<1%) — NVLink, very fast on single node
├─ LPT plan + barrier + P2P:    1.4ms  (<1%) — all small
└─ Other:                        ~2ms   ( 1%) — allgather counts, unsort, postprocess
```

### 6.5 Key findings

1. **FSDP weight unwrap is 65% of LLEP time (109ms/call).** This is `w1.redistribute(placements=new_placements).to_local()` which triggers an FSDP2 all-gather to reassemble 3 expert weight tensors (32×2048×3072 bf16 = 400MB each, 1.2GB total) from their sharded state. This is **outside LLEP scope** — it's FSDP infrastructure.

2. **grouped_mm is ~30ms real compute (18% of LLEP)** — 3 CUTLASS grouped GEMMs on (393K padded tokens × 3072 dim) × (32 experts × 2048 hidden). This is compute-bound and well-optimized by PyTorch's CUTLASS backend. Not optimizable from our side.

3. **Imbalance check shows 14.2ms but the Triton kernel is 0.04ms.** The 14ms is the implicit GPU sync point where the previously-launched async AllGather (`dist.all_gather`) actually blocks waiting for the NCCL operation to complete. The real imbalance computation is negligible.

4. **AllToAll is NOT the bottleneck on single-node NVLink.** 0.42ms + 0.25ms = 0.67ms per call. On multi-node InfiniBand this will be dramatically different.

5. **Our Triton kernels (pad, unpad, silu_gate, assign, imbalance) are all <2ms real cost.** They're well-optimized and no longer bottlenecks.

6. **pack_weights at 3.5ms is the last feasible LLEP-scope optimization** — the pre-allocate-and-recv-directly idea would save this.

---

## 7. Remaining Optimization Opportunities

### 7.1 Pre-allocate packed weights buffer (eliminate pack_weights — ~3.5ms)

**Idea:** Instead of P2P recv → separate buffer → copy to packed buffer, pre-allocate the packed buffer in sorted expert order and P2P recv directly into the correct positions. This eliminates the 3.5ms pack_weights gather entirely.

**Challenge:** The packed order must match the token sort order, which depends on `recv_experts` from AllToAll. But we can predict which experts will appear from the LPT plan + output_split_sizes, computed before AllToAll.

**Estimated savings:** ~3.5ms × 112 calls/step = 392ms/step → ~1.5% TPS improvement.

### 7.2 Overlap FSDP weight gather with prior layer's compute (eliminate 109ms — outside LLEP scope)

**Idea:** The 109ms FSDP weight materialization happens synchronously before each LLEP call. If the model's layer loop prefetched the next layer's weights while the current layer runs FFN + AllToAll combine, this 109ms would be hidden.

**Challenge:** Requires restructuring torchtitan's model forward loop to pipeline weight prefetch. This is outside LLEP scope — it's a training framework optimization.

**Estimated savings:** Up to 109ms × 112 calls/step = 12.2s/step → potentially **40%+ TPS improvement** if fully hidden. This is by far the largest opportunity.

### 7.3 Reduce async sync bubbles

**Idea:** The 14.2ms "imbalance check" and inflated sort/unpad times are caused by implicit synchronization points where async NCCL and GEMM operations force a wait. Better pipelining of async operations could reduce these bubbles.

**Challenge:** Requires careful CUDA stream management and potentially breaking the sequential structure of llep_moe_forward.

### 7.4 Fuse sort + gather (save ~1ms)

**Idea:** `expert_ids.sort()` followed by `x[sort_perm]` could be fused into a single kernel that does the sort and gather in one pass, avoiding materializing `sort_perm`.

**Challenge:** Moderate complexity Triton kernel. Small savings (~1ms).

---

## 8. Lessons Learned

1. **Profile before optimizing.** The initial assumption was that Python loops and PyTorch ops were the bottleneck. After the first round of Triton kernels, profiling revealed plan encoding (301μs → 24.5μs) was the real issue. After the second round, nsys showed AllToAll wasn't the bottleneck (contrary to the kernel-level stats which were skewed by warmup). The deep profiling revealed FSDP weight unwrap is actually the #1 cost.

2. **Use median, not mean.** Step 1 warmup (Triton compilation, CUDA graph capture, JIT) massively skews averages. The median AllToAll time was 170× smaller than the mean.

3. **Async kernels make NVTX timing misleading.** grouped_mm shows 0.07ms median because it's an async launch. The real ~10ms GPU compute appears in the NEXT synchronizing op (sort, unpad). You must understand the async execution model to interpret nsys NVTX correctly.

4. **Triton kernel parallelism matters more than algorithm cleverness.** The first pad/unpad kernel used one program per expert (256 programs, each looping over ~1536 rows) = 90ms. The second version used one program per row (393K programs) = 1.6ms. Same algorithm, 56× speedup just from parallelism.

5. **H2D transfer granularity matters.** Building GPU tensors element-by-element from Python (plan encoding) was 12× slower than building numpy arrays and doing a single H2D transfer.

6. **`torch.where` materializes both branches.** For large tensors (3.1GB per branch), this doubles memory traffic. Selective indexing with `nonzero()` + gather halves it.

7. **Dispatch thresholds are real.** For-loop beats vectorized PyTorch for <32 experts due to GPU kernel launch overhead. The crossover point varies by hardware.

8. **FSDP weight materialization is the hidden dominant cost.** It doesn't show up as an "LLEP kernel" in profiling because it's DTensor infrastructure (`redistribute().to_local()`), but it's 65% of LLEP wall-clock time. The single biggest optimization opportunity is pipelining this with the prior layer's compute.

9. **On single-node NVLink, AllToAll is cheap.** The 0.67ms/call for dispatching+combining 393K × 3072 bf16 tokens across 8 GPUs is negligible. This will be very different on multi-node InfiniBand where AllToAll could be 10-100× slower.

10. **nsys kernel-level stats can mislead.** The kernel summary showed AllToAll as 64% of GPU time. This was correct for raw kernel time but irrelevant for optimization because (a) step-1 warmup dominated the mean, and (b) the FSDP all-gather was categorized separately from LLEP in the kernel view. NVTX phase-level profiling gave the actionable picture.
