# Kernel-fusion opportunities in one Qwen3 30B-A3B training step (GraphTrainer, graph arm)

## 1. Setup

- Program: Qwen3 30B-A3B (48 decoder layers, d_model 2048, 32 q heads / 4 kv heads x 128, 128 experts top-8, 16 local experts per rank with intermediate 768, vocab 151936) from `qwen3_30b_a3b_dist_moe_mxfp8_8gpu`. Attention linears (wqkv, wo) are MXFP8 (`_scaled_mm_v2` with torchao `mxfp8_quantize`); routed experts run in Dist-MoE (`dist_moe.block_scaled_forward/backward`, MXFP8 grouped GEMMs with fused dispatch/combine and SwiGLU); attention is FlashAttention (CuTe DSL) varlen with causal document masking; RMSNorm is the fused ATen/quack kernel; RoPE runs eagerly in fp32; the router gate runs in fp32 output with a hi/mid/lo bf16 split for its backward; the loss is an 8-chunk (4096 tokens each) LM-head + cross-entropy.
- Layout: FSDP 8 x EP 8 (fake process group: one process runs rank 0 and emulates collectives with local copies), one 32,768-token microbatch, memory_policy `none` (nothing recomputed), local_compile off, regional Inductor present but it changed nothing (pre-Inductor and final graphs are identical), deterministic algorithms on, CUDA graphs off. Step 10 profiled after warmup.
- Routing in this benchmark is balanced round-robin: expert id of (token t, slot k) is (8t+k) mod 128, so routing is data-independent.
- Roofline: NVIDIA GB300, HBM copy bandwidth 6921.3 GB/s (used for all memory-bound ideals), MXFP8 GEMM peak 3615 TFLOP/s, BF16 GEMM peak 1930 TFLOP/s (used only for the per-op attribution table; no region contains GEMM flops).
- Region ideal = external bytes / HBM bandwidth (region inputs read once, outputs written once). For T3 regions that fold glue into a GEMM/attention prologue or epilogue, the ideal counts only the extra bytes the epilogue would add (e.g. writing an extra fp8 copy), since the GEMM/attention already reads/writes its own operands.

## 2. Step budget

| category | ms | % of step busy |
|---|---|---|
| gemm | 319.02 | 33.6 |
| attention | 157.77 | 16.6 |
| communication | 26.18 | 2.8 |
| memory_bound_glue | 409.10 | 43.1 |
| optimizer | 27.14 | 2.9 |
| other | 9.75 | 1.0 |
| **step GPU busy (union)** | **948.58** | 100 |

- Forward+backward GPU busy: 911.68 ms (kernel-time sum 912.06 ms). Optimizer (fused AdamW): 27.14 ms. `other` = post-backward gradient clipping and bookkeeping outside the graph (9.75 ms: `_foreach_norm`, `_foreach_mul_`, one `index_put_`). Out of scope for regions.
- `communication` = 3.24 ms of collective stand-in copies under c10d ops + 22.94 ms of fake-PG-only zero-fills of the 1 GiB expert-parallel peer-scatter buffers inside the MoE ops (see no_opportunity).
- Not counted: 21.70 ms of the previous step's AdamW kernels that were launched before the profiler window but executed inside it.
- Per-op attribution (module, layer, phase, shapes/dtypes, bytes, flops, ideal, measured/ideal) is in `out/node_attribution.csv`; per-GPU-event attribution (category, region, op, module, layer, phase) in `out/kernel_attribution.csv`.
- Attribution coverage: 99.41% of forward+backward kernel time maps to a graph node through its `<node>: <op> (<file:line>)` label; the remaining 0.59% (435 device-to-device copies) maps through the Python stack to the post-graph `param.grad = grad.clone()` in the gradient-accumulation helper. Total: 100%.
- GPU idle: 56.5 s over the step (56.3 s inside forward+backward), i.e. the GPU is busy only 1.7% of the step's span. This is CPU launch overhead amplified by the profiler (Python stacks + shapes recorded for ~1.9 M events, a record_function per graph node); it says nothing about the unprofiled step, so the ranking below uses GPU time only.
- Memory-bound glue is the largest category (43% of busy time); the summed region gap is 329.6 ms (34.7% of step busy time).

## 3. Ranked regions

| id | title | phases | scope | kind | inst | measured ms | ideal ms | gap ms | % step | tier |
|---|---|---|---|---|---|---|---|---|---|---|
| R-01 | RoPE backward + q/k RMSNorm backward + dQKV concat + MXFP8 quantize of dQKV | backward | cross-module, horizontal | pointwise + row-reduction chain with interleaved write and MXFP8 quantize epilogue | 48 | 122.76 | 6.88 | 115.88 | 12.22 | T1 |
| R-02 | QKV split + q/k RMSNorm + RoPE forward (fp32 round trip) | forward | cross-module, horizontal, cross-layer | pointwise+row-reduction chain (split -> RMSNorm -> RoPE) into one per-head kernel | 48 | 110.87 | 4.72 | 106.15 | 11.19 | T1 |
| R-03 | Chunked LM-head cross-entropy: fp32 upcast + log_softmax + NLL forward and backward + bf16 downcast | forward, backward | forward-backward, intra-module | fused softmax cross-entropy forward+backward (online logsumexp, grad = softmax - onehot) | 8 | 35.27 | 2.88 | 32.39 | 3.41 | T1 |
| R-04 | Router backward: aux-loss grad + top-k grad scatter + softmax backward + hi/mid/lo bf16 split of the gate grad | backward | cross-module | pointwise + row-reduction + per-row scatter into one kernel | 48 | 17.24 | 0.34 | 16.91 | 1.78 | T1 |
| R-05 | Router softmax + top-k score normalization + routing map + aux-loss statistics | forward | cross-module | row-softmax + small reductions into one or two kernels | 48 | 13.03 | 0.28 | 12.75 | 1.34 | T1 |
| R-06 | MoE dx top-k sum + ffn-input gradient sum + ffn RMSNorm backward + residual add + MXFP8 quantize of the wo gradient | backward | cross-module, communication-glue | reduction chain with RMSNorm backward and MXFP8 quantize epilogue | 48 | 20.95 | 12.15 | 8.79 | 0.93 | T1 |
| R-07 | Attention RMSNorm backward + residual-gradient add + MoE dy broadcast-and-scale of the layer below | backward | cross-module, cross-layer, communication-glue | row-reduction producer + broadcast epilogue | 48 | 26.53 | 18.70 | 7.82 | 0.82 | T1 |
| R-08 | Expert weight-grad bf16->fp32 cast + gradient clone | backward | intra-module, horizontal | cast fused with the final copy (write fp32 directly into param.grad) | 48 | 7.59 | 3.14 | 4.45 | 0.47 | T0 |
| R-09 | MoE weighted top-k sum + residual add + next-layer attention RMSNorm + MXFP8 quantize of wqkv input | forward | cross-module, cross-layer, communication-glue | reduction epilogue chain: top-k combine -> residual add -> RMSNorm -> MXFP8 quantize | 48 | 22.53 | 18.18 | 4.35 | 0.46 | T1 |
| R-10 | Colwise MXFP8 of the saved wqkv and wo inputs moved from backward into forward | forward, backward | forward-backward, gemm-adjacent | dual-orientation quantization at production time (save fp8 colwise instead of re-reading bf16) | 48 | 4.99 | 1.44 | 3.55 | 0.37 | T3 |
| R-11 | MoE routing metadata + forward/backward execution plans | forward, backward | communication-glue, forward-backward | single-pass routing compaction producing both plans | 48 | 3.97 | 0.47 | 3.50 | 0.37 | T2 |
| R-12 | Expert weight fp32->bf16 cast + MXFP8 weight preparation | forward | intra-module, horizontal | cast + block-quantize in one pass | 48 | 6.11 | 2.65 | 3.46 | 0.36 | T1 |
| R-13 | Attention output MXFP8 rowwise quantize + scale swizzle as attention epilogue | forward | gemm-adjacent | attention epilogue (quantize the output tile while it is in registers) | 48 | 3.38 | 0.96 | 2.42 | 0.26 | T3 |
| R-14 | FSDP reduce-scatter pack + post-reduce-scatter fp32 cast + gradient clone | backward | communication-glue, horizontal | pack + cast + copy collapsed (post-RS cast writes into param.grad) | 49 | 2.61 | 0.63 | 1.98 | 0.21 | T0 |
| R-15 | FSDP all-gather pack/unpack + wqkv/wo weight MXFP8 quantize | forward | communication-glue, horizontal | unpack fused with weight quantization | 50 | 2.65 | 0.76 | 1.89 | 0.20 | T1 |
| R-16 | Post-attention residual add + ffn RMSNorm + copy into the MoE dispatch buffer | forward | cross-module, communication-glue | add+RMSNorm with dual-destination store | 48 | 6.56 | 4.68 | 1.87 | 0.20 | T1 |
| R-17 | Embedding backward + reduce-scatter pack + post-reduce-scatter cast | backward | communication-glue, intra-module | segment-sum writing directly into the communication bucket | 1 | 0.86 | 0.14 | 0.72 | 0.08 | T2 |
| R-18 | LM-head weight-grad post-reduce-scatter cast + chunk accumulation | backward | communication-glue, intra-module | cast + accumulate in one pass | 8 | 0.74 | 0.43 | 0.31 | 0.03 | T0 |
| R-19 | Router gate weight-grad: hi/mid/lo partial GEMM sum + bf16 cast | backward | gemm-adjacent, horizontal | horizontal GEMM fusion with sum+cast epilogue | 48 | 0.28 | 0.00 | 0.27 | 0.03 | T3 |
| R-20 | LM-head dgrad chunk staging (zero-init + copy into the full gradient) | forward | gemm-adjacent | GEMM writes its output in place | 1 | 0.08 | 0.00 | 0.08 | 0.01 | T3 |
| R-21 | Embedding gather + layer-0 attention RMSNorm + MXFP8 quantize | forward | cross-module | gather prologue + RMSNorm + quantize | 1 | 0.11 | 0.07 | 0.04 | 0.00 | T1 |

Per-layer instances are uniform: for every per-layer region the slowest layer is within 1% of the median (except the first/last layer of the cross-layer regions, which have fewer members). Every in-scope memory-bound kernel belongs to exactly one region, so measured times do not overlap. `overlaps_with` marks design overlap only: R-10 (colwise MXFP8 at production time) would be emitted by the producers of R-09/R-13/R-21.

## 4. Region details

### R-01 RoPE backward + q/k RMSNorm backward + dQKV concat + MXFP8 quantize of dQKV
- What: From the attention gradients dq [32768,32,128] and dk/dv [32768,4,128] bf16: RoPE backward in fp32 (upcast, mul by cos/sin, neg, two zero-filled slice_backward scatters, gradient adds, downcast), q/k RMSNorm backward, concatenation into the interleaved [32768,4,10,128] dQKV and its rowwise+colwise MXFP8 quantization for the wqkv dgrad/wgrad GEMMs.
- Where: backward `layers.*.attention.k_norm` ['_fused_rms_norm_backward']; backward `layers.*.attention.q_norm` ['_fused_rms_norm_backward']; backward `layers.*.attention.qkv_linear` ['cat']; backward `layers.*.attention.qkv_linear.wqkv` ['torchao.mxfp8_quantize', 'torchao.triton_mx_block_rearrange']; backward `layers.*.attention.rope` ['_to_copy', 'add', 'mul', 'neg', 'slice_backward']. Source: torchtitan/models/common/rope.py:171, torchtitan/models/common/nn_modules.py:155, torchtitan/models/common/attention/attention.py:189, torchtitan/quantization/mxfp8/linear.py:129.
- Numbers: 48 instances x 30.0 kernels; 2557.6 us measured vs 143.4 us ideal per instance (992.7 MB external); gap 115.88 ms (12.22% of step). Shapes: in dq [32768,32,128] bf16, dk/dv [32768,4,128] bf16, q/k pre-norm, rstd, cos/sin; out dQKV fp8 e4m3 rowwise+colwise [32768,5120] + e8m0 scales; intermediates fp32 [32768,32,128].
- Why fusable: dq/dk -> fp32 -> (g*cos) + rotate_half^T(g*sin) -> bf16 -> rms_norm_backward(q/k) -> cat with dv into [T,4,10,128] -> mxfp8_quantize (row+col) -> scale swizzle. Each step is per (token, head) except the norm dweight and the colwise quantize (32-row blocks), which tile naturally over 32-token blocks.
- Blockers: Colwise MXFP8 needs 32-token tiles of all 5120 columns, so the kernel must tile tokens by 32 and all heads; dweight for q_norm/k_norm needs a cross-CTA reduction with fixed order. The FA backward postprocess (dq fp32 accum -> bf16) could also be folded in (T3), but is owned by the attention library.
- Tier T1: RMSNorm backward needs per-(token,head) dot products over head_dim and a token-wide dweight reduction ([128] per norm, two-stage in the current kernel); MXFP8 needs per-32-element amax. RoPE backward is pointwise (slice_backward + add is a pure interleave).
- Notes: slice_backward zero-fills 536 MB (q) per op before the scatter; two of them plus two fp32 adds per q per layer.

### R-02 QKV split + q/k RMSNorm + RoPE forward (fp32 round trip)
- What: After the wqkv projection [32768, 5120] bf16, q (32 heads), k and v (4 heads) are copied out of the GQA-interleaved layout, q and k are RMS-normalized over head_dim 128, then RoPE upcasts q/k to fp32, builds rotate_half with neg+cat, multiplies by cos/sin gathered per position, adds and downcasts to bf16.
- Where: forward `layers.*.attention.k_norm` ['_fused_rms_norm']; forward `layers.*.attention.q_norm` ['_fused_rms_norm']; forward `layers.*.attention.qkv_linear` ['clone']; forward `layers.*.attention.rope` ['_to_copy', 'add', 'cat', 'index', 'mul', 'neg']. Source: torchtitan/models/common/attention/attention.py:194-196, torchtitan/models/common/nn_modules.py:155, torchtitan/models/common/rope.py:170-171.
- Numbers: 48 instances x 20.0 kernels; 2309.8 us measured vs 98.3 us ideal per instance (680.3 MB external); gap 106.15 ms (11.19% of step). Shapes: in qkv [32768,4,10,128] bf16, cos/sin table [4096,256] fp32 indexed by position; out q [32768,32,128] bf16, k/v [32768,4,128] bf16, rstd fp32; intermediates [32768,32,128] fp32.
- Why fusable: Every op is per (token, head) and consumes only the previous op's output plus a per-position cos/sin row: clone(view of qkv) -> rms_norm -> to fp32 -> mul/neg/cat/mul/add -> to bf16 -> attention. No intermediate is needed by any other consumer, so the whole chain can run in registers; q and k share the same cos/sin rows (horizontal), and the cos/sin gather by position is identical in all 48 layers (cross-layer hoist).
- Blockers: Backward needs the pre-norm q/k and rstd: the fused kernel must keep the qkv GEMM output alive (instead of the contiguous clones) or write it, and backward must read it strided. Needs a custom kernel or Inductor region (not enabled in this run).
- Tier T1: RMSNorm reduces over head_dim=128 per (token, head); the fused kernel must reproduce the fused_rms_norm reduction order. RoPE math is pointwise in fp32 (x*cos + rotate_half(x)*sin, rounded to bf16 once) and is trivially reproducible.
- Notes: Second-largest gap. The fp32 intermediates (536 MB per q op) dominate; mul with broadcast cos/sin runs a non-vectorized elementwise kernel (~370 us per q mul). T3 variant: a wqkv GEMM epilogue whose N tile holds whole 128-dim heads could apply q/k RMSNorm + RoPE before the store, also removing the 335 MB GEMM-output round trip.

### R-03 Chunked LM-head cross-entropy: fp32 upcast + log_softmax + NLL forward and backward + bf16 downcast
- What: For each of 8 chunks of 4096 tokens, the bf16 logits [4096,151936] are upcast to fp32, log_softmax'ed and NLL-reduced; backward zero-fills and scatters the NLL grad, runs log_softmax backward and downcasts the logit gradient to bf16 for the two LM-head GEMMs.
- Where: backward `loss` ['_log_softmax_backward_data', '_to_copy', 'div', 'nll_loss_backward']; forward `loss` ['_log_softmax', '_to_copy', 'add', 'clone', 'div', 'new_zeros', 'nll_loss_forward', 'ones_like']. Source: torchtitan/experiments/graph_trainer/chunked_loss.py:107, torchtitan/distributed/local_compile.py:96.
- Numbers: 8 instances x 13.25 kernels; 4408.1 us measured vs 359.7 us ideal per instance (2489.4 MB external); gap 32.39 ms (3.41% of step). Shapes: logits [4096,151936] bf16 -> fp32 copies (2.49 GB each) -> grad logits [4096,151936] bf16.
- Why fusable: The loss is the last op, so the logit gradient depends only on the logits, labels and the loss scale: one kernel can read the bf16 logits once, compute logsumexp and the NLL, and write grad = (softmax - onehot) * scale in bf16 immediately (or in a second pass over the same logits), never materializing the fp32 copy, the fp32 log-probs saved for backward, the fp32 NLL grad or the fp32 softmax grad.
- Blockers: Forward and backward live in different phases of the joint graph; fusing them requires computing the logit gradient during forward (it only depends on the loss scale, known at trace time). A row of 151936 classes does not fit in shared memory, so a two-pass (online lse + gradient) design reads logits twice unless fused into the LM-head GEMM epilogue (T3, then reduction order changes).
- Tier T1: Row max and sum-exp over 151936 classes; to be bitwise the kernel must match cunn_SoftMaxForward/Backward's block reduction order and the fp32 rounding of log-probs before the NLL gather.
- Notes: Per chunk: 4.4 ms measured vs 0.36 ms ideal. A fused linear+CE (T3) would also remove the 1.24 GB logit write/read per chunk.

### R-04 Router backward: aux-loss grad + top-k grad scatter + softmax backward + hi/mid/lo bf16 split of the gate grad
- What: Backward of the router: aux-loss and score-normalization gradients on [32768,128]/[32768,8] fp32, a deterministic sort-based scatter_add of the 8 route gradients into [32768,128], softmax backward, and splitting the fp32 logit gradient into three bf16 parts (hi/mid/lo via casts, shifts and subtracts) concatenated to [32768,384] for the gate dgrad GEMM.
- Where: backward `layers.*.feed_forward.routed_experts` ['masked_fill']; backward `layers.*.feed_forward.router` ['_softmax_backward_data', 'add', 'div', 'mul', 'neg', 'new_zeros', 'scatter_add', 'sum']; backward `layers.*.feed_forward.router.aux_loss` ['add', 'div', 'gt', 'mul', 'neg', 'ones_like', 'scalar_tensor', 'sgn', 'sum', 'where']; backward `layers.*.feed_forward.router.gate` ['__lshift__', '_to_copy', 'cat', 'sub']. Source: torchtitan/models/common/moe/moe.py:257-292, torchtitan/models/common/moe/moe.py:702-705, torchtitan/models/common/linear/hi_mid_lo_linear.py:117.
- Numbers: 48 instances x 74.0 kernels; 359.2 us measured vs 7.0 us ideal per instance (48.6 MB external); gap 16.91 ms (1.78% of step). Shapes: [32768,128] fp32, [32768,8] fp32/int64, out [32768,384] bf16 and [384,2048] bf16.
- Why fusable: Every op is row-local over the 128 experts of a token: grad of scores -> scatter into the row -> add aux grad -> softmax backward -> split to bf16 hi/mid/lo -> concatenated row. One row-per-warp kernel produces the [T,384] GEMM operand directly.
- Blockers: The ATen deterministic scatter_add path (sort + index_put, 33 kernels) is chosen because deterministic algorithms are on; a fused kernel must know the indices are row-local.
- Tier T1: Softmax backward row sum over 128 and aux-loss row sums; the scatter_add has no collisions (the 8 expert IDs of a row are distinct), so its order does not matter. The hi/mid/lo split is exact bit manipulation.
- Notes: scatter_add alone is 161 us/layer for a 1 MB scatter.

### R-05 Router softmax + top-k score normalization + routing map + aux-loss statistics
- What: From router logits [32768,128] fp32: softmax, gather of the 8 routed experts' probabilities and their normalization, routing mask and tokens-per-expert count, aux-loss statistics (per-expert means, L1 norms, clamps, scalar products) and masking of padded tokens' expert IDs/scores, as ~45 small kernels.
- Where: forward `layers.*.feed_forward.routed_experts` ['masked_fill']; forward `layers.*.feed_forward.router` ['_softmax', 'add', 'add_', 'arange', 'bitwise_and', 'bitwise_not', 'div', 'gather', 'mul', 'remainder', 'scatter_', 'sum', 'zeros_like']; forward `layers.*.feed_forward.router.aux_loss` ['_to_copy', 'add_', 'bitwise_not', 'cat', 'clamp_min', 'clone', 'div', 'linalg_vector_norm', 'mul', 'reciprocal', 'sum']. Source: torchtitan/models/common/moe/moe.py:257-318, torchtitan/models/common/moe/moe.py:675-705, torchtitan/models/common/dist_moe/routed_experts.py:195-196.
- Numbers: 48 instances x 46.0 kernels; 271.5 us measured vs 5.8 us ideal per instance (40.2 MB external); gap 12.75 ms (1.34% of step). Shapes: [32768,128] fp32 probabilities, [32768,8] fp32/int64 routes, [128] counters, scalars.
- Why fusable: All ops read the same softmax output (or its per-row derivatives) and produce per-row results plus per-expert column sums; a row-tile kernel can compute softmax, top-k gather/normalize, the routing mask, and accumulate column partial sums, with a tiny second stage for the [128]-vector and scalar aux-loss math.
- Blockers: Column sums over all tokens need a deterministic two-stage reduction; one all_reduce (fake here) sits in the middle of the aux-loss statistics.
- Tier T1: Softmax row reduction over 128, column sums over 32768 tokens (tokens-per-expert, aux means) and L1 norms need fixed reduction orders.
- Notes: Expert IDs are (8t+k) mod 128 (balanced round-robin routing in this benchmark), so the routing pattern is identical in every layer; that cross-layer reuse is a benchmark artifact and is not counted.

### R-06 MoE dx top-k sum + ffn-input gradient sum + ffn RMSNorm backward + residual add + MXFP8 quantize of the wo gradient
- What: In backward, the 8 expert dx rows per token are summed ([32768,8,2048] -> [32768,2048]), added to the router-gate dx, passed through the ffn RMSNorm backward, added to the residual gradient, and the result is quantized rowwise+colwise to MXFP8 as the wo GEMM grad_output.
- Where: backward `layers.*` ['add']; backward `layers.*.attention.wo` ['torchao.mxfp8_quantize', 'torchao.triton_mx_block_rearrange']; backward `layers.*.feed_forward` ['add']; backward `layers.*.feed_forward.routed_experts` ['dist_moe.block_scaled_backward']; backward `layers.*.ffn_norm` ['_fused_rms_norm_backward']. Source: dist_moe/_blockscaled.py:4377 (scale_and_sum), torchtitan/models/common/nn_modules.py:155, torchtitan/quantization/mxfp8/linear.py:129.
- Numbers: 48 instances x 8.0 kernels; 436.4 us measured vs 253.2 us ideal per instance (1752.3 MB external); gap 8.79 ms (0.93% of step). Shapes: in dx rows [32768,8,2048] bf16, router dx [32768,2048] bf16, saved norm input, residual grad; out residual grad bf16, fp8 rowwise+colwise [32768,2048] + scales.
- Why fusable: scale_and_sum(dx rows) -> add(router dx) -> rms_norm_backward(ffn_norm) -> add(residual grad) -> mxfp8_quantize(row+col) -> scale swizzle; each intermediate has a single consumer except the residual grad (also needed by the next residual add), which stays an output. The router backward currently runs between the dx sum and the first add; it depends only on the score gradients, so the fused kernel can be scheduled after the router dgrad GEMM.
- Blockers: Crosses the dist_moe boundary (dx sum is inside block_scaled_backward); colwise quantization requires 32-token tiles.
- Tier T1: Top-k sum order, RMSNorm backward reductions, per-32 amax for both directions; the bf16 roundings of the two adds must be reproduced (round after each add).

### R-07 Attention RMSNorm backward + residual-gradient add + MoE dy broadcast-and-scale of the layer below
- What: In backward, layer N+1's attention RMSNorm backward and residual-gradient add produce the residual-stream gradient [32768,2048] bf16, which layer N's MoE immediately broadcasts to its 8 expert rows scaled by the routing scores ([32768,8,2048] bf16) while computing the score gradients; the final norm backward plays this role for the last layer.
- Where: backward `layers.*` ['add']; backward `layers.*.attention_norm` ['_fused_rms_norm_backward']; backward `layers.*.feed_forward.routed_experts` ['dist_moe.block_scaled_backward']; backward `norm` ['_fused_rms_norm_backward']. Source: torchtitan/models/common/nn_modules.py:155, dist_moe/_blockscaled.py:4377 (broadcast_and_scale).
- Numbers: 48 instances x 4.04 kernels; 552.7 us measured vs 389.7 us ideal per instance (2697.1 MB external); gap 7.82 ms (0.82% of step). Shapes: in wqkv dgrad [32768,2048] bf16, saved norm input, rstd, residual grad, saved expert rows [32768,8,2048] bf16, scores; out residual grad, dx rows [32768,8,2048] bf16 (dispatch send buffer), dscores [32768,8] fp32.
- Why fusable: rms_norm_backward(dx) -> add(residual grad) -> this sum is exactly grad_output_TD of the MoE backward below, whose first kernel broadcasts it to [T,8,D] and reduces <dy, x_k> per route. One row-local kernel can emit the residual grad, the scaled rows and dscores from registers.
- Blockers: Crosses the dist_moe op boundary and the layer boundary; the broadcast writes into the expert-parallel dispatch buffer. A T3 alternative folds the scale into the dgrad-dispatch grouped GEMM prologue (no 1 GiB dx write) but is owned by dist_moe.
- Tier T1: RMSNorm backward row reductions and dweight reduction; dscores = <dy, expert_row> dot over 2048 in the library's order.
- Notes: broadcast_and_scale alone is 410 us/layer (ideal 325 us standalone). Instances: 47 (attention_norm bwd of N+1, broadcast of N) + final-norm bwd with layer 47; layer-0 attention_norm bwd + add has no MoE below and is included as a tail.

### R-08 Expert weight-grad bf16->fp32 cast + gradient clone
- What: The expert weight gradients [16,1536,2048] and [16,2048,768] come out of the grouped wgrad GEMMs in bf16, are cast to fp32 to match the parameter dtype, and are cloned again into param.grad after backward.
- Where: backward `(outside modules)` ['clone']; backward `layers.*.feed_forward.routed_experts` ['_to_copy']. Source: torchtitan/experiments/graph_trainer/simple_fsdp.py (grad cast), torchtitan/experiments/graph_trainer/common_utils.py:322 (clone).
- Numbers: 48 instances x 4.0 kernels; 158.1 us measured vs 65.5 us ideal per instance (453.0 MB external); gap 4.45 ms (0.47% of step). Shapes: bf16 [16,1536,2048] + [16,2048,768] -> fp32 (302 MB) -> fp32 clone.
- Why fusable: wgrad (bf16) -> _to_copy fp32 -> returned as graph output -> grad.clone() into param.grad; the fp32 intermediate has no other consumer.
- Blockers: The clone happens outside the graph (grad accumulation helper clones to avoid aliasing graph-owned buffers); the graph would have to write into param.grad storage. A T3 variant has the wgrad GEMM write fp32 (rounded through bf16) directly.
- Tier T0: Pure cast and copy.

### R-09 MoE weighted top-k sum + residual add + next-layer attention RMSNorm + MXFP8 quantize of wqkv input
- What: The MoE output is the score-weighted sum of 8 expert rows per token ([32768,8,2048] bf16 -> [32768,2048]); it is added to the residual stream, and the next layer's attention RMSNorm and rowwise MXFP8 quantization (plus scale swizzle) of the wqkv input follow; the last layer feeds the final norm instead.
- Where: forward `layers.*` ['add']; forward `layers.*.attention.qkv_linear.wqkv` ['torchao.mxfp8_quantize', 'torchao.triton_mx_block_rearrange']; forward `layers.*.attention_norm` ['_fused_rms_norm']; forward `layers.*.feed_forward.routed_experts` ['dist_moe.block_scaled_forward']; forward `norm` ['_fused_rms_norm']. Source: dist_moe/_blockscaled.py:4377 (scale_and_sum), torchtitan/models/qwen3/model.py:73, torchtitan/models/common/nn_modules.py:155, torchtitan/quantization/mxfp8/linear.py:162-168.
- Numbers: 48 instances x 5.96 kernels; 469.3 us measured vs 378.7 us ideal per instance (2621.2 MB external); gap 4.35 ms (0.46% of step). Shapes: in expert rows [32768,8,2048] bf16 (combine receive buffer), scores [32768,8] fp32, residual [32768,2048] bf16; out residual, normed bf16, fp8 [32768,2048] + e8m0 [32768,64], rstd; the 1 GiB expert-row copy saved for backward.
- Why fusable: scale_and_sum writes y[T,D]; add(h, y) is the only consumer of y; the sum h' feeds attention_norm (and the next residual); the normed output feeds mxfp8_quantize whose scales feed the swizzle. All are row-local over D=2048, so one row-per-CTA kernel can produce h', normed, fp8 and swizzled scales while it streams the expert rows.
- Blockers: Crosses the dist_moe op boundary (scale_and_sum is inside dist_moe.block_scaled_forward) and the decoder-layer boundary. The 1 GiB copy of expert rows for backward is kept as an external write here; removing it would need the combine GEMM to write into a per-layer saved slot (symmetric-memory capacity).
- Tier T1: Top-k sum order (the library already preserves it in its fused RMSNorm+combine kernel), RMSNorm row reduction over 2048 and 32-element amax must be reproduced.
- Notes: Layer-0 attention_norm is in the embedding region; the last layer's instance feeds the final norm.
- Overlaps with: R-10

### R-10 Colwise MXFP8 of the saved wqkv and wo inputs moved from backward into forward
- What: In backward, the saved bf16 inputs of wqkv (normalized hidden [32768,2048]) and wo (attention output [32768,4096]) are re-read and quantized columnwise to MXFP8 for the weight-gradient GEMMs; the forward pass already produces the same tensors and quantizes them rowwise.
- Where: backward `layers.*.attention.qkv_linear.wqkv` ['torchao.mxfp8_quantize', 'torchao.triton_mx_block_rearrange']; backward `layers.*.attention.wo` ['torchao.mxfp8_quantize', 'torchao.triton_mx_block_rearrange']. Source: torchtitan/quantization/mxfp8/linear.py:129.
- Numbers: 48 instances x 4.0 kernels; 104.0 us measured vs 30.0 us ideal per instance (207.6 MB external); gap 3.55 ms (0.37% of step). Shapes: [32768,2048] and [32768,4096] bf16 -> fp8 colwise + e8m0 [2048,1024]/[4096,1024].
- Why fusable: The colwise quantize reads exactly the tensor produced in forward by attention_norm (wqkv input) and by attention (wo input); emitting it at production time removes the backward re-read.
- Blockers: Keeps fp8 colwise copies alive from forward to backward (+138 MB per layer for wo since the bf16 attention output is still needed by attention backward; the wqkv bf16 copy could be dropped, net -64 MB per layer). Colwise needs 32-token tiles in the producer.
- Tier T3: Colwise amax over 32 tokens; quantization of identical bf16 values gives identical fp8, wherever it runs. The wo half needs the attention kernel epilogue (T3); the wqkv half fits in the RMSNorm kernel (T1).
- Notes: Overlaps the attention-output quantize and the combine/RMSNorm region because the same producers would emit both orientations.
- Overlaps with: R-13, R-09, R-21

### R-11 MoE routing metadata + forward/backward execution plans
- What: Per layer, the expert IDs of all expert-parallel peers are gathered and turned into dispatch/combine gather and scatter index tables (four passes), a forward plan and a backward plan are built, and conditional activation copies are issued.
- Where: backward `layers.*.feed_forward.routed_experts` ['dist_moe.block_scaled_backward']; forward `layers.*.feed_forward.routed_experts` ['dist_moe.block_scaled_forward']. Source: dist_moe/_blockscaled.py:4377 (dist_dispatch_routing kernels, blockscaled plans).
- Numbers: 48 instances x 9.0 kernels; 82.7 us measured vs 9.8 us ideal per instance (68.2 MB external); gap 3.50 ms (0.37% of step). Shapes: expert IDs [8 x 32768 x 8] int64 gathered from peers -> int64 index tables [2099200] x 3, small plans.
- Why fusable: All kernels derive from the same expert IDs; the backward plan is a function of the forward routing, so it can be produced in forward.
- Blockers: Global compaction needs a grid-wide prefix (decoupled look-back) and reads peer buffers; owned by dist_moe.
- Tier T2: Prefix sums / compaction define row order of the dispatched tokens; any fused kernel must produce the same order.
- Notes: Ideal bytes are an estimate from the index-table sizes. Kernel times are measured; ideal bytes are estimated.

### R-12 Expert weight fp32->bf16 cast + MXFP8 weight preparation
- What: Each layer's local expert weights w13 [16,1536,2048] and w2 [16,2048,768] are cast from fp32 to bf16 and then quantized to MXFP8 with scales for both GEMM orientations.
- Where: forward `layers.*.feed_forward.routed_experts` ['_to_copy', 'dist_moe.prepare_mxfp8_weight']. Source: torchtitan/experiments/graph_trainer/simple_fsdp.py:268,366 (mixed-precision cast), dist_moe prepare_mxfp8_weight.
- Numbers: 48 instances x 4.0 kernels; 127.3 us measured vs 55.2 us ideal per instance (382.2 MB external); gap 3.46 ms (0.36% of step). Shapes: in fp32 [16,1536,2048] + [16,2048,768]; out fp8 e4m3 + e8m0 scales (2 orientations).
- Why fusable: _to_copy(fp32->bf16) has a single consumer, prepare_mxfp8_weight; w13 and w2 are independent siblings.
- Blockers: The cast is emitted by the FSDP mixed-precision wrapper and the quantize by dist_moe; could also move into the optimizer step (write the fp8 copy when updating), which crosses the step boundary.
- Tier T1: Per-32 amax and scale computation; the bf16 rounding before quantization must be reproduced (cast in registers).

### R-13 Attention output MXFP8 rowwise quantize + scale swizzle as attention epilogue
- What: The attention output [32768,32,128] bf16 is re-read to quantize it rowwise to MXFP8 and swizzle the scales for the wo GEMM.
- Where: forward `layers.*.attention.wo` ['torchao.mxfp8_quantize', 'torchao.triton_mx_block_rearrange']. Source: torchtitan/quantization/mxfp8/linear.py:162-168.
- Numbers: 48 instances x 2.0 kernels; 70.4 us measured vs 20.0 us ideal per instance (138.4 MB external); gap 2.42 ms (0.26% of step). Shapes: [32768,4096] bf16 -> fp8 e4m3 [32768,4096] + e8m0 [32768,128] (swizzled).
- Why fusable: mxfp8_quantize is the only consumer of the attention output in forward besides saving it for backward; scale swizzle consumes only the quantize's scales.
- Blockers: Needs a FlashAttention (CuTe DSL) epilogue change in an external library.
- Tier T3: Rowwise 32-element blocks lie within one head's 128-dim output row, so the amax is local to an attention output tile; values must be the bf16-rounded outputs.
- Notes: Standalone (no attention change) the gap would only be ~11 us/layer.
- Overlaps with: R-10

### R-14 FSDP reduce-scatter pack + post-reduce-scatter fp32 cast + gradient clone
- What: Per layer, the bf16 weight gradients of the dense parameters are packed into the reduce-scatter bucket; after the reduce-scatter each shard is cast to fp32, and after backward every gradient is cloned into param.grad.
- Where: backward `(outside modules)` ['clone']; backward `layers.*.attention.qkv_linear.wqkv` ['_to_copy', 'bucketing._pre_bucket_reduce_scatter']; backward `norm` ['_to_copy', 'bucketing._pre_bucket_reduce_scatter']. Source: torchtitan/experiments/graph_trainer/simple_fsdp.py:366, torchtitan/experiments/graph_trainer/common_utils.py:322.
- Numbers: 49 instances x 15.8 kernels; 53.3 us measured vs 12.9 us ideal per instance (89.1 MB external); gap 1.98 ms (0.21% of step). Shapes: 19.1M bf16 bucket per layer -> 2.39M bf16 shard -> fp32 shard -> fp32 clone; plus the embedding and LM-head fp32 grad clones (155 MB each).
- Why fusable: _pre_bucket_reduce_scatter only concatenates the wgrad outputs; the post-RS casts (7 per layer) are siblings over one buffer; the clone copies the cast output. A T3 variant writes the wgrad GEMM outputs straight into the bucket.
- Blockers: The clone runs outside the graph; the graph would have to receive param.grad storage as an output buffer.
- Tier T0: Copies and casts only.

### R-15 FSDP all-gather pack/unpack + wqkv/wo weight MXFP8 quantize
- What: Per layer, the fp32 parameter shards (wqkv, wo, router gate, norms) are cast to bf16 and packed into the all-gather buffer; after the all-gather, the gathered [8, chunk] buffer is copied out per parameter and wqkv [5120,2048] / wo [2048,4096] are quantized to MXFP8 with 32x32 block scales.
- Where: forward `layers.*.attention.qkv_linear.wqkv` ['bucketing._pre_bucket_all_gather', 'split_with_sizes_copy', 'torchao.triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale']; forward `layers.*.attention.wo` ['torchao.triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale']; forward `norm` ['bucketing._pre_bucket_all_gather', 'split_with_sizes_copy']; forward `tok_embeddings` ['bucketing._pre_bucket_all_gather']. Source: torchtitan/experiments/graph_trainer/simple_fsdp.py:268,366.
- Numbers: 50 instances x 4.88 kernels; 52.9 us measured vs 15.1 us ideal per instance (104.5 MB external); gap 1.89 ms (0.20% of step). Shapes: per layer: 2.39M fp32 per shard -> 19.1M bf16 gathered -> fp8 [5120,2048], [2048,4096]; plus the embedding and norm/LM-head buckets.
- Why fusable: split_with_sizes_copy writes per-parameter bf16 tensors whose only consumer for wqkv/wo is the 32x32 quantizer; a fused unpack can quantize straight from the gathered layout.
- Blockers: The pre-all-gather cast must stay a separate kernel (the all-gather sits between); the norm/LM-head bucket's copy-out cannot be skipped because each rank's chunk starts with a 256-element norm shard that misaligns the LM-head rows.
- Tier T1: 32x32 block amax for the weight quantization; pack/unpack and casts are exact.
- Notes: The norm/LM-head bucket instance is near roofline (223 us vs 214 us).

### R-16 Post-attention residual add + ffn RMSNorm + copy into the MoE dispatch buffer
- What: The wo projection output is added to the residual stream, RMS-normalized by ffn_norm, and the normalized [32768,2048] bf16 activations plus routing IDs are copied into the expert-parallel dispatch buffer.
- Where: forward `layers.*` ['add']; forward `layers.*.feed_forward.routed_experts` ['dist_moe.block_scaled_forward']; forward `layers.*.ffn_norm` ['_fused_rms_norm']. Source: torchtitan/models/qwen3/model.py:61-65, torchtitan/models/common/nn_modules.py:155, dist_moe/_blockscaled.py:4377 (copy_routing_and_dispatch).
- Numbers: 48 instances x 3.0 kernels; 136.6 us measured vs 97.6 us ideal per instance (675.4 MB external); gap 1.87 ms (0.20% of step). Shapes: in residual + wo output [32768,2048] bf16; out residual, normed bf16 (saved), dispatch copy bf16, rstd, routing IDs [32768,8] int64.
- Why fusable: add -> rms_norm -> (router gate GEMM, saved for wgrad, and dispatch copy). The dispatch copy is a pure copy of the norm output, so the norm kernel can store its output to both destinations.
- Blockers: The dispatch buffer is dist_moe symmetric memory published inside the dist_moe op; the norm would need the buffer handle, and the routing-ID part of the publish (which depends on the router that runs after the norm) would move to the router kernel. The residual add could alternatively go into the wo GEMM epilogue (T3).
- Tier T1: RMSNorm row reduction over 2048 must match the fused_rms_norm kernel.

### R-17 Embedding backward + reduce-scatter pack + post-reduce-scatter cast
- What: The embedding gradient [151936,2048] bf16 is built by a deterministic sort-and-segment-sum over 32768 token IDs (with a full zero-fill), then copied into the reduce-scatter bucket and the shard cast to fp32.
- Where: backward `tok_embeddings` ['_to_copy', 'bucketing._pre_bucket_reduce_scatter', 'embedding_dense_backward']. Source: torchtitan/models/common/embedding.py:41.
- Numbers: 1 instances x 24.0 kernels; 865.0 us measured vs 143.1 us ideal per instance (990.2 MB external); gap 0.72 ms (0.08% of step). Shapes: grad [32768,2048] bf16 + IDs -> [151936,2048] bf16 (622 MB) -> bucket copy -> [18992,2048] fp32.
- Why fusable: The bucket pack for a single parameter is a pure copy of the embedding gradient.
- Blockers: Library sort kernels; the bucket is created by the bucketing pass.
- Tier T2: Duplicate token IDs are summed in sorted order; the fused kernel must keep the segment order.

### R-18 LM-head weight-grad post-reduce-scatter cast + chunk accumulation
- What: Each of the 8 loss chunks reduce-scatters its LM-head weight gradient [151936,2048] bf16 to a [18992,2048] shard, casts it to fp32 and adds it into an fp32 accumulator.
- Where: backward `loss` ['_to_copy', 'add_']. Source: torchtitan/experiments/graph_trainer/chunked_loss.py:107.
- Numbers: 8 instances x 1.88 kernels; 92.3 us measured vs 53.4 us ideal per instance (369.5 MB external); gap 0.31 ms (0.03% of step). Shapes: [18992,2048] bf16 -> fp32 accumulate.
- Why fusable: _to_copy has a single consumer, the add_ into the accumulator.
- Blockers: None beyond codegen; accumulating before the reduce-scatter would change numerics.
- Tier T0: Same cast and same left-to-right accumulation order.

### R-19 Router gate weight-grad: hi/mid/lo partial GEMM sum + bf16 cast
- What: The router gate weight gradient [128,2048] is computed as three fp32 GEMMs (one per bf16 split part of the logit gradient) against the same normalized input; their outputs are summed with two in-place adds and cast to bf16.
- Where: backward `layers.*.feed_forward.router.gate` ['_to_copy', 'add_']. Source: torchtitan/models/common/linear/hi_mid_lo_linear.py:117.
- Numbers: 48 instances x 3.0 kernels; 5.8 us measured vs 0.1 us ideal per instance (0.5 MB external); gap 0.27 ms (0.03% of step). Shapes: 3 x [128,2048] fp32 partials -> [128,2048] bf16.
- Why fusable: The three GEMMs share the B operand (normed input [32768,2048]) and differ only in A; their outputs are consumed only by the adds and the cast.
- Blockers: Needs a custom GEMM epilogue (cuBLAS split-K + separate reduce today).
- Tier T3: The fp32 sum order ((hi+mid)+lo) and the split-K reduction inside each GEMM must be kept; a single GEMM with M=384 and a fixed-order epilogue sum preserves it.
- Notes: Tiny; listed for completeness.

### R-20 LM-head dgrad chunk staging (zero-init + copy into the full gradient)
- What: The hidden-state gradient buffer [32768,2048] bf16 is zero-initialized and each chunk's LM-head dgrad [4096,2048] is copied into its slice.
- Where: forward `loss` ['copy_', 'zeros_like']. Source: torchtitan/experiments/graph_trainer/chunked_loss.py:107.
- Numbers: 1 instances x 9.0 kernels; 77.8 us measured vs 0.0 us ideal per instance (0.0 MB external); gap 0.08 ms (0.01% of step). Shapes: [32768,2048] bf16, 8 x [4096,2048].
- Why fusable: Each copy_ consumes one dgrad GEMM output; the chunks tile the buffer completely, so the zero-fill is dead.
- Blockers: None (mm supports out= into a contiguous row slice).
- Tier T3: Same GEMM, only the output pointer changes.
- Notes: Tiny.

### R-21 Embedding gather + layer-0 attention RMSNorm + MXFP8 quantize
- What: The token embedding gather [32768,2048] bf16 feeds the residual stream and layer 0's attention RMSNorm and rowwise MXFP8 quantization of the wqkv input.
- Where: forward `layers.*.attention.qkv_linear.wqkv` ['torchao.mxfp8_quantize', 'torchao.triton_mx_block_rearrange']; forward `layers.*.attention_norm` ['_fused_rms_norm']; forward `tok_embeddings` ['embedding']. Source: torchtitan/models/common/embedding.py:41, torchtitan/models/common/nn_modules.py:155, torchtitan/quantization/mxfp8/linear.py:162-168.
- Numbers: 1 instances x 4.0 kernels; 110.9 us measured vs 68.2 us ideal per instance (472.3 MB external); gap 0.04 ms (0.00% of step). Shapes: [32768,2048] bf16.
- Why fusable: embedding output -> attention_norm -> quantize; the embedding output is also kept as residual.
- Blockers: None specific.
- Tier T1: RMSNorm row reduction and per-32 amax.
- Overlaps with: R-10

## 5. No opportunity

| what | ms | reason |
|---|---|---|
| Fake-process-group initialization memsets of the 1 GiB expert-parallel peer-scatter buffers inside the MoE forward (1/layer) and backward (2/layer) | 22.94 | Emulation artifact: dist_moe zero-fills a symmetric buffer only when it is a fake symmetric memory (rows owned by virtual peers have no writer); native runs skip it. Counted as communication, not a fusion target. |
| c10d all-gather / reduce-scatter / all-reduce stand-in copies (Memcpy DtoD under collective ops) | 3.24 | Communication stand-ins; excluded from compute and regions by definition. |
| FlashAttention-internal memory-bound kernels (backward preprocess dO.O, dq/dk/dv postprocess, accumulator memsets) | 17.46 | Inside the attention library (counted as attention). The dq postprocess (fp32 accum -> bf16) feeding the RoPE backward's bf16 -> fp32 upcast is a T3 extension of the RoPE-backward region, noted there. |
| Grouped-GEMM prepare kernels, cuBLAS split-K reduce, and the skipped MoE forward-recompute GEMM launches | 2.66 | Part of the GEMM implementations (counted as gemm); launch-latency sized, no memory-bound data movement to fuse. |

All non-GEMM, non-attention, non-communication kernels in the forward and backward are assigned to a region (409.1 ms of glue, 0 ms unassigned); the rows above are memory-bound time that sits inside other categories, listed for completeness. Context for the GEMM budget: the Dist-MoE grouped GEMMs (with fused dispatch/combine) run at 1.1-4.5x their MXFP8 flop ideal (fprop 87.9 ms vs 32.9 ms, dgrad 108.0 ms vs 32.9 ms, wgrad 36.0 ms vs 32.8 ms); the dense MXFP8 `_scaled_mm_v2` GEMMs are at their ideal; these are kernel-efficiency, not fusion, questions.

## 6. Observations about working from these artifacts

**What was easy**
- The per-node `_RecordFunctionFast` labels made attribution nearly mechanical: one CPU thread, one GPU stream, and kernel -> runtime launch (correlation id) -> innermost `<node>: <op>` range gave the graph node for 99.4% of kernel time. The node dump then supplied target, shapes/strides/dtypes, `custom.module_fqn` and `autograd_backward`, so module, layer and phase came for free for almost every kernel.
- `custom.module_fqn` is present on backward nodes too and names the forward module being differentiated (e.g. `_fused_rms_norm_backward` -> `layers.46.ffn_norm`), so phase/module for backward ops did not need seq_nr matching.
- Graph order is execution order (single stream), so adjacency of producers/consumers could be read directly from the node list (`out/layer_view.py`).
- The `moe_*` user annotations inside the Dist-MoE ops split their 15-18 kernels into preprocess / routing / plan / main / postprocess.

**What was hard / needed workarounds**
- `nn_module_stack` is null everywhere; only `custom.module_fqn` exists. Gradient-accumulation nodes ("Gradient addition node due to multiple use of tensor") have no module and their seq_nr matches no forward node. I assigned them the longest common module prefix of their producers, and mapped bf16 [32768,2048] sums next to a norm backward to the decoder block (residual connection).
- The Dist-MoE ops are opaque graph nodes; to know what `_triton_scale_and_sum`, `_triton_broadcast_and_scale`, `_triton_copy_routing_and_dispatch` and the 1 GiB memsets move, I had to read the installed dist_moe source. Bytes are only recorded for memcpy/memset events, so kernel bytes were reconstructed from node shapes and library docstrings.
- 0.6% of forward+backward time ran outside the graph (gradient clones). Those kernels carry no node label; I found them through the Python stack (`accumulate_param_grads_`) and identified the parameter of each clone only by its byte size.
- The profiled step's GPU span is 57 s for 0.95 s of GPU work, so idle time and launch-gap analysis are not meaningful from this trace.
- FlashAttention's internal memsets/pre/postprocess kernels can only be classed as attention; their interaction with RoPE backward (fp32 dq accumulator -> bf16 -> fp32 again) is visible only by reading kernel names in order.

**Missing evidence**
- No hardware counters or per-kernel DRAM bytes; achieved bandwidth is inferred from shapes. No `cu_seqlens` values, so attention flops (and an attention roofline) could not be computed.
- No record of which graph outputs alias `param.grad`, so the post-graph clone's necessity had to be read from `common_utils.py`.
- The low-entropy fake-PG data makes the LM-head GEMMs run faster than the measured BF16 peak (ratio 0.9 in `node_attribution.csv`); GEMM ratios are therefore optimistic.

**How phase and module were identified for backward ops**
- Phase = node `autograd_backward`; module = node `custom.module_fqn` (already the forward module); for gradient additions without metadata, the producers' common module prefix (residual-stream sums -> `layers.N`; the ffn-input grad sum of router dx + expert dx -> `layers.N.feed_forward`; RoPE grad sums -> `layers.N.attention.rope`). Kernels inside `dist_moe.block_scaled_backward` were split by the `moe_*` annotation and kernel name. Out-of-graph clones are labeled phase backward, module "".

**False leads discarded**
- The 144 one-GiB memsets inside the MoE ops (22.9 ms) first looked like zero-fills to fold into the combine; dist_moe zero-fills those buffers only for fake symmetric memory, so they are an emulation artifact (moved to communication / no_opportunity).
- The forward `_triton_scale_and_sum` (326 us) looked 2x off roofline for 1 GiB in / 128 MB out, but it also writes the 1 GiB expert-row copy saved for backward, which puts it at the measured copy bandwidth; the opportunity is only in what follows it (R-09).
- Routing is identical in every layer (balanced round-robin), which suggests computing routing metadata once per step; that is a benchmark artifact and not counted.
- The router-backward `scatter_add` (33 kernels, sort-based because determinism is on) looked order-sensitive (T2); the 8 expert ids in a row are distinct, so there are no collisions and a per-row scatter is order-free (T1).
- An early "gradient clone" region was merged into the cast regions that produce the cloned tensors (R-08, R-14); early "MoE zero-fill + combine" regions were dropped with the memset finding.
- `regional_inductor_pass` changed nothing (no region annotations), so the pre-Inductor graph is not a source of differences.

**Kinds of opportunity I may have missed**
- Fusion inside library kernels: Dist-MoE grouped GEMMs are 1.1-4.5x off their flop ideal and FlashAttention backward is 113.7 ms; I did not analyze their internals.
- Further T3 variants not listed as separate regions: q/k RMSNorm + RoPE in the wqkv GEMM epilogue (a 256-wide N tile holds whole 128-dim heads, saving the 335 MB GEMM output round trip), router softmax/top-k in the router GEMM epilogue (N=128 fits one tile), the post-attention residual add in the wo GEMM epilogue, and a fused linear + cross-entropy that never materializes the logits.
- Cross-step fusion (optimizer writing the bf16/MXFP8 weight copies for the next forward) and recompute-vs-save trade-offs, which this memory_policy=none run does not exercise.
- Real-communication glue and overlap effects (NVLink dispatch/combine, NCCL copies) that the fake process group hides.

**Effort**
- Wall time: started 16:48, all deliverables written by 17:12 (about 24 minutes of CPU-only analysis; no GPU use).
- Scripts (all in `out/`): `attribute.py` (trace -> kernel/launch/op stacks), `join.py` (+ node metadata, module inference), `assign.py` (categories and region rules), `regions.py` (bytes formulas, ideal/gap, regions.json), `attribution.py` (per-op and per-kernel CSVs), `layer_view.py` (per-layer op listing with timings), `log_region.py` (regions_log.jsonl appender), `make_report.py` (this report).

## 7. Reproduction

All scripts are CPU-only and read only the artifacts:

```bash
cd /home/bahuang/local/step_analysis/qwen3_30b_graph
python out/attribute.py    # trace -> out/kernels.pkl (kernel -> launch -> CPU op stack, node label, moe_* annotation)
python out/join.py         # + node metadata and module inference -> out/joined.pkl
python out/assign.py       # category + region per kernel -> out/assigned.pkl (prints totals, unassigned glue)
python out/regions.py      # -> out/regions.json, out/regions_table.md, out/step_budget_details.json
python out/attribution.py  # -> out/node_attribution.csv (per op instance), out/kernel_attribution.csv (per GPU event)
python out/make_report.py  # -> out/report.md
python out/layer_view.py 5 F   # optional: layer 5 forward ops with measured time (B for backward, none for non-layer)
```

`out/regions_log.jsonl` records each region when it was first identified (titles there are the working titles; some were later merged or dropped as described in section 6).
