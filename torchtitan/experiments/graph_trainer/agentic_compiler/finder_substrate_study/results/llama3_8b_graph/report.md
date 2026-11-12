# Kernel-fusion opportunities in one Llama 3 8B training step (GraphTrainer, FSDP 8, fake PG)

## 1. Setup

- Program: Llama 3 8B (`llama3_8b(seq_len=8192)`), BF16 compute with fp32 master weights and fp32 gradient reduce-scatter, 32 decoder blocks with fused `wqkv` [6144,4096] and fused gate/up `w13` [2,14336,4096] projections, GQA (32 q heads, 8 kv heads, head dim 128), document-masked varlen flash attention (CuTe DSL), RMSNorm (quack CuTe DSL), RoPE as an fp32 complex multiply, vocabulary 128256 with an untied `lm_head` and a chunked cross-entropy (8 chunks of 1024 tokens, each chunk's backward run inside the loss forward).
- Execution: GraphTrainer traced joint forward+backward FX graph (6907 nodes), `memory_policy="none"` (every activation saved, unsharded weights kept from forward to backward), regional Inductor with no regions (every op runs as its eager ATen/library kernel), custom codegen wrapping each node in a labeled record function. FSDP 8 via SimpleFSDP with per-block bucketing (6 parameters per block; `tok_embeddings` alone; `norm`+`lm_head` together), emulated by a fake process group on one GPU (rank 0's computation; collectives are local copies). One 8192-token microbatch, deterministic algorithms, no activation recompute, no CUDA graphs. Profiled step 10 (trace `ProfilerStep#9`).
- GPU: NVIDIA GB300. Roofline: HBM 6921.3 GB/s (copy), BF16 GEMM peak 1929.5 TFLOP/s (max of the measured shapes); ideal = max(bytes / HBM, flops / peak).
- Attention flops cannot be computed exactly (document boundaries are data-dependent and not recorded); attention ideals use the full-causal upper bound and attention is not a region target.
- GEMMs measured *below* the random-data peak in aggregate (185.4 ms measured vs 191.2 ms ideal), consistent with the low-entropy data note in the hardware profile; GEMM-adjacent (T3) gaps are therefore conservative.

## 2. Step budget

| category | GPU ms | % of busy |
|---|---:|---:|
| gemm | 185.358 | 54.0% |
| attention | 23.212 | 6.8% |
| communication | 7.270 | 2.1% |
| memory_bound_glue | 118.844 | 34.6% |
| optimizer | 6.100 | 1.8% |
| other | 2.392 | 0.7% |
| **total busy** | **343.170** | 100% |

- Step GPU window (`ProfilerStep#9` on the GPU): 7539.7 ms; GPU busy 343.2 ms; **GPU idle 7196.5 ms (95.4%)**.
- Forward+backward span: 336.259 ms busy (forward 107.917 ms, backward 226.768 ms in the graph, plus 1.581 ms post-backward gradient clone outside the graph); idle inside that span 7120.6 ms.
- Attribution coverage: **99.53%** of forward+backward kernel time maps to a graph node through its labeled record function; the remaining 0.47% (1.581 ms, 195 device-to-device copies) is the post-backward `.grad` clone (`accumulate_param_grads_`), attributed by its Python stack, so coverage is 100%.
- `optimizer` = fused AdamW (4.24 ms) + gradient-norm clipping (`_foreach_norm`, `_foreach_mul_`, 1.86 ms). `other` = post-backward `.grad` clone (1.58 ms) + attention-metadata construction before the forward (deterministic `index_put_`, 0.64 ms) and small bookkeeping kernels.
- Communication is the fake process group's local-copy emulation (all-gather = 7 device copies per bucket; reduce-scatter = 1 copy); it is not representative of real network time.
- 17.1 ms of kernels at the start of the trace (before the step's GPU annotation, no CPU correlation in the window) are the tail of step 9 and are excluded.
- The idle time is host-side: the median graph node spends 0.64 ms on the CPU under the profiler (Python stack capture on, 187k Python events), so the GPU starves. The logged throughput at step 10 (6469 tok/s, apparently averaged over steps 2-10) implies unprofiled steps are much shorter than this profiled one; kernel durations are unaffected, but no conclusion about overlap can be drawn from this timeline.

## 3. Ranked regions

Ranked by `gap_ms_total` (= (measured - ideal) x instances). `% of step` is relative to GPU-busy time (343.2 ms). Several regions are alternative or superset formulations of the same glue (see `overlaps`); the non-overlapping primary set R-04, R-05, R-07, R-09, R-13, R-14, R-15, R-16, R-17, R-19, R-20, R-21, R-23, R-24, R-25, R-26, R-27, R-28 sums to **89.2 ms of gap (26.0% of GPU-busy time)** over 124.7 ms of measured time.

| id | title | phases | scope | kind | inst | measured ms | ideal ms | gap ms | % of step | tier | overlaps |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| R-01 | SwiGLU forward+backward pair: recompute silu in backward instead of saving it | for+bac | forward-backward, intra-module | forward/backward co-design (activation not saved) | 32 | 44.754 | 8.688 | 36.066 | 10.51 | T0 | R-03,R-05,R-06,R-07 |
| R-02 | Weight-grad GEMM epilogues writing fp32 directly into the reduce-scatter bucket | bac | gemm-adjacent, communication-glue | GEMM epilogue with scattered fp32 store | 32 | 90.034 | 59.263 | 30.771 | 8.97 | T3 | R-04 |
| R-03 | w2 input-gradient GEMM with SwiGLU-backward epilogue | bac | gemm-adjacent, intra-module | GEMM epilogue | 32 | 41.736 | 15.955 | 25.780 | 7.51 | T3 | R-01,R-05 |
| R-04 | FSDP reduce-scatter pack: weight-grad fp32 casts, w13 re-layout and bucket copy | bac | communication-glue, horizontal, intra-module | cast + permute + pack (multi-tensor copy) | 32 | 31.737 | 6.051 | 25.686 | 7.49 | T2 | R-02 |
| R-05 | SwiGLU backward: two muls + silu_backward + stack | bac | intra-module, horizontal | elementwise chain + stack into strided output | 32 | 26.884 | 6.516 | 20.368 | 5.93 | T0 | R-01,R-03 |
| R-06 | w13 GEMM with SwiGLU epilogue | for | gemm-adjacent, intra-module | GEMM epilogue | 32 | 46.570 | 31.911 | 14.659 | 4.27 | T3 | R-01,R-07 |
| R-07 | SwiGLU forward: silu + mul on the strided gate/up halves of the w13 output | for | intra-module | elementwise chain | 32 | 17.870 | 4.344 | 13.526 | 3.94 | T0 | R-01,R-06 |
| R-08 | lm_head GEMM + cross-entropy + input-gradient GEMM per chunk (logits never written) | for+bac | gemm-adjacent, forward-backward | fused linear + cross-entropy (online softmax) | 8 | 16.872 | 8.921 | 7.950 | 2.32 | T3 | R-09,R-26 |
| R-09 | Chunked cross-entropy: logits cast, log_softmax, NLL and their backward per 1024-token chunk | for+bac | intra-module, forward-backward | row-reduction softmax/CE with fused gradient | 8 | 8.284 | 0.607 | 7.677 | 2.24 | T1 | R-08 |
| R-10 | Weight unpack folded into GEMM operand loads (consume the all-gather buffer in place) | for | communication-glue, gemm-adjacent | GEMM prologue (multi-base-pointer operand) | 33 | 6.936 | 0.000 | 6.936 | 2.02 | T3 | R-17 |
| R-11 | Attention-backward postprocess fused with RoPE backward and d_qkv concat | bac | gemm-adjacent, cross-module | attention epilogue | 32 | 7.293 | 1.416 | 5.877 | 1.71 | T3 | R-14,R-22 |
| R-12 | wqkv GEMM epilogue emitting rotated q/k and contiguous v | for | gemm-adjacent, cross-module | GEMM epilogue | 32 | 12.241 | 6.838 | 5.403 | 1.57 | T3 | R-13,R-22 |
| R-13 | RoPE forward: q/k/v contiguous copies + fp32 complex rotation of q and k | for | intra-module, cross-module, horizontal | strided split + pointwise rotation | 32 | 5.719 | 0.951 | 4.769 | 1.39 | T0 | R-12,R-22 |
| R-14 | RoPE backward: inverse rotation of dq/dk + concat into d_qkv | bac | intra-module, cross-module, horizontal | pointwise rotation + concat | 32 | 5.469 | 0.950 | 4.519 | 1.32 | T0 | R-11,R-22 |
| R-15 | lm_head weight-gradient fp32 cast fused into the per-chunk wgrad GEMM epilogue | bac | gemm-adjacent, communication-glue | GEMM epilogue (round-to-bf16 then widen) | 8 | 7.520 | 4.461 | 3.059 | 0.89 | T3 | - |
| R-16 | RMSNorm backward + residual-gradient accumulation add | bac | cross-module, cross-layer | row-reduction + pointwise epilogue | 64 | 5.436 | 2.502 | 2.934 | 0.85 | T1 | - |
| R-17 | FSDP all-gather unpack: split_with_sizes_copy + w13 re-layout cat | for | communication-glue, intra-module, horizontal | unpack + permute (multi-tensor copy) | 32 | 6.627 | 4.034 | 2.593 | 0.76 | T2 | R-10 |
| R-18 | Residual add in the wo / w2 GEMM epilogues | for | gemm-adjacent, cross-module | GEMM epilogue | 64 | 22.225 | 20.514 | 1.711 | 0.50 | T3 | R-23,R-24 |
| R-19 | Token-embedding backward: zero-fill, deterministic scatter-add, fp32 cast and bucket copy | bac | communication-glue, intra-module | segmented scatter-add + cast + pack | 1 | 1.845 | 0.313 | 1.531 | 0.45 | T2 | - |
| R-20 | Attention-backward accumulator zero-fill folded into the preprocess kernel | bac | horizontal, gemm-adjacent | horizontal (memset + preprocess) | 32 | 2.158 | 1.104 | 1.054 | 0.31 | T3 | - |
| R-21 | lm_head per-chunk reduce-scattered gradient accumulation folded into the reduce-scatter | bac | communication-glue | collective reduction epilogue | 7 | 0.770 | 0.266 | 0.504 | 0.15 | T1 | - |
| R-22 | Rotary table gather and conjugate computed once per step instead of per layer | for+bac | cross-layer, horizontal | common-subexpression elimination across layers | 1 | 0.390 | 0.002 | 0.388 | 0.11 | T0 | R-11,R-12,R-13,R-14 |
| R-23 | Attention residual add + ffn_norm | for | cross-module | pointwise prologue + row reduction (add+RMSNorm) | 32 | 1.603 | 1.241 | 0.362 | 0.10 | T1 | R-18 |
| R-24 | FFN residual add + next layer's attention_norm (final norm after layer 31) | for | cross-layer, cross-module | pointwise prologue + row reduction (add+RMSNorm) | 32 | 1.580 | 1.241 | 0.338 | 0.10 | T1 | R-18 |
| R-25 | FSDP all-gather input packs (fp32 shard -> bf16 slice) batched into one launch per step | for | communication-glue, cross-layer, horizontal | horizontal multi-tensor cast across buckets | 1 | 1.113 | 0.870 | 0.243 | 0.07 | T0 | - |
| R-26 | lm_head input-gradient written in place into the hidden-state gradient buffer | for | gemm-adjacent, intra-module | GEMM output to strided destination | 1 | 0.047 | 0.000 | 0.047 | 0.01 | T3 | R-08 |
| R-27 | Token-embedding gather + layer-0 attention_norm | for | cross-module, cross-layer | gather + row reduction | 1 | 0.037 | 0.029 | 0.008 | 0.00 | T1 | - |
| R-28 | Final norm weight-gradient fp32 cast + reduce-scatter bucket copy | bac | communication-glue, intra-module | cast + pack | 1 | 0.005 | 0.000 | 0.005 | 0.00 | T0 | - |

## 4. Region details

### R-01 SwiGLU forward+backward pair: recompute silu in backward instead of saving it (T0, measured)

- What: With fused SwiGLU forward and backward kernels, the forward no longer needs to write silu(gate) [8192,14336] bf16 and the backward no longer reads it, since the backward kernel already reads gate; this pairs the forward and backward of each feed_forward.
- Where: forward layers.*.feed_forward [all] {silu, mul}; backward layers.*.feed_forward [all] {mul, silu_backward, stack}. Source: `torchtitan/models/common/feed_forward.py:75`. 32 instances/step, 6.0 kernels/instance. Shapes: forward: bf16 [8192,2,14336] -> [8192,14336]; backward: dY bf16 [8192,14336] + gate/up -> [8192,2,14336]
- Numbers: 1398.5 us measured vs 271.5 us ideal per instance (external bytes 1879.0 MB); gap 36.066 ms/step.
- Dataflow: forward silu output is saved only for backward mul(dY, silu); the backward kernel reads gate anyway.
- Blockers: memory_policy 'none' saves every forward activation; changing what is saved is a joint forward/backward decision.
- Tier T0: Recomputing silu in fp32 then rounding to bf16 reproduces the saved value exactly.
- Notes: Incremental over the two separate SwiGLU regions: 2 x 234.9MB per layer (about 68us per layer, 2.2ms per step).

### R-02 Weight-grad GEMM epilogues writing fp32 directly into the reduce-scatter bucket (T3, measured)

- What: The four weight-gradient GEMMs of each block (wqkv, wo, w13, w2) can store their results, rounded to bf16 and widened to fp32, straight into their rank-major slots of the reduce-scatter buffer, removing the cast, re-layout and pack copies.
- Where: backward layers.*.attention.qkv_linear.wqkv [all] {mm, _to_copy, bucketing._pre_bucket_reduce_scatter}; backward layers.*.attention.wo [all] {mm, _to_copy}; backward layers.*.feed_forward.w13 [all] {mm, _to_copy, cat}; backward layers.*.feed_forward.w2 [all] {mm, _to_copy}; backward layers.*.ffn_norm [all] {_to_copy}; backward layers.*.attention_norm [all] {_to_copy}. Source: `torchtitan/models/common/linear/linear.py:115,360 ; torch/nn/modules/normalization.py:427 ; torch/distributed/tensor/_redistribute.py:1993`. 32 instances/step, 13.0 kernels/instance. Shapes: wgrads bf16-rounded -> f32 bucket [218112000]
- Numbers: 2813.6 us measured vs 1852.0 us ideal per instance (external bytes 2013.3 MB); gap 30.771 ms/step.
- Dataflow: mm(wgrad) -> _to_copy -> (cat) -> _pre_bucket_reduce_scatter -> reduce_scatter.
- Blockers: GEMM output must be split into 8 rank chunks at arbitrary row (and for w13, Shard(1) dim) boundaries; norm-weight grads come from a different kernel.
- Tier T3: Bitwise if the epilogue rounds to bf16 before widening.
- Notes: Superset of the reduce-scatter pack region; GEMM part measured below the random-data peak.

### R-03 w2 input-gradient GEMM with SwiGLU-backward epilogue (T3, measured)

- What: The w2 input-gradient GEMM [8192,4096]x[4096,14336] produces dY for SwiGLU; its epilogue can read gate, up and silu for the same tile and write the stacked [8192,2,14336] gradient directly.
- Where: backward layers.*.feed_forward.w2 [all] {mm}; backward layers.*.feed_forward [all] {mul, silu_backward, stack}. Source: `torchtitan/models/common/linear/linear.py:360 ; torchtitan/models/common/feed_forward.py:75`. 32 instances/step, 5.0 kernels/instance. Shapes: dout bf16 [8192,4096] x W2 bf16 [4096,14336] -> dY (in-register) -> d_gate/d_up bf16 [8192,2,14336] <!-- codespell:ignore -->
- Numbers: 1304.2 us measured vs 498.6 us ideal per instance (external bytes 1359.0 MB); gap 25.780 ms/step.
- Dataflow: mm(dout, W2) -> dY -> mul/mul/silu_backward/stack -> w13 backward GEMMs; dY is also needed by the w2 wgrad GEMM (mm(mul_out^T... ) uses dout, not dY), so dY has no other consumer. <!-- codespell:ignore -->
- Blockers: Custom epilogue GEMM reading three extra operands per tile.
- Tier T3: Bitwise if dY is rounded to bf16 in the epilogue before the SwiGLU backward math.

### R-04 FSDP reduce-scatter pack: weight-grad fp32 casts, w13 re-layout and bucket copy (T2, measured)

- What: After each decoder block's backward, its six bf16 weight gradients (wqkv 6144x4096, wo 4096x4096, w13 2x14336x4096, w2 4096x14336, two 4096 norm weights) are cast to fp32 one by one, the w13 gradient is re-laid out rank-major with a second full copy, and everything is copied again into the flat fp32 reduce-scatter buffer; one kernel can read each bf16 gradient once and write fp32 straight into its rank-major slot of the buffer.
- Where: backward layers.*.feed_forward.w2 [all] {_to_copy}; backward layers.*.feed_forward.w13 [all] {_to_copy, cat}; backward layers.*.ffn_norm [all] {_to_copy}; backward layers.*.attention.wo [all] {_to_copy}; backward layers.*.attention.qkv_linear.wqkv [all] {_to_copy, bucketing._pre_bucket_reduce_scatter}; backward layers.*.attention_norm [all] {_to_copy}. Source: `torchtitan/models/common/linear/linear.py:115,360 ; torch/nn/modules/normalization.py:427 ; torch/distributed/tensor/_redistribute.py:1993`. 32 instances/step, 9.0 kernels/instance. Shapes: in: bf16 [6144,4096],[4096,4096],[2,14336,4096],[4096,14336],[4096]x2 -> out: f32 [218112000] bucket (8 rank-major chunks of 27264000)
- Numbers: 991.8 us measured vs 189.1 us ideal per instance (external bytes 1308.7 MB); gap 25.686 ms/step.
- Dataflow: mm(wgrad, bf16) -> _to_copy(f32) [-> view [2,14336,4096] -> split(1792,dim1) -> cat(dim0) for w13] -> _pre_bucket_reduce_scatter (chunk_cat into f32[8*27264000]) -> reduce_scatter. Every intermediate has exactly one consumer, so the chain collapses to a single gather-cast-scatter.
- Blockers: The cast, the w13 re-layout and the bucket copy come from three layers of software (FSDP grad redistribute, the DTensor Shard(1) placement of w13, and the bucketing pass); a fused kernel needs the bucket layout (per-rank offsets) at graph-pass time. The w13 Shard(1) placement is what forces the extra re-layout copy.
- Tier T2: Pure cast (bf16->fp32 is exact) plus copies; the only constraint is reproducing the exact rank-major placement of each element in the bucket, which is deterministic index math.
- Notes: The pack node is annotated with the block's wqkv module but packs all six parameters of the block. The w13 'cat' alone runs at 2.9x its own roofline (non-vectorized CatArrayBatchedCopy on fp32).

### R-05 SwiGLU backward: two muls + silu_backward + stack (T0, measured)

- What: In each feed_forward backward, the gradient of silu(gate)*up ([8192,14336] bf16) is computed by two separate multiplies, a silu_backward and a stack that interleaves d_gate/d_up into [8192,2,14336]; one elementwise kernel can read dY, gate, up and the saved silu once and write the stacked gradient directly.
- Where: backward layers.*.feed_forward [all] {mul, silu_backward, stack}. Source: `torchtitan/models/common/feed_forward.py:75`. 32 instances/step, 4.0 kernels/instance. Shapes: dY bf16 [8192,14336]; gate/up bf16 [8192,2,14336] (stride 28672); silu bf16 [8192,14336] -> bf16 [8192,2,14336]
- Numbers: 840.1 us measured vs 203.6 us ideal per instance (external bytes 1409.3 MB); gap 20.368 ms/step.
- Dataflow: mul(dY, silu_saved) -> d_up; mul(dY, up) -> silu_backward(., gate) -> d_gate; stack([d_gate, d_up], -2) -> w13 backward GEMMs. All intermediates single-use.
- Blockers: None structural; the strided gate/up halves (unbind of the fused w13 output) are what make the separate aten kernels fall back to the slow non-vectorized path.
- Tier T0: Pointwise only; must keep aten's rounding points (silu value rounded to bf16 before dY*silu; dY*up rounded to bf16 before silu_backward).
- Notes: mul(dY, up) and silu_backward read the stride-28672 halves and run at ~2.8-3.1x roofline; the contiguous mul(dY, silu) runs at roofline.

### R-06 w13 GEMM with SwiGLU epilogue (T3, measured)

- What: The fused gate/up projection [8192,4096]x[4096,28672] can compute silu(gate)*up in its epilogue (pairing gate and up columns in the same tile), writing the product next to the gate/up output kept for backward.
- Where: forward layers.*.feed_forward.w13 [all] {mm}; forward layers.*.feed_forward [all] {silu, mul}. Source: `torchtitan/models/common/linear/linear.py:115 ; torchtitan/models/common/feed_forward.py:75`. 32 instances/step, 3.0 kernels/instance. Shapes: x bf16 [8192,4096] x W bf16 [28672,4096]^T -> gate/up bf16 [8192,2,14336] + silu, product bf16 [8192,14336]
- Numbers: 1455.3 us measured vs 997.2 us ideal per instance (external bytes 1241.5 MB); gap 14.659 ms/step.
- Dataflow: mm(w13) -> view [8192,2,14336] -> unbind -> silu, mul -> w2 GEMM.
- Blockers: Tile must cover matching gate and up columns (they are 14336 columns apart in the fused weight); custom epilogue GEMM.
- Tier T3: Bitwise if the epilogue rounds gate/up to bf16 first and evaluates silu and the product with aten's rounding points.

### R-07 SwiGLU forward: silu + mul on the strided gate/up halves of the w13 output (T0, measured)

- What: Each feed_forward applies silu to the gate half and multiplies by the up half of the fused w13 output [8192,2,14336] bf16 in two kernels that both read stride-28672 views; one kernel can read gate/up once and write silu (kept for backward) and the product.
- Where: forward layers.*.feed_forward [all] {silu, mul}. Source: `torchtitan/models/common/feed_forward.py:75`. 32 instances/step, 2.0 kernels/instance. Shapes: bf16 [8192,2,14336] -> bf16 [8192,14336] x2 (silu saved, product)
- Numbers: 558.4 us measured vs 135.7 us ideal per instance (external bytes 939.5 MB); gap 13.526 ms/step.
- Dataflow: mm(w13) -> view [8192,2,14336] -> unbind -> silu(gate) ; mul(silu, up) -> w2 GEMM. silu output also saved for backward (memory_policy none).
- Blockers: None structural.
- Tier T0: Pointwise; product must use the bf16-rounded silu value as aten does.
- Notes: Both kernels run on non-contiguous views (elementwise_kernel<128,4>, 4.1x and 2.8x roofline). Ideal keeps the silu output write because backward consumes it; see the forward-backward region for dropping it.

### R-08 lm_head GEMM + cross-entropy + input-gradient GEMM per chunk (logits never written) (T3, measured)

- What: For each 1024-token chunk, the lm_head projection to 128256-way logits, the cross-entropy and its gradient, and the input-gradient GEMM can run as one fused linear-cross-entropy kernel that keeps the logits tile on chip, instead of writing and re-reading [1024,128256] bf16/fp32 tensors five times.
- Where: forward lm_head [all] {mm}; backward lm_head [all] {mm}; forward loss [all] {_to_copy, _log_softmax, nll_loss_forward, div, add, ones_like, new_zeros, clone}; backward loss [all] {div, nll_loss_backward, _log_softmax_backward_data, _to_copy}. Source: `torchtitan/models/common/linear/linear.py:115 ; torchtitan/distributed/local_compile.py:96 ; torchtitan/experiments/graph_trainer/chunked_loss.py:107`. 8 instances/step, 13.2 kernels/instance. Shapes: h bf16 [1024,4096] x W bf16 [128256,4096]^T -> (logits on chip) -> loss f32, dh bf16 [1024,4096]
- Numbers: 2108.9 us measured vs 1115.2 us ideal per instance (external bytes 1067.5 MB); gap 7.950 ms/step.
- Dataflow: mm(h, W^T) -> CE chain (R: chunked CE) -> mm(dlogits, W) -> dh; the wgrad GEMM (dlogits^T h) also needs dlogits, so either it is fused too (recompute logits per tile) or dlogits is still written once.
- Blockers: The wgrad GEMM needs dlogits for all vocab rows; a fused kernel must either also produce dW (accumulating across chunks) or still write dlogits once. Bitwise parity with the unfused softmax is unlikely.
- Tier T3: Online-softmax reductions change the reduction order versus cunn_SoftMaxForward; bitwise identity would require replicating the row-wise reduction tree across vocab tiles, which is hard.
- Notes: Ideal counts only the two GEMMs it absorbs (forward logits and input gradient) at the bf16 peak; measured includes those GEMMs plus the CE chain. GEMMs run below the random-data peak here.

### R-09 Chunked cross-entropy: logits cast, log_softmax, NLL and their backward per 1024-token chunk (T1, measured)

- What: For each of the 8 chunks of 1024 tokens, the bf16 logits [1024,128256] are cast to fp32, log-softmaxed, NLL-reduced, and immediately differentiated (nll_loss_backward zero-fill + scatter, log_softmax backward, cast back to bf16) in 6 large and 4 scalar kernels; a single kernel can read the bf16 logits once and emit the loss contribution and the bf16 logit gradient.
- Where: forward loss [all] {_to_copy, _log_softmax, nll_loss_forward, div, add, ones_like, new_zeros, clone}; backward loss [all] {div, nll_loss_backward, _log_softmax_backward_data, _to_copy}. Source: `torchtitan/distributed/local_compile.py:96 ; torchtitan/experiments/graph_trainer/chunked_loss.py:107`. 8 instances/step, 11.2 kernels/instance. Shapes: logits bf16 [1024,128256] -> f32 [1024,128256] (x3 intermediates) -> grad bf16 [1024,128256]; labels i64 [1024]
- Numbers: 1035.5 us measured vs 75.9 us ideal per instance (external bytes 525.3 MB); gap 7.677 ms/step.
- Dataflow: mm(lm_head) bf16 -> _to_copy f32 -> _log_softmax (saved) -> nll_loss_forward -> div(token count) -> add (loss accumulator); ones_like -> div -> nll_loss_backward (memset 525MB + scatter) -> _log_softmax_backward_data -> _to_copy bf16 -> lm_head backward GEMMs. The backward for chunk i runs right after its forward, so the fp32 log-probs never need to leave the kernel.
- Blockers: The chunk's backward is driven by autograd inside the loss forward; fusing forward and backward needs the upstream gradient scale (1/token count) available at forward time, which it is in this joint graph.
- Tier T1: Row max / sum-exp reductions must use the same reduction order as cunn_SoftMaxForward; the gradient (exp(logp)*sum(g) - g) then follows pointwise; the per-chunk loss scalar adds must keep chunk order.
- Notes: Even without the forward-backward fusion, fusing the forward (cast+log_softmax+nll) and the backward (nll_bwd+log_softmax_bwd+cast) separately would remove most of the gap; nll_loss_backward's 525MB zero-fill alone is 78us per chunk.

### R-10 Weight unpack folded into GEMM operand loads (consume the all-gather buffer in place) (T3, measured)

- What: Instead of copying the gathered weights out of the rank-major all-gather buffer, the wqkv/wo/w13/w2/lm_head GEMMs and the norms read their operand tiles straight from the 8 per-rank chunks, removing the unpack copies entirely.
- Where: forward layers.*.attention.qkv_linear.wqkv [all] {split_with_sizes_copy}; forward layers.*.feed_forward.w13 [all] {cat}; forward norm [all] {split_with_sizes_copy}. Source: `torchtitan/models/common/linear/linear.py:114 ; torch/nn/modules/normalization.py:427`. 33 instances/step, 3.0 kernels/instance. Shapes: bf16 [8, per-rank shard] chunks -> GEMM B operands [6144,4096],[4096,4096],[28672,4096],[4096,14336],[128256,4096]
- Numbers: 210.2 us measured vs 0.0 us ideal per instance (external bytes 0.0 MB); gap 6.936 ms/step.
- Dataflow: all_gather output -> (copy) -> weight -> mm. The copy has a single consumer per parameter.
- Blockers: Needs a GEMM that takes a rank-chunked B operand (grouped/TMA multi-descriptor), and for w13 the Shard(1) chunks interleave within each of the gate/up halves; the backward GEMMs also reuse the unpacked weights, so the copy must either stay or both forward and backward GEMMs read the chunked buffer (which must then stay alive).
- Tier T3: GEMM must accept a chunked operand (8 base pointers / TMA descriptors); results bitwise equal if the GEMM's K-order is unchanged.
- Notes: Ideal 0: the GEMMs read the weights anyway. Includes the lm_head+norm bucket unpack (1 instance).

### R-11 Attention-backward postprocess fused with RoPE backward and d_qkv concat (T3, measured)

- What: The attention backward converts its fp32 dq/dk/dv accumulators to bf16 in three postprocess kernels, after which RoPE backward re-casts dq/dk to fp32, rotates, casts back and concatenates into d_qkv; the postprocess can apply the inverse rotation and write d_qkv [8192,8,6,128] directly.
- Where: backward layers.*.attention.inner_attention [all] {torch_attn._varlen_attn_backward}; backward layers.*.attention.rope [all] {_to_copy, clone, mul}; backward layers.*.attention.qkv_linear [all] {cat}. Source: `torchtitan/models/common/rope.py:171 ; torchtitan/models/common/attention/attention.py:189`. 32 instances/step, 12.0 kernels/instance. Shapes: dq_accum f32 [8192,32,128], dk/dv accum f32 [8192,8,128] -> d_qkv bf16 [8192,8,6,128]
- Numbers: 227.9 us measured vs 44.2 us ideal per instance (external bytes 306.2 MB); gap 5.877 ms/step.
- Dataflow: flash_bwd -> postprocess (dq,dk,dv -> bf16) -> _to_copy f32 -> mul(conj freqs) -> _to_copy bf16 -> cat -> wqkv backward GEMMs.
- Blockers: Attention library internals (flash_attn cute postprocess kernels); needs a custom epilogue hook.
- Tier T3: Bitwise only if the epilogue reproduces the bf16 rounding of dq/dk before the fp32 rotation.
- Notes: Assumes all three postprocess kernels convert fp32 accumulators (GQA dK/dV accumulate in fp32 in deterministic mode).

### R-12 wqkv GEMM epilogue emitting rotated q/k and contiguous v (T3, measured)

- What: The wqkv projection [8192,4096]x[4096,6144] can apply RoPE in its epilogue and store q, k, v in the contiguous per-head layouts attention consumes, removing the 10 prologue kernels.
- Where: forward layers.*.attention.qkv_linear.wqkv [all] {mm}; forward layers.*.attention.qkv_linear [all] {clone}; forward layers.*.attention.rope [all] {index, _to_copy, mul}. Source: `torchtitan/models/common/attention/attention.py:194-196 ; torchtitan/models/common/rope.py:170-171 ; torchtitan/models/common/linear/linear.py:115`. 32 instances/step, 11.0 kernels/instance. Shapes: x bf16 [8192,4096] x W bf16 [6144,4096]^T -> q bf16 [8192,32,128], k/v bf16 [8192,8,128]
- Numbers: 382.5 us measured vs 213.7 us ideal per instance (external bytes 222.4 MB); gap 5.403 ms/step.
- Dataflow: mm(wqkv) -> split/clone -> rope -> attention.
- Blockers: The backward needs the rotated q/k (saved for attention backward) - unchanged; needs a GEMM with a custom epilogue and positional table input.
- Tier T3: Bitwise if the epilogue rounds the accumulator to bf16 before the fp32 rotation, matching the unfused rounding points.
- Notes: GEMM measured below the random-data peak, so the gap is GEMM-ideal-relative.

### R-13 RoPE forward: q/k/v contiguous copies + fp32 complex rotation of q and k (T0, measured)

- What: Each attention splits the wqkv output [8192,6144] bf16 into q/k/v, makes each contiguous, gathers the rotary table for the positions, and rotates q [8192,32,128] and k [8192,8,128] via cast-to-fp32, complex multiply and cast-back (10 kernels); one kernel can read the qkv rows once and write rotated q, k and contiguous v.
- Where: forward layers.*.attention.qkv_linear [all] {clone}; forward layers.*.attention.rope [all] {index, _to_copy, mul}. Source: `torchtitan/models/common/attention/attention.py:194-196 ; torchtitan/models/common/rope.py:170-171`. 32 instances/step, 10.0 kernels/instance. Shapes: qkv bf16 [8192,8,6,128] (stride 6144) + freqs c64 [8192,64] -> q bf16 [8192,32,128], k/v bf16 [8192,8,128]
- Numbers: 178.7 us measured vs 29.7 us ideal per instance (external bytes 205.6 MB); gap 4.769 ms/step.
- Dataflow: mm(wqkv) -> view/split -> clone(q), clone(k), clone(v) -> _to_copy f32 -> view_as_complex -> mul(freqs[index(positions)]) -> view_as_real -> _to_copy bf16 -> _varlen_attn. The three clones and two rotations are siblings over the same qkv rows.
- Blockers: The rotary table gather is done per layer from the same positions; a fused kernel should take positions + table directly. Attention needs contiguous q/k/v in this layout (varlen packed).
- Tier T0: Casts and complex multiply only; the fused kernel must evaluate the complex product with the same operation order/FMA contraction as aten's complex mul in fp32.
- Notes: Overlaps the wqkv-epilogue (T3) and the rotary-table CSE regions.

### R-14 RoPE backward: inverse rotation of dq/dk + concat into d_qkv (T0, measured)

- What: Each attention backward rotates dq [8192,32,128] and dk [8192,8,128] by the conjugate rotary table (cast to fp32, materialize conj, complex multiply, cast back) and concatenates dq/dk/dv into the [8192,8,6,128] wqkv-output gradient (9 kernels); one kernel can read dq/dk/dv once and write d_qkv.
- Where: backward layers.*.attention.rope [all] {_to_copy, clone, mul}; backward layers.*.attention.qkv_linear [all] {cat}. Source: `torchtitan/models/common/rope.py:171 ; torchtitan/models/common/attention/attention.py:189`. 32 instances/step, 9.0 kernels/instance. Shapes: dq bf16 [8192,32,128], dk/dv bf16 [8192,8,128], freqs c64 [8192,1,64] -> d_qkv bf16 [8192,8,6,128]
- Numbers: 170.9 us measured vs 29.7 us ideal per instance (external bytes 205.5 MB); gap 4.519 ms/step.
- Dataflow: _varlen_attn_backward -> (dq, dk, dv) -> _to_copy f32 -> mul(conj(freqs) cloned) -> _to_copy bf16 -> cat(dim=heads-group) -> wqkv backward GEMMs.
- Blockers: None structural.
- Tier T0: Casts + complex multiply with the conjugated table; same rounding points as the aten chain.
- Notes: Overlaps the attention-backward-postprocess (T3) and the rotary-table CSE regions.

### R-15 lm_head weight-gradient fp32 cast fused into the per-chunk wgrad GEMM epilogue (T3, measured)

- What: For each loss chunk, the lm_head weight gradient [128256,4096] is produced in bf16 by a GEMM and then cast to fp32 for the reduce-scatter by a separate 3.15GB-traffic kernel; the GEMM epilogue can round to bf16 and store fp32 directly.
- Where: backward lm_head [all] {mm, _to_copy}. Source: `torchtitan/models/common/linear/linear.py:115`. 8 instances/step, 2.0 kernels/instance. Shapes: dlogits bf16 [1024,128256]^T x h bf16 [1024,4096] -> f32 [128256,4096]
- Numbers: 940.0 us measured vs 557.6 us ideal per instance (external bytes 2372.4 MB); gap 3.059 ms/step.
- Dataflow: mm(dlogits^T, h) bf16 -> _to_copy f32 -> reduce_scatter -> add_ (accumulate across chunks).
- Blockers: Needs a GEMM epilogue with custom rounding; the reduce-scatter per chunk (8x the gradient traffic) is itself a structural cost of chunking the loss before reduce-scatter.
- Tier T3: Bitwise if the epilogue rounds the fp32 accumulator to bf16 before widening to fp32.
- Notes: Measured GEMM time is below the random-data peak (low-entropy data), so the fused ideal (GEMM at peak) is conservative.

### R-16 RMSNorm backward + residual-gradient accumulation add (T1, measured)

- What: At every attention_norm and ffn_norm backward ([8192,4096] bf16), the normalized-input gradient is written out and then added to the residual-stream gradient by a separate kernel; one kernel can compute the norm backward and add the incoming residual gradient before writing.
- Where: backward norm [all] {_fused_rms_norm_backward}; backward layers.*.ffn_norm [all] {_fused_rms_norm_backward}; backward layers.* [all] {add}; backward layers.*.attention_norm [all] {_fused_rms_norm_backward}. Source: `torch/nn/modules/normalization.py:427`. 64 instances/step, 3.0 kernels/instance. Shapes: dy, x, residual-grad bf16 [8192,4096]; rstd f32 [8192,1]; weight bf16 [4096] -> dx bf16 [8192,4096], dweight bf16 [4096]
- Numbers: 84.9 us measured vs 39.1 us ideal per instance (external bytes 270.6 MB); gap 2.934 ms/step.
- Dataflow: mm(dgrad of w13 or wqkv) -> _fused_rms_norm_backward -> dx -> add(residual grad, dx) -> next GEMMs. The add's other input is the running residual-stream gradient; dx has no other consumer.
- Blockers: The add is created by autograd's gradient accumulation (multiple uses of the residual tensor), outside any module; the norm backward is a CuTe DSL kernel (quack) so the add must become its epilogue.
- Tier T1: Row reductions (dy*w*x dot) and the column reduction for dweight must keep quack's reduction order; the residual add is a bf16 pointwise epilogue on the rounded dx.
- Notes: The first instance (layers.31 FFN residual) also absorbs the final norm's backward, whose dx feeds the same add.

### R-17 FSDP all-gather unpack: split_with_sizes_copy + w13 re-layout cat (T2, measured)

- What: After each block's weight all-gather, the rank-major bf16 buffer [8,27264000] is copied out into six per-parameter tensors, and the w13 weight is then copied a second time from rank-major [16,1792,4096] into [2,14336,4096]; a single copy can write every parameter, including w13, directly in its final layout.
- Where: forward layers.*.attention.qkv_linear.wqkv [all] {split_with_sizes_copy}; forward layers.*.feed_forward.w13 [all] {cat}. Source: `torchtitan/models/common/linear/linear.py:114`. 32 instances/step, 3.0 kernels/instance. Shapes: bf16 [8,27264000] -> bf16 [6144,4096],[4096,4096],[2,14336,4096],[4096,14336],[4096]x2
- Numbers: 207.1 us measured vs 126.0 us ideal per instance (external bytes 872.4 MB); gap 2.593 ms/step.
- Dataflow: all_gather_into_tensor_out -> wait -> view [8,-1] -> split_with_sizes_copy(out=6 buffers) -> (w13) split(2) -> cat(dim1) -> reshape [28672,4096] -> mm. Each output is consumed only by its parameter's GEMM/norm.
- Blockers: Unpack is generated by the bucketing pass and the w13 re-layout by DTensor Shard(1) redistribution; fusing needs the w13 placement known to the unpack.
- Tier T2: Pure data movement; exact element placement (rank-major chunks, Shard(1) for w13) must be reproduced.
- Notes: Weights stay unsharded through backward (memory_policy none), so there is no backward all-gather unpack.

### R-18 Residual add in the wo / w2 GEMM epilogues (T3, measured)

- What: The attention output projection (wo) and the FFN down projection (w2) each write [8192,4096] bf16 that is immediately added to the residual stream; the GEMM epilogue can add the residual and write h directly.
- Where: forward layers.*.attention.wo [all] {mm}; forward layers.* [all] {add}; forward layers.*.feed_forward.w2 [all] {mm}. Source: `torchtitan/models/llama3/model.py:60-64 ; torchtitan/models/common/linear/linear.py:360`. 64 instances/step, 2.0 kernels/instance. Shapes: GEMM out bf16 [8192,4096] + residual bf16 [8192,4096] -> h bf16 [8192,4096]
- Numbers: 347.3 us measured vs 320.5 us ideal per instance (external bytes 360.7 MB); gap 1.711 ms/step.
- Dataflow: mm(wo or w2) -> add(residual, .) -> h -> norm.
- Blockers: Competes with the add+RMSNorm fusions for the same add; the norm needs the whole row, which a GEMM tile of N=4096 rarely covers.
- Tier T3: Bitwise if the epilogue rounds the accumulator to bf16 before adding the residual (aten adds two bf16 tensors).
- Notes: Ideal per instance is the mean of the two GEMMs' compute ideals; GEMMs measured slightly below that ideal.

### R-19 Token-embedding backward: zero-fill, deterministic scatter-add, fp32 cast and bucket copy (T2, measured)

- What: The embedding weight gradient [128256,4096] is zero-filled in bf16, scatter-added from the [8192,4096] token gradients via sort/segment kernels, cast to fp32 and then copied again into the reduce-scatter buffer; one pass can write the fp32 bucket directly (zeros for untouched rows, segment sums for touched rows).
- Where: backward tok_embeddings [all] {embedding_dense_backward, _to_copy, bucketing._pre_bucket_reduce_scatter}. Source: `torchtitan/models/common/embedding.py:41 ; torch/distributed/tensor/_redistribute.py:1993`. 1 instances/step, 24.0 kernels/instance. Shapes: grad bf16 [8192,4096], ids i64 [8192] -> bf16 [128256,4096] -> f32 [128256,4096] -> f32 bucket [525336576]
- Numbers: 1844.8 us measured vs 313.3 us ideal per instance (external bytes 2168.5 MB); gap 1.531 ms/step.
- Dataflow: embedding_dense_backward (memset 1.05GB, radix sort, compute_grad_weight, sum_and_scatter) -> _to_copy f32 -> _pre_bucket_reduce_scatter (chunk_cat) -> reduce_scatter.
- Blockers: Deterministic-algorithms mode selects the sort-based embedding backward; the bucket pack is from the bucketing pass.
- Tier T2: Deterministic segment sums (sorted by token id) must keep the same per-row accumulation order and the bf16 rounding before widening.

### R-20 Attention-backward accumulator zero-fill folded into the preprocess kernel (T3, measured)

- What: Each attention backward launches five memsets (two 51MB fp32 accumulators and three small buffers) before its preprocess kernel (rowsum of dO*O); the preprocess kernel can zero the accumulators in the same pass.
- Where: backward layers.*.attention.inner_attention [all] {torch_attn._varlen_attn_backward}. Source: `torch/_library/autograd.py:53 (varlen attention backward)`. 32 instances/step, 6.0 kernels/instance. Shapes: memsets 2x51380224 B + 3x278528 B; preprocess reads O,dO bf16 [8192,32,128], writes f32 [32,8192]
- Numbers: 67.5 us measured vs 34.5 us ideal per instance (external bytes 238.9 MB); gap 1.054 ms/step.
- Dataflow: memset(dq_accum), memset(dk/dv accum), memset(semaphores) ; flash_bwd_preprocess(O, dO) -> flash_bwd.
- Blockers: Attention library internals (flash_attn cute); requires a library change.
- Tier T3: Inside the attention library; zero-fill is exact.
- Notes: Footprint op is the attention-backward op; only its memset and preprocess kernels are members.

### R-21 lm_head per-chunk reduce-scattered gradient accumulation folded into the reduce-scatter (T1, estimated)

- What: The fp32 lm_head gradient shard [16032,4096] reduce-scattered for each loss chunk is added into a running fp32 accumulator by a separate kernel (7 times); the reduce-scatter's own reduction can take the accumulator as one more operand.
- Where: backward loss [all] {add_}. Source: `torchtitan/experiments/graph_trainer/chunked_loss.py (chunk loop, lm_head grad reduce-scatter per chunk)`. 7 instances/step, 1.0 kernels/instance. Shapes: f32 [16032,4096] + f32 [16032,4096] -> f32 [16032,4096]
- Numbers: 110.0 us measured vs 38.0 us ideal per instance (external bytes 262.7 MB); gap 0.504 ms/step.
- Dataflow: reduce_scatter(chunk i) -> wait -> add_(acc, .) -> ... -> final grad.
- Blockers: Requires a reduce-scatter with an accumulate-into-output epilogue (not available in NCCL); alternatively accumulate the 8 chunk wgrads in fp32 before a single reduce-scatter, which changes summation order (not bitwise) but removes 7 of 8 reduce-scatters.
- Tier T1: Changes where the add happens inside the reduction; bitwise only if the accumulator is added after the cross-rank sum, in the same order.
- Notes: Ideal assumes the collective reads the accumulator once more (262.7MB); this is a model, not a measurement of a fused kernel.

### R-22 Rotary table gather and conjugate computed once per step instead of per layer (T0, measured)

- What: Every layer's attention gathers the same [8192,64] complex rotary rows for the same positions in forward, and every layer's backward materializes the same conjugated table twice (for dq and dk); computing the gathered table and its conjugate once per step removes 95 of 96 kernels.
- Where: forward layers.*.attention.rope [all] {index}; backward layers.*.attention.rope [all] {clone}. Source: `torchtitan/models/common/rope.py:170-171`. 1 instances/step, 96.0 kernels/instance. Shapes: freqs c64 [max_seq,64], positions i64 [8192] -> c64 [8192,64]; conj c64 [8192,1,64]
- Numbers: 390.4 us measured vs 2.4 us ideal per instance (external bytes 16.8 MB); gap 0.388 ms/step.
- Dataflow: index(freqs_cis, positions) per layer (identical inputs) ; _conj -> clone per q/k per layer (identical).
- Blockers: Each layer's rope module traces its own gather; a graph pass must recognize identical inputs (same placeholder table and positions).
- Tier T0: Gather and conjugation are exact.
- Notes: Absorbed automatically if the fused RoPE kernels read positions + table directly.

### R-23 Attention residual add + ffn_norm (T1, measured)

- What: In each decoder block, h = x + attn_out ([8192,4096] bf16) is written and immediately re-read by ffn_norm; one kernel can add, write h (still needed as residual and for backward), and write the normalized output and rstd.
- Where: forward layers.* [all] {add}; forward layers.*.ffn_norm [all] {_fused_rms_norm}. Source: `torchtitan/models/llama3/model.py:60 ; torch/nn/modules/normalization.py:427`. 32 instances/step, 2.0 kernels/instance. Shapes: bf16 [8192,4096] x2 + weight bf16 [4096] -> h bf16, normed bf16 [8192,4096], rstd f32 [8192,1]
- Numbers: 50.1 us measured vs 38.8 us ideal per instance (external bytes 268.5 MB); gap 0.362 ms/step.
- Dataflow: mm(wo) -> add(x, .) -> h -> _fused_rms_norm(h) -> w13 GEMM; h also feeds the next residual add.
- Blockers: Norm is a library CuTe DSL kernel; fused add+norm variant needed.
- Tier T1: RMSNorm row reduction order must match quack's kernel; the add is bf16 pointwise ahead of it.
- Notes: T3 alternative: fold the residual add into the wo GEMM epilogue (see residual-epilogue region).

### R-24 FFN residual add + next layer's attention_norm (final norm after layer 31) (T1, measured)

- What: The block output out = h + ffn_out ([8192,4096] bf16) is written by one kernel and re-read by the next block's attention_norm (or the final norm); one kernel can do both across the layer boundary.
- Where: forward layers.* [all] {add}; forward layers.*.attention_norm [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31] {_fused_rms_norm}; forward norm [all] {_fused_rms_norm}. Source: `torchtitan/models/llama3/model.py:64 ; torch/nn/modules/normalization.py:427`. 32 instances/step, 2.0 kernels/instance. Shapes: bf16 [8192,4096] x2 + weight bf16 [4096] -> out bf16, normed bf16 [8192,4096], rstd f32 [8192,1]
- Numbers: 49.4 us measured vs 38.8 us ideal per instance (external bytes 268.5 MB); gap 0.338 ms/step.
- Dataflow: mm(w2) -> add(h, .) -> out -> next layer _fused_rms_norm(out) -> wqkv GEMM; out also feeds the next residual add.
- Blockers: The next layer's FSDP all-gather unpack sits between the two ops in program order (no data dependency).
- Tier T1: Same as the intra-layer add+norm.

### R-25 FSDP all-gather input packs (fp32 shard -> bf16 slice) batched into one launch per step (T0, measured)

- What: Before every block's weight all-gather, its fp32 parameter shards are cast to bf16 into this rank's slice of the gather buffer by one multi-tensor kernel per bucket (34 per step, about 1.3x roofline each); since all shards are available at step start, the 34 casts can run as one multi-tensor launch.
- Where: forward tok_embeddings [all] {bucketing._pre_bucket_all_gather}; forward layers.*.attention.qkv_linear.wqkv [all] {bucketing._pre_bucket_all_gather}; forward norm [all] {bucketing._pre_bucket_all_gather}. Source: `torchtitan/models/common/embedding.py:43 ; torchtitan/models/common/linear/linear.py:114 ; torch/nn/modules/normalization.py:427`. 1 instances/step, 34.0 kernels/instance. Shapes: per layer: f32 [768,4096],[512,4096],[2,1792,4096],[512,14336],[512]x2 -> bf16 slice of [218112000]; tok_embeddings / norm+lm_head buckets f32 [16032,4096]
- Numbers: 1113.5 us measured vs 870.2 us ideal per instance (external bytes 6022.7 MB); gap 0.243 ms/step.
- Dataflow: param shard (placeholder) -> _pre_bucket_all_gather (cast + copy into slice) -> all_gather_into_tensor_out (in place). The 34 packs have no data dependencies on each other or on the forward.
- Blockers: All 34 gather output buffers (436MB each per layer) would be allocated at step start instead of just before each layer's all-gather; with memory_policy 'none' they all stay alive until backward anyway, so peak memory changes little, but the bucketing pass emits one pack per bucket by construction.
- Tier T0: fp32 -> bf16 casts only (round-to-nearest-even, identical in any kernel).
- Notes: Small: the gap is launch and tail overhead of 34 separate multi-tensor launches. A further step would write the bf16 copy in the previous step's optimizer update (out of scope).

### R-26 lm_head input-gradient written in place into the hidden-state gradient buffer (T3, measured)

- What: Each chunk's lm_head input gradient [1024,4096] bf16 is produced by a GEMM and then copied into a zero-initialized [8192,4096] buffer; the GEMM can write its rows of the buffer directly and the zero-fill is unnecessary because the 8 chunks cover all rows.
- Where: forward loss [all] {zeros_like, copy_}. Source: `torchtitan/experiments/graph_trainer/chunked_loss.py (chunk loop)`. 1 instances/step, 9.0 kernels/instance. Shapes: bf16 [1024,4096] x8 -> slices of bf16 [8192,4096]
- Numbers: 47.1 us measured vs 0.0 us ideal per instance (external bytes 0.0 MB); gap 0.047 ms/step.
- Dataflow: zeros_like(h) ; mm(dlogits, W) -> copy_ into slice [i*1024:(i+1)*1024] -> final norm backward.
- Blockers: The copy is generated by the chunked loss's manual gradient assembly; the zero-fill is only removable if every row is provably written.
- Tier T3: Writing the GEMM result to a row slice is bitwise identical.
- Notes: These ops are tagged as forward in the graph (they run in the loss module's forward, outside autograd's backward).

### R-27 Token-embedding gather + layer-0 attention_norm (T1, measured)

- What: The embedding gather writes the [8192,4096] bf16 hidden state and layer 0's attention_norm re-reads it; one kernel can gather rows, write the residual and the normalized output.
- Where: forward tok_embeddings [all] {embedding}; forward layers.*.attention_norm [0] {_fused_rms_norm}. Source: `torchtitan/models/common/embedding.py:41 ; torch/nn/modules/normalization.py:427`. 1 instances/step, 2.0 kernels/instance. Shapes: ids i64 [8192], table bf16 [128256,4096] -> h bf16 [8192,4096], normed bf16 [8192,4096]
- Numbers: 37.0 us measured vs 29.1 us ideal per instance (external bytes 201.4 MB); gap 0.008 ms/step.
- Dataflow: embedding(table, ids) -> h0 -> _fused_rms_norm -> wqkv GEMM; h0 also the residual.
- Blockers: None structural; tiny.
- Tier T1: Norm reduction order as quack; the gather is exact.
- Notes: The embedding gather itself measured below its byte roofline (repeated token rows hit in L2).

### R-28 Final norm weight-gradient fp32 cast + reduce-scatter bucket copy (T0, measured)

- What: The final norm's 4096-element bf16 weight gradient is cast to fp32 and then copied into its own reduce-scatter bucket by a second kernel (plus a host-to-device metadata copy); one kernel can write the fp32 bucket directly.
- Where: backward norm [all] {_to_copy, bucketing._pre_bucket_reduce_scatter}. Source: `torch/nn/modules/normalization.py:427 ; torch/distributed/tensor/_redistribute.py:1993`. 1 instances/step, 3.0 kernels/instance. Shapes: bf16 [4096] -> f32 [4096] -> f32 bucket [4096]
- Numbers: 4.9 us measured vs 0.0 us ideal per instance (external bytes 0.0 MB); gap 0.005 ms/step.
- Dataflow: _fused_rms_norm_backward (dweight) -> _to_copy f32 -> _pre_bucket_reduce_scatter -> reduce_scatter.
- Blockers: Launch-bound; a better fix is putting this 4096-element parameter in the same bucket as lm_head (both are packed separately today).
- Tier T0: Exact cast + copy.
- Notes: Pure launch overhead.

## 5. No opportunity

All 118.844 ms of forward/backward memory-bound glue is inside at least one region (coverage check: 118.844 ms in regions, 0.000 ms uncovered). The remaining entries are outside the glue category or outside scope:

| what | ms | reason |
|---|---:|---|
| Attention forward internal memsets (3 tiny zero-fills per layer inside the attention op) | 0.081 | Inside the attention library; ~2.4us per layer of launch-bound memsets. Same fix as the backward memset region (fold into the main kernel), negligible size; listed for completeness. |
| Post-backward gradient accumulation clone into .grad (outside the graph, after backward) | 1.581 | Out of scope (after backward). Would be removable by letting the reduce-scatter write directly into .grad. |

Also not region targets: GEMMs (185.4 ms, each within 0.9-1.1x of the random-data peak), attention main kernels (forward 3.9 ms, backward 19.3 ms; within the full-causal flops bound, true bound unknown), communication emulation (7.3 ms), optimizer and other (out of scope).


## 6. Observations about working from these artifacts

**What the artifacts made easy**
- Exact kernel -> node attribution. Every node runs inside `_RecordFunctionFast("<node>: <op> (<file:line>)")`, so a kernel's correlation id -> CUDA runtime call -> innermost enclosing labeled CPU op gives the graph node with no heuristics. 99.5% of forward/backward kernel time mapped this way once the label pattern was right (see below).
- Shapes, strides and dtypes on every node (`val`), plus `args`/`users`, gave bytes and single-consumer checks directly. Strides explained the worst ratios: `silu`/`mul` in the FFN read the two stride-28672 halves of the fused `w13` output and fall back to the non-vectorized elementwise kernel (2.8-4.1x roofline), while the contiguous `mul` next to them runs at roofline.
- `custom.module_fqn` is present on forward *and* backward nodes (the backward node carries the forward module), and `autograd_backward` gives the phase, so backward ops did not need seq_nr matching.
- The readable graph with per-op annotations showed the FSDP glue structure (bucket pack -> collective -> unpack -> Shard(1) re-layout) in a few screens, and memset/memcpy records carry byte counts.

**What made it hard / needed workarounds**
- `nn_module_stack` is null on every node; the module path comes only from `custom.module_fqn` (a quoted string). The 64 autograd gradient-accumulation adds of the residual stream have no module and a stack trace of "Gradient addition node due to multiple use of tensor"; I recovered their block by following their inputs through `getitem` to the producing `_fused_rms_norm_backward`.
- Some record-function labels have no `(file:line)` suffix (e.g. `zeros_like: zeros_like.default`); my first label regex required it and silently dropped 10% of the time to "unattributed" until fixed.
- Loss-chunk identity is not recorded. The chunked loss runs each chunk's backward inside the forward (`ChunkedLossWrapperWithParamGrads`); its stack traces stop at a `local_compile.py:96` wrapper or are empty. I assigned chunk indices by time order of the `lm_head` forward GEMMs. The gradient assembly (`zeros_like`, `copy_` into slices) is tagged forward while the per-chunk autograd ops are tagged backward.
- Generic byte models are wrong for gathers and partial writes: `embedding`, `index`, `nll_loss_forward` (reads 1024 of 131M elements), `nll_loss_backward`, `embedding_dense_backward`, and `_pre_bucket_all_gather` (writes only this rank's 1/8 slice). I used op-specific models for these.
- Attention ideal is unknowable without `cu_seqlens` values (document lengths); only a full-causal upper bound is possible.
- The bucket layouts (rank-major interleaving, 27264000 elements per rank per block) had to be derived from `split_with_sizes` sizes and buffer shapes; they are not recorded as metadata.
- Footprints are op-granular, but two attention-adjacent regions (memsets+preprocess, postprocess epilogue) are kernel-granular inside one op; I selected those kernels by name.
- The first 17 ms of the trace belongs to the previous step (no CPU correlation in the window).
- The profile's GPU timeline is ~95% idle because of host overhead under the profiler, so it says nothing about real comm/compute overlap; communication time is fake-PG copy emulation.

**How phase and module were identified for backward ops**
- Phase: the node's `autograd_backward` flag. Module: the node's `custom.module_fqn`, which for backward nodes is the forward module whose op they differentiate (consistent with the forward stack trace attached to the backward node). Exceptions: the residual gradient-accumulation adds (module from the producing norm backward, labeled `layers.N` = residual fan-out of block N), and the bucket pack node, whose annotation names the block's `wqkv` although it packs all six block parameters.

**False leads discarded**
- Attention backward at 12x "roofline" with a bytes-only ideal; against the full-causal flops bound it measures 0.85x (forward 0.43x), so it is compute-bound work, not a glue target.
- `_pre_bucket_all_gather` at 0.4x and `reduce_scatter` at 0.2x of roofline: wrong byte models (local slice) and emulated communication respectively.
- RMSNorm as a GEMM prologue: the normalized activation must still be written because the weight-gradient GEMM consumes it (everything is saved), so no traffic is saved. RMSNorm backward as a GEMM epilogue: needs full 4096-wide row reductions. Both dropped.
- The chunked-loss code comment says lm_head gradients are reduce-scattered by a last-chunk hook; the graph instead has 8 full-size reduce-scatters of the fp32 `lm_head` gradient (one per chunk) followed by 7 accumulation adds (R-21). That is a real structural cost (8x the lm_head gradient traffic), not a measurement artifact.
- The 7.2 s of GPU idle is profiler/host overhead, not a fusion target.

**Kinds of opportunity I may have missed**
- Fusions inside library kernels beyond their obvious prologue/epilogue (flash-attention internals, quack RMSNorm, nvjet GEMMs) and anything requiring document-length data.
- Scheduling/overlap opportunities (comm/compute overlap, stream concurrency): invisible with a fake process group and a host-bound profile.
- Optimizer-side fusions (AdamW writing the next step's bf16 all-gather input, clipping fused into AdamW): out of scope by the task definition.
- Changes to what is saved for backward beyond SwiGLU (e.g. saving pre-RoPE q/k), and multi-step or cross-microbatch fusion (only one microbatch here).
- Ideal times for T3 regions assume GEMMs at the random-data peak; since the measured GEMMs beat it, real headroom of epilogue fusions may be underestimated.

**Effort**
- Wall time: about 20 minutes from the first command (16:47) to the written deliverables (17:07), well inside the 80/90-minute limits.
- Scripts (all under `out/`): `attribute.py` (kernel -> node attribution), `nodes_table.py` (per-node bytes/flops/ideal, class table, per-kernel CSV), `regions.py` (region definitions, measured/ideal/gap, budget, coverage check, `regions.json`), `write_report.py` (this report). `regions_log.jsonl` was appended inline as regions were identified.

## 7. Reproduction

```bash
cd /home/bahuang/local/step_analysis/llama3_8b_graph
python -I out/attribute.py     # -> out/kernels.jsonl, out/attribution_summary.json
python -I out/nodes_table.py   # -> out/node_table.jsonl, out/class_table.json, out/kernel_attribution.csv
python -I out/regions.py       # -> out/regions.json, out/region_members.json
python -I out/write_report.py  # -> out/report.md
```
CPU-only, standard library only. `out/kernel_attribution.csv` is the per-kernel attribution (node, op, phase, module, layer, category, node-level shapes/bytes/flops/ideal/ratio);
`out/class_table.json` aggregates nodes across layers; `out/region_members.json` lists the member nodes of each region.
