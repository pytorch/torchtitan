# Kernel-fusion opportunities in one Qwen3 30B-A3B training step (eager Trainer)

## 1. Setup

- **Program:** Qwen3 30B-A3B (48 layers, dim 2048, 32 q heads / 4 kv heads x 128, 128 routed experts top-8, expert hidden 768, vocab 151936) from `torchtitan_recipes.graph_trainer.megatron_bridge_benchmarks.qwen3_30b_a3b_dist_moe_mxfp8_8gpu`, run with the eager `Trainer`. Attention projections (`wqkv`, `wo`) are MXFP8 linears (1x32 activation blocks, 32x32 weight blocks); routed experts are Dist-MoE with MXFP8 grouped GEMMs that fuse dispatch, SwiGLU and combine into the GEMMs; FlashAttention (CuTe DSL, SM100) varlen causal attention; router gate is a HiMidLo (fp32-emulating) linear with round-robin top-8 routing and a batch-wise load-balance aux loss; loss is an 8-chunk cross-entropy that runs each chunk's backward immediately.
- **Layout:** FSDP 8 x EP 8 emulated by one process with a fake process group (rank 0's work). One microbatch of 32768 tokens. Attention-group parameters are fp32 and FSDP-sharded 8 ways (bf16 all-gather); expert parameters live in a size-1 FSDP group (fp32, cast to bf16 then quantized each unshard). Blocks reshard after forward, so every unshard is repeated in the backward.
- **Capture:** step 10 after warmup, `local_compile` off (so the model's `rope`, `loss`, `fp32_to_bf16_split` and `fused_binary_activation` compile regions are all eager), deterministic algorithms on, no activation recompute, no CUDA graphs.
- **Roofline numbers used** (from `hw_profile.json`, NVIDIA GB300): HBM copy bandwidth 6921.3 GB/s; GEMM peaks bf16 1929.5 TFLOP/s and MXFP8 3615.4 TFLOP/s (max of the measured shapes). Ideal = max(external bytes / HBM copy bandwidth, flops / peak of the matching precision).
- **Shape letters used below:** T = 32768 tokens, D = 2048, K = 8 (top-k), E = 128 experts, V = 151936.

## 2. Step budget

| category | kernel time (ms) | notes |
|---|---:|---|
| gemm | 317.5 | dense MXFP8/bf16 GEMMs, Dist-MoE grouped GEMMs (dispatch/combine fused), cuBLAS split-K reduce |
| attention | 155.7 | flash-attention fwd/bwd; includes 15.5 ms of FA pre/post-processing kernels that are covered by regions |
| communication | 6.0 | kernels under `c10d::` / `_c10d_functional::` ops (emulated all-gather, reduce-scatter, all-reduce copies); excluded from regions |
| memory_bound_glue | 454.1 | everything else in forward/backward (includes 22.9 ms of fake-PG-only zero-fill) |
| optimizer | 27.2 | fused AdamW + grad-norm, out of scope |
| other | 9.8 | before forward (zero_grad, data) and after backward before the optimizer, out of scope |

| phase | GPU-busy (ms) | gemm | attention | communication | glue |
|---|---:|---:|---:|---:|---:|
| forward | 340.5 | 112.4 | 26.6 | 2.6 | 198.9 |
| backward | 591.6 | 205.1 | 129.1 | 3.3 | 255.2 |

- Step span (first to last kernel of the profiled step): 9677.1 ms; GPU busy 969.1 ms; **GPU idle 8708.0 ms** (90%).
- Forward+backward: busy 932.1 ms (kernel-time sum 933.3 ms; streams barely overlap), idle 8483.7 ms.
- Idle is spread over ~16k gaps, not a few syncs: 7601 gaps of 0.1-1 ms (2562 ms) and 2600 gaps of 1-10 ms (5721 ms); largest single gap 10.7 ms. The step is host-bound under this instrumentation (execution-trace observer, Python stacks and shapes recorded for 16.5k kernels), so idle time says little about an uninstrumented step; percentages below use GPU-busy time (969.1 ms) as the denominator.
- **Attribution coverage:** 100.000% of forward+backward kernel time is attributed to a launching op, phase and module (16,573 GPU activities; 22 FusedAdam kernels of the previous step without a launch record lie before the step window).

## 3. Ranked regions

34 regions. Several are deliberate alternatives that overlap others (marked in `overlaps_with`), so gaps must not be summed blindly: the non-overlapping base partition sums to **347.9 ms**, and the best non-overlapping choice (largest alternatives) to **372.7 ms** (38% of GPU-busy time).

| id | title | phases | scope | kind | inst | measured ms | ideal ms | gap ms | % step | tier | overlaps |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| R-01 | Attention-backward chain: FA dq/dk/dv postprocess + RoPE bwd + QK-norm bwd + concat + MXFP8 dy quantize | backward | cross-module, gemm-adjacent, horizontal | attention epilogue + pointwise + row-norm backward + quantize prologue | 48 | 130.10 | 9.73 | 120.36 | 12.42 | T1 | R-04, R-17 |
| R-02 | wqkv MXFP8 GEMM epilogue: QKV split + QK-norm + RoPE | forward | gemm-adjacent, cross-module | GEMM epilogue (row-norm + pointwise + layout) | 48 | 119.44 | 9.12 | 110.31 | 11.38 | T3 | R-03, R-05, R-19 |
| R-03 | QKV split + QK-norm + RoPE forward (one kernel per layer) | forward | cross-module, horizontal | layout copy + row-norm + pointwise epilogue | 48 | 111.37 | 4.72 | 106.66 | 11.01 | T1 | R-02, R-05, R-19 |
| R-04 | RoPE backward: q/k rotate-half backward chain in one pass | backward | intra-module, horizontal | vertical pointwise chain + q/k sibling fusion | 48 | 103.26 | 4.22 | 99.04 | 10.22 | T0 | R-01 |
| R-05 | RoPE forward: q/k rotate-half chain in one pass | forward | intra-module, horizontal | vertical pointwise chain + q/k sibling fusion | 48 | 101.67 | 4.22 | 97.45 | 10.06 | T0 | R-02, R-03 |
| R-06 | Chunked cross-entropy forward + backward per chunk | forward, backward | forward-backward, intra-module | fused softmax cross-entropy forward+backward | 8 | 35.35 | 2.88 | 32.47 | 3.35 | T1 | R-07 |
| R-07 | lm_head GEMMs + chunked cross-entropy forward/backward (linear cross-entropy) | forward, backward | gemm-adjacent, forward-backward | fused linear cross-entropy | 8 | 63.90 | 31.71 | 32.19 | 3.32 | T3 | R-06 |
| R-08 | Expert weight prep per unshard: fp32->bf16 cast, copy-in, MXFP8 quantize, operand copies | forward, backward | communication-glue, forward-backward, horizontal | cast + quantize + in-place operand write | 96 | 19.72 | 5.24 | 14.48 | 1.49 | T1 |  |
| R-09 | Router + aux-loss backward glue (incl. deterministic gather backward) | backward | cross-module | scatter + row softmax backward + pointwise | 48 | 14.49 | 0.28 | 14.21 | 1.47 | T2 | R-11 |
| R-10 | Router scoring + top-k bookkeeping + aux-loss forward glue | forward | cross-module, intra-module | row softmax + gather/scatter + column reductions | 48 | 13.23 | 0.28 | 12.95 | 1.34 | T1 | R-11 |
| R-11 | Aux-loss forward + backward pair: score gradient computed in the forward pass | forward, backward | forward-backward, intra-module | forward-backward fusion | 48 | 12.51 | 0.23 | 12.28 | 1.27 | T1 | R-09, R-10 |
| R-12 | Cross-layer: MoE combine sum + ffn residual add + next attention_norm + MXFP8 quantize | forward | cross-layer, cross-module, communication-glue, gemm-adjacent | reduction + residual + RMSNorm + quantize | 48 | 22.41 | 10.73 | 11.68 | 1.21 | T1 | R-15, R-27 |
| R-13 | wo dgrad GEMM epilogue computes the flash-attention backward preprocess; fold accumulator zero-fills | backward | cross-module, gemm-adjacent | GEMM epilogue reduction | 48 | 17.25 | 7.30 | 9.96 | 1.03 | T3 |  |
| R-14 | MoE dx combine + router-gate dgrad add + ffn_norm backward + residual-grad add + MXFP8 quantize of the wo grad | backward | cross-module, communication-glue, gemm-adjacent | reduction + gradient accumulation + RMSNorm backward + quantize | 48 | 21.06 | 12.14 | 8.92 | 0.92 | T1 | R-29 |
| R-15 | Dist-MoE forward combine: weighted top-k sum | forward | intra-module, communication-glue | weighted reduction | 48 | 15.67 | 8.38 | 7.29 | 0.75 | T1 | R-12 |
| R-16 | FSDP reduce-scatter glue: chunk_cat, size-1 group copy, fp32 gradient cast | backward | communication-glue | pack + cast-on-copy | 48 | 10.42 | 3.77 | 6.65 | 0.69 | T0 |  |
| R-17 | q_norm/k_norm backward + dqkv concat | backward | cross-module, horizontal | row-norm backward + layout concat | 48 | 13.11 | 6.78 | 6.33 | 0.65 | T1 | R-01 |
| R-18 | FSDP attention-group unshard glue (forward and backward re-unshard) | forward, backward | communication-glue, forward-backward, horizontal | cast + pack (pre-collective) and unpack + quantize (post-collective) | 96 | 6.36 | 1.00 | 5.36 | 0.55 | T1 |  |
| R-19 | QKV split/contiguous copies + q_norm + k_norm forward | forward | cross-module, horizontal | layout copy + row-norm fusion | 48 | 9.71 | 4.69 | 5.02 | 0.52 | T1 | R-02, R-03 |
| R-20 | Dist-MoE forward routing-plan metadata kernels | forward | communication-glue, intra-module | metadata pipeline fusion | 48 | 4.19 | 0.16 | 4.03 | 0.42 | T2 |  |
| R-21 | Dist-MoE backward: broadcast and scale dy into [T,K,D] with score gradients | backward | intra-module, communication-glue | broadcast + dot-product reduction | 48 | 19.68 | 15.84 | 3.84 | 0.40 | T1 |  |
| R-22 | HiMidLo gate weight gradient: three per-piece GEMMs + adds -> one GEMM | backward | horizontal, gemm-adjacent | horizontal GEMM fusion with epilogue accumulation | 48 | 5.08 | 1.28 | 3.79 | 0.39 | T3 | R-26 |
| R-23 | Forward-backward: column-wise MXFP8 of the wqkv/wo inputs produced in the forward | backward | forward-backward, gemm-adjacent | forward producer emits backward operand | 48 | 4.99 | 1.44 | 3.55 | 0.37 | T1 |  |
| R-24 | attention_norm backward + residual-grad add | backward | cross-module | RMSNorm backward with fused residual-grad accumulation | 48 | 6.61 | 3.72 | 2.88 | 0.30 | T1 |  |
| R-25 | Attention residual add + ffn_norm + Dist-MoE dispatch copy | forward | cross-module, communication-glue | residual add + RMSNorm writing into the dispatch buffer | 48 | 6.59 | 3.77 | 2.83 | 0.29 | T1 | R-30 |
| R-26 | HiMidLo router-gate backward glue: bf16 piece split, concats, weight-grad adds | backward | intra-module, gemm-adjacent | pointwise split + layout concat | 48 | 2.90 | 0.33 | 2.57 | 0.27 | T0 | R-22 |
| R-27 | Cross-layer: ffn residual add + next attention_norm + MXFP8 quantize of the wqkv input | forward | cross-layer, cross-module, gemm-adjacent | residual add + RMSNorm + quantize prologue | 48 | 6.74 | 4.20 | 2.53 | 0.26 | T1 | R-12 |
| R-28 | MXFP8 quantize of the attention output in the flash-attention forward epilogue | forward | gemm-adjacent | attention epilogue quantize | 48 | 3.37 | 0.96 | 2.41 | 0.25 | T3 |  |
| R-29 | Router gate dgrad GEMM accumulates into the expert input gradient (beta=1 epilogue) | backward | gemm-adjacent, cross-module | GEMM epilogue accumulate (C += A*B) | 48 | 4.26 | 2.05 | 2.21 | 0.23 | T3 | R-14 |
| R-30 | wo MXFP8 GEMM epilogue adds the attention residual | forward | gemm-adjacent, cross-module | GEMM epilogue residual add | 48 | 9.13 | 7.30 | 1.83 | 0.19 | T3 | R-25 |
| R-31 | Dist-MoE backward plan/metadata kernels | backward | communication-glue, forward-backward | metadata pipeline fusion / reuse of forward plan | 48 | 1.58 | 0.07 | 1.51 | 0.16 | T2 |  |
| R-32 | Root FSDP groups (tok_embeddings, norm+lm_head): unshard and reduce-scatter glue | forward, backward | communication-glue | copy elimination / cast-on-copy | 2 | 1.55 | 0.32 | 1.23 | 0.13 | T0 |  |
| R-33 | Embedding backward (deterministic sort-based scatter-add) | backward | intra-module | sorted segment reduction | 1 | 0.53 | 0.11 | 0.42 | 0.04 | T2 |  |
| R-34 | Last-layer residual add + final norm forward, final norm backward | forward, backward | cross-module | residual add + RMSNorm (fwd), RMSNorm backward | 1 | 0.19 | 0.15 | 0.03 | 0.00 | T1 |  |

## 4. Region details

### R-01 Attention-backward chain: FA dq/dk/dv postprocess + RoPE bwd + QK-norm bwd + concat + MXFP8 dy quantize
- **What:** From the flash-attention fp32 dq/dk/dv accumulators to the MXFP8 (row- and column-wise) wqkv output gradient: accumulator conversion, RoPE backward, q/k RMSNorm backward, the dqkv cat and the 1x32/32x1 quantization with scale swizzles.
- **Where:** `torchtitan/models/common/attention/attention.py:180-197,287-324; torchtitan/models/common/rope.py:455-482; torchtitan/quantization/mxfp8/linear.py:233-300`; 33.0 kernels x 48 instances; dq_accum [32768,32,128] fp32, dk/dv accum [4,6553600] fp32, saved xq,xk bf16, rstd -> dqkv fp8 rowwise + colwise [32768,5120] + scales.
- **Numbers:** 2710.3 us measured vs 202.8 us ideal per instance (1403.5 MB external); gap 120.36 ms.
- **Dataflow:** Each stage consumes the previous stage row by row (token, head) and the only consumer of the bf16 dqkv is the MXFP8 quantizer feeding the wqkv dgrad/wgrad GEMMs; the 32x1 column-wise blocks span 32 tokens, so the kernel tiles 32 tokens x full row.
- **Blockers:** The FA postprocess kernels are inside the flash_attn library call; RMSNorm weight-grad reduction; the quantizer lives inside the MXFP8Linear backward.
- **Tier T1:** The FA postprocess (scale + bf16 round) and RoPE are pointwise; the bf16 rounding points of the eager chain must be kept; RMSNorm backward reductions; amax is order-insensitive.
- **Notes:** Alternative to taking R-04 and R-17 separately; also absorbs the FA postprocess (7.6 ms) and wqkv dy quantization (6.6 ms). Member kernels: 1584; ideal from 1403.5 MB external bytes per instance.

### R-02 wqkv MXFP8 GEMM epilogue: QKV split + QK-norm + RoPE
- **What:** The wqkv GEMM [32768,2048]x[2048,5120] epilogue writes contiguous q/k/v directly, applying q_norm/k_norm (128-wide rows fit one output tile) and RoPE, so the [32768,5120] qkv tensor is never materialized.
- **Where:** `torchtitan/quantization/mxfp8/linear.py:77-200; torchtitan/models/common/attention/attention.py:169-197,287-306; torchtitan/models/common/rope.py:455-482`; 24.0 kernels x 48 instances; x fp8 [32768,2048] x W fp8 [2048,5120] -> q [32768,32,128], k,v [32768,4,128] bf16; rstd fp32.
- **Numbers:** 2488.3 us measured vs 190.1 us ideal per instance (424.4 MB external); gap 110.31 ms.
- **Dataflow:** Each 128-column slice of the GEMM output tile is one (kv-group, slot) head; the norm needs only that slice and RoPE only its row position, so the whole chain is a per-tile epilogue.
- **Blockers:** Needs a custom epilogue on the cuBLAS/nvjet MXFP8 GEMM; the RMSNorm backward needs its pre-norm input, which then must be recomputed or written as an extra output.
- **Tier T3:** The GEMM output must be rounded to bf16 before the norm exactly as the eager GEMM writes it; RMSNorm reduction order over each 128-column head slice.
- **Notes:** Extends the QKV split + QK-norm + RoPE region by also removing the qkv write and re-read. Member kernels: 1152; ideal from 424.4 MB external bytes and 687.2 GFLOP per instance.

### R-03 QKV split + QK-norm + RoPE forward (one kernel per layer)
- **What:** Merge of the QKV split/contiguous copies, q_norm/k_norm and RoPE: read the wqkv output once and write rotated, normalized q, k and contiguous v.
- **Where:** `torchtitan/models/common/attention/attention.py:169-197,287-306; torchtitan/models/common/rope.py:455-482`; 23.0 kernels x 48 instances; qkv [32768,5120] bf16 -> q [32768,32,128], k,v [32768,4,128] bf16; rstd fp32.
- **Numbers:** 2320.3 us measured vs 98.3 us ideal per instance (680.3 MB external); gap 106.66 ms.
- **Dataflow:** RoPE consumes exactly the normalized q/k rows produced by q_norm/k_norm, row by row, so the normalized values never need to reach HBM. Backward needs only rstd plus the pre-norm q/k (a view of qkv) and the cos/sin cache.
- **Blockers:** Same as the two parts; RMSNorm backward would have to read its saved input from the qkv buffer.
- **Tier T1:** Inherits the RMSNorm reduction order constraint of the QK-norm part; RoPE part is pointwise.
- **Notes:** Alternative to taking the QKV+QK-norm and RoPE regions separately; saves one more [32768,4608] bf16 round trip. Member kernels: 1104; ideal from 680.3 MB external bytes per instance.

### R-04 RoPE backward: q/k rotate-half backward chain in one pass
- **What:** Backward of CosSinRoPE for dq [32768,32,128] and dk [32768,4,128]: bf16->fp32 cast, two muls, neg, slice-backward zero-fill + copies, two adds, fp32->bf16 cast, 22 kernels per layer.
- **Where:** `torchtitan/models/common/rope.py:455-482`; 22.0 kernels x 48 instances; dq [32768,32,128] bf16, dk [32768,4,128] bf16 -> bf16 (fp32 math); cos/sin cache [4096,256] fp32.
- **Numbers:** 2151.2 us measured vs 87.9 us ideal per instance (608.4 MB external); gap 99.04 ms.
- **Dataflow:** The transpose of rotate_half is again a fixed permutation within the head dim; every output element depends on two input elements of the same row and the cache row.
- **Blockers:** None structural; disabled "rope" local_compile region. Slice backward materializes zero-filled fp32 [32768,32,128] buffers twice.
- **Tier T0:** Pointwise: grad*cos + rotate_half^T(grad*sin) in fp32 with the eager rounding order (autograd accumulates the two branches with add_).
- **Notes:** Member kernels: 1056; ideal from 608.4 MB external bytes per instance.

### R-05 RoPE forward: q/k rotate-half chain in one pass
- **What:** CosSinRoPE on q [32768,32,128] and k [32768,4,128]: upcast to fp32, x*cos + rotate_half(x)*sin, cast back to bf16, plus the position bound check and cache gather, run as 18 separate kernels per layer.
- **Where:** `torchtitan/models/common/rope.py:25-37,455-482,485-505`; 18.0 kernels x 48 instances; q [32768,32,128] bf16, k [32768,4,128] bf16 -> fp32 math -> bf16; cos/sin cache [4096,256] fp32 gathered by positions [32768] int64.
- **Numbers:** 2118.1 us measured vs 87.9 us ideal per instance (608.4 MB external); gap 97.45 ms.
- **Dataflow:** Every op is elementwise over (token, head, dim) except rotate_half, which is a fixed permutation within the 128-wide head dim (neg + cat of halves); the cos/sin row depends only on the token position. Each element of q_out/k_out depends on two elements of the same input row and one cache row, so one pass reading q,k once and writing once is legal. q and k share the cache row (sibling fusion).
- **Blockers:** None structural. This model has a "rope" local_compile region that is disabled in this capture; eager runs every op as its own kernel with fp32 intermediates (q fp32 copy, rotated copy, two products, sum).
- **Tier T0:** Pure pointwise: two fp32 muls and one fp32 add per element, then a bf16 round; the fused kernel must keep the same op order and avoid FMA contraction (x*cos + y*sin evaluated as two rounded products then a rounded add).
- **Notes:** Intermediates are fp32 [32768,32,128] (537 MB each), which is why eager costs ~2.1 ms/layer against an 88 us roofline. Member kernels: 864; ideal from 608.4 MB external bytes per instance.

### R-06 Chunked cross-entropy forward + backward per chunk
- **What:** Per 4096-token chunk: fp32 upcast of the bf16 logits [4096,151936], log_softmax, nll, then immediately nll/log_softmax backward, zero-fill and bf16 downcast of the logits gradient, plus the gradient-accumulator writes.
- **Where:** `torchtitan/components/loss.py:54-90,620-675,900-1000`; 14.38 kernels x 8 instances; logits [4096,151936] bf16 x 8 chunks -> loss scalar; grad logits [4096,151936] bf16; grad accumulator [32768,2048] bf16.
- **Numbers:** 4418.7 us measured vs 359.7 us ideal per instance (2489.4 MB external); gap 32.47 ms.
- **Dataflow:** Backward runs right after forward on the same chunk (chunk_loss.backward()); the gradient needs only the logits row and its logsumexp, so one kernel can read logits once and write grad once. The gradient accumulator zero-fill is redundant since every slice is overwritten.
- **Blockers:** Inner loss and lm_head are separate modules; the "loss" local_compile region is disabled. T3 extension: fuse with the lm_head GEMMs (linear cross-entropy).
- **Tier T1:** Row max/logsumexp over 151936 fp32 values must follow the cunn_SoftMax reduction order; gradient = softmax - onehot, scaled, rounded to bf16.
- **Notes:** ~4.4 ms per chunk against a 0.36 ms roofline. Member kernels: 115; ideal from 2489.4 MB external bytes per instance.

### R-07 lm_head GEMMs + chunked cross-entropy forward/backward (linear cross-entropy)
- **What:** Per 4096-token chunk, the lm_head forward GEMM, the cross-entropy forward/backward glue and the lm_head dgrad/wgrad GEMMs fused so the [4096,151936] logits and their gradient never reach HBM.
- **Where:** `torchtitan/components/loss.py:54-90,900-1000`; 17.38 kernels x 8 instances; h chunk [4096,2048] bf16, W [151936,2048] bf16 -> loss, dh [4096,2048] bf16, dW [151936,2048] (accumulated).
- **Numbers:** 7986.9 us measured vs 3963.2 us ideal per instance (1900.6 MB external); gap 32.19 ms.
- **Dataflow:** The logits are consumed only by the loss and its gradient only by the two lm_head backward GEMMs; an online-softmax formulation over vocab tiles needs logits only in registers.
- **Blockers:** Bitwise parity with cuBLAS GEMM outputs; lm_head is an FSDP root module with chunked backward hooks.
- **Tier T3:** Logit values come out of a GEMM; matching eager bitwise requires the same GEMM tiling/accumulation and the same logsumexp order.
- **Notes:** GEMMs run at the bf16 peak, so the gap equals the cross-entropy glue gap. Member kernels: 139; ideal from 1900.6 MB external bytes and 7647.2 GFLOP per instance.

### R-08 Expert weight prep per unshard: fp32->bf16 cast, copy-in, MXFP8 quantize, operand copies
- **What:** Per layer and per unshard (forward and backward), the local experts w13 [16,1536,2048] and w2 [16,2048,768] fp32 are cast to bf16, copied through the size-1 FSDP group buffer, quantized to MXFP8 with 32x32 scales by Dist-MoE, and copied into the persistent operand buffers.
- **Where:** `torchtitan/models/common/dist_moe/mxfp8.py:64-120,170-220; torchtitan/quantization/_fsdp_tensor.py:352-411`; 12.0 kernels x 96 instances; w13 [16,1536,2048] + w2 [16,2048,768] fp32 -> bf16 -> fp8 e4m3 + e8m0 scales ([24576,64],[32768,48],[32768,24],[12288,64]).
- **Numbers:** 205.4 us measured vs 54.6 us ideal per instance (377.6 MB external); gap 14.48 ms.
- **Dataflow:** The bf16 copy exists only to feed the quantizer; the expert FSDP group has size 1 so the all-gather is a no-op copy. One kernel can read fp32 weights and write fp8 data + both scale layouts directly into the operand buffers. Expert weights do not change within a step, so the backward instance repeats identical work.
- **Blockers:** FSDP2 mixed-precision cast and copy-in are generic; dist_moe.prepare_mxfp8_weight takes bf16 input and returns new tensors that are then copied. Skipping the backward re-quantization means keeping 80 MB/layer of fp8 expert weights alive across the step.
- **Tier T1:** Cast then 32x32 amax quantization; amax is order-insensitive.
- **Notes:** 96 instances = 48 forward + 48 backward unshards. Member kernels: 1152; ideal from 377.6 MB external bytes per instance.

### R-09 Router + aux-loss backward glue (incl. deterministic gather backward)
- **What:** Backward of top-k score gather (deterministic index_put with radix sort, bounds asserts), route normalization, softmax and the load-balance loss: 62 kernels per layer over [32768,128] and [32768,8].
- **Where:** `torchtitan/models/common/moe/moe.py:233-330,662-705; torchtitan/models/common/aux_loss.py:141-220`; 62.0 kernels x 48 instances; d topk scores [32768,8] fp32, saved scores [32768,128] fp32, ids [32768,8] int64 -> d gate logits [32768,128] fp32.
- **Numbers:** 301.9 us measured vs 5.9 us ideal per instance (40.9 MB external); gap 14.21 ms.
- **Dataflow:** Every op is row-local in tokens except the aux-loss column statistics (already reduced in forward); one row-parallel kernel can scatter the 8 gradients, add the aux-loss gradient and apply the softmax backward.
- **Blockers:** Deterministic mode makes index_put sort-based (~25 kernels); the bounds asserts come from index validation.
- **Tier T2:** The gather backward is a scatter-add; with unique expert ids per token each target is written once, so a row-wise scatter is exact, but matching the eager accumulation order of the aux-loss gradient and gather gradient (add_) is required.
- **Notes:** Member kernels: 2976; ideal from 40.9 MB external bytes per instance.

### R-10 Router scoring + top-k bookkeeping + aux-loss forward glue
- **What:** Softmax over gate logits, round-robin expert ids, gather of top-k scores, normalization, routing-map scatter, expert counts, the batch-wise load-balance loss statistics and the padding masked_fill: 45 small kernels per layer on [32768,128] and [32768,8] tensors.
- **Where:** `torchtitan/models/common/moe/moe.py:233-330,339-361,662-705; torchtitan/models/common/aux_loss.py:141-200`; 45.0 kernels x 48 instances; gate logits [32768,128] fp32 -> scores [32768,128] fp32, ids [32768,8] int64, topk scores [32768,8] fp32, routing map [32768,128] bool, counts [128] fp32.
- **Numbers:** 275.7 us measured vs 5.9 us ideal per instance (40.9 MB external); gap 12.95 ms.
- **Dataflow:** All ops consume the [32768,128] gate output or the [32768,8] top-k tensors; row-wise ops (softmax, gather, normalize, scatter, L1 normalize) fuse into one pass, and the column sums can be accumulated in the same pass as partial sums.
- **Blockers:** The expert-count all-reduce (DP/EP) splits the region into a pre- and post-collective kernel; the deterministic column reductions over T are long and narrow (128 outputs), so a fused kernel needs a fixed-order tree reduction to stay bitwise.
- **Tier T1:** Row softmax and column sums over 32768 tokens (expert counts, normalized-score sums) must keep the eager reduction order; the count all-reduce sits between the count and loss stages.
- **Notes:** The two column sums alone (bool->long sum and fp32 prob sums) take ~130 us/layer for 4-17 MB of input. Extension: fold the softmax into the gate GEMM epilogue (T3). Member kernels: 2160; ideal from 40.9 MB external bytes per instance.

### R-11 Aux-loss forward + backward pair: score gradient computed in the forward pass
- **What:** The batch-wise load-balance loss forward statistics and its backward (division, norm and clamp backward over [32768,128] scores) computed together, since all backward inputs are known after the forward count all-reduce.
- **Where:** `torchtitan/models/common/moe/moe.py:662-705; torchtitan/models/common/aux_loss.py:141-220`; 41.0 kernels x 48 instances; scores [32768,128] fp32, routing map bool -> loss scalar and dLoss/dscores [32768,128] fp32.
- **Numbers:** 260.6 us measured vs 4.8 us ideal per instance (33.6 MB external); gap 12.28 ms.
- **Dataflow:** The injected gradient depends only on scores, routing counts (after the all-reduce), the denominator and coeff, all available at forward time; computing it once in the forward and adding it in the router backward removes ~20 backward kernels.
- **Blockers:** AuxLoss.inject relies on autograd to deliver the gradient; computing it eagerly needs a custom autograd Function and stores an extra [32768,128] fp32 tensor.
- **Tier T1:** Column reductions over T and row L1 norms must keep the eager order; the backward chain (DivBackward, LinalgVectorNormBackward, ClampMinBackward) must be replicated op by op.
- **Notes:** Overlaps the router forward and router backward regions. Member kernels: 1968; ideal from 33.6 MB external bytes per instance.

### R-12 Cross-layer: MoE combine sum + ffn residual add + next attention_norm + MXFP8 quantize
- **What:** The weighted top-k combine of layer i-1 feeding its residual add, layer i attention_norm and the wqkv-input quantization, without materializing the MoE output.
- **Where:** `dist_moe/_postprocess.py:502-551; torchtitan/models/qwen3/model.py:57-73; torchtitan/models/common/norm.py:36-51`; 4.98 kernels x 48 instances; [32768,8,2048] bf16 + scores, x [32768,2048] bf16 -> x, normed bf16, fp8 + scales.
- **Numbers:** 466.8 us measured vs 223.5 us ideal per instance (1546.6 MB external); gap 11.68 ms.
- **Dataflow:** The MoE output row is consumed only by the residual add, whose row is consumed by the next norm and quantizer: a row-parallel kernel can do all of it.
- **Blockers:** Crosses the Dist-MoE library boundary and the FSDP block boundary.
- **Tier T1:** K-sum order, RMSNorm order.
- **Notes:** Alternative to taking R-27 and R-15 separately. Member kernels: 239; ideal from 1546.6 MB external bytes per instance.

### R-13 wo dgrad GEMM epilogue computes the flash-attention backward preprocess; fold accumulator zero-fills
- **What:** The wo dgrad GEMM writes dO [32768,4096]; flash-attention backward then re-reads dO and O to compute rowsum(dO*O) per head and zero-fills its fp32 dq/dk/dv accumulators and semaphores.
- **Where:** `torchtitan/models/common/attention/attention.py:318-324; torchtitan/quantization/mxfp8/linear.py:233-300`; 9.0 kernels x 48 instances; dy fp8 [32768,2048] x W fp8 [2048,4096] -> dO [32768,4096] bf16; O [32768,4096] bf16 -> D [32,32768] fp32; accumulators [4,6553600] fp32 x2, [32,32768] fp32.
- **Numbers:** 359.4 us measured vs 152.1 us ideal per instance (832.6 MB external); gap 9.96 ms.
- **Dataflow:** Each 128-column slice of a dO output tile is one head; the epilogue has dO in registers and can load the matching O tile and reduce. The zero-fills have no inputs and can be issued by the same kernel.
- **Blockers:** Requires a custom epilogue on the cuBLAS/nvjet MXFP8 GEMM and passing D into the flash-attention backward (library API change).
- **Tier T3:** rowsum(dO*O) over 128 elements per head in fp32 must match the FA preprocess order, computed from the bf16-rounded dO.
- **Notes:** The GEMM alone runs at the measured mxfp8 peak; gap is the preprocess + zero-fills. Member kernels: 432; ideal from 832.6 MB external bytes and 549.8 GFLOP per instance.

### R-14 MoE dx combine + router-gate dgrad add + ffn_norm backward + residual-grad add + MXFP8 quantize of the wo grad
- **What:** The top-k sum of the expert input gradients, the add of the router gate input gradient, the ffn_norm RMSNorm backward, the residual gradient accumulation, and the 1x32/32x1 quantization of the result for the wo backward GEMMs.
- **Where:** `dist_moe/_postprocess.py:502-551; torchtitan/models/qwen3/model.py:57-73; torchtitan/quantization/mxfp8/linear.py:233-300`; 8.0 kernels x 48 instances; d route [32768,8,2048] bf16 + scores, d gate-input [32768,2048] bf16, saved x bf16, rstd, residual grad [32768,2048] bf16 -> grad x [32768,2048] bf16 + fp8 rowwise/colwise + scales.
- **Numbers:** 438.7 us measured vs 252.9 us ideal per instance (1750.2 MB external); gap 8.92 ms.
- **Dataflow:** The ffn_norm output gradient is the sum of two producers (experts and gate) and is consumed only by the norm backward; the norm backward output plus the residual gradient is the grad of the attention residual stream, consumed by the wo backward (quantized) and passed down the residual.
- **Blockers:** The expert dx sum is inside the Dist-MoE backward custom op; autograd performs the gradient accumulation as separate add_ kernels; ffn_norm weight-grad column reduction.
- **Tier T1:** K-sum order, autograd accumulation order (experts first, then gate), RMSNorm backward reductions; amax is order-insensitive.
- **Notes:** Member kernels: 384; ideal from 1750.2 MB external bytes per instance.

### R-15 Dist-MoE forward combine: weighted top-k sum
- **What:** Reduces the expert outputs scattered into [32768,8,2048] bf16 by the router scores into the MoE output [32768,2048].
- **Where:** `dist_moe/_postprocess.py:502-551; dist_moe/kernels/triton/broadcast_n_reduction.py:190`; 1.0 kernels x 48 instances; route outputs [32768,8,2048] bf16, scores [32768,8] fp32 -> out [32768,2048] bf16.
- **Numbers:** 326.5 us measured vs 174.7 us ideal per instance (1209.0 MB external); gap 7.29 ms.
- **Dataflow:** Single memory-bound reduction; runs at 54% of HBM roofline.
- **Blockers:** The [T,K,D] buffer is written by remote ranks' combine GEMM epilogues (symmetric memory); folding the K-sum into those epilogues would need cross-rank atomics (non-deterministic).
- **Tier T1:** Sum over K=8 in fixed order with fp32 accumulation.
- **Notes:** The zero-fill that precedes it in this capture is a fake-PG artifact (see no_opportunity). Member kernels: 48; ideal from 1209.0 MB external bytes per instance.

### R-16 FSDP reduce-scatter glue: chunk_cat, size-1 group copy, fp32 gradient cast
- **What:** Per layer: chunk_cat of the expert gradients and attention-group gradients into reduce-scatter buffers, the size-1 expert group copy, and casts of the reduced shards to fp32.
- **Where:** `torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:300-319`; 7.0 kernels x 48 instances; expert grads [16,2,768,2048]+[16,2048,768] bf16 -> fp32 shard; attention grads -> bf16 [8,2392608] -> fp32 shard.
- **Numbers:** 217.2 us measured vs 78.6 us ideal per instance (543.9 MB external); gap 6.65 ms.
- **Dataflow:** For the size-1 expert group, chunk_cat -> copy -> cast is one bf16->fp32 cast of the gradients; for the attention group the post-reduce cast can be folded into the reduce-scatter output copy.
- **Blockers:** FSDP2 generic reduce-scatter path; T3 extension: wgrad GEMMs writing directly into the reduce-scatter buffer.
- **Tier T0:** Copies and casts only.
- **Notes:** Member kernels: 336; ideal from 543.9 MB external bytes per instance.

### R-17 q_norm/k_norm backward + dqkv concat
- **What:** RMSNorm backward for q and k heads (with their weight-grad row reductions) and the SplitWithSizes backward cat that reassembles dq/dk/dv into the [32768,5120] wqkv gradient.
- **Where:** `torchtitan/models/common/attention/attention.py:180-197,296-302`; 5.0 kernels x 48 instances; dq,dk bf16 + saved xq,xk bf16 + rstd fp32, dv [32768,4,128] bf16 -> dqkv [32768,4,10,128] bf16; dweight [128] fp32.
- **Numbers:** 273.1 us measured vs 141.3 us ideal per instance (977.8 MB external); gap 6.33 ms.
- **Dataflow:** q_norm/k_norm backward outputs are consumed only by the cat; writing directly into the strided dqkv slots removes the cat.
- **Blockers:** Weight-grad column reduction across 32768x32 rows needs a second pass or a fixed-order cross-CTA reduction.
- **Tier T1:** RMSNorm backward row reduction and the [128] weight-grad column reduction order.
- **Notes:** Member kernels: 240; ideal from 977.8 MB external bytes per instance.

### R-18 FSDP attention-group unshard glue (forward and backward re-unshard)
- **What:** Per layer and per unshard: fp32->bf16 casts of the wqkv/wo shards, copy-in to the all-gather buffer, copy-out split, MXFP8 32x32 weight quantization of wqkv [5120,2048] and wo [2048,4096] with swizzled scales, and copies of the fp8 data/scales into FSDP-managed operand buffers.
- **Where:** `torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:36,259-294; torchtitan/quantization/_fsdp_tensor.py:352-411; torchtitan/quantization/mxfp8/tensor.py:71-110`; 14.0 kernels x 96 instances; fp32 shards [640,2048], [256,4096] + small params -> bf16 [8,2392608] all-gather buffer -> fp8 wqkv [5120,2048], wo [2048,4096] + 32x32 e8m0 scales.
- **Numbers:** 66.3 us measured vs 10.4 us ideal per instance (72.1 MB external); gap 5.36 ms.
- **Dataflow:** Pre-collective: the cast output is only copied into the all-gather input buffer (cast directly into it). Post-collective: the gathered bf16 weights are only quantized and the quantized operands are copied into persistent buffers (quantize directly into them; skip the bf16 unsharded copy-out for wqkv/wo).
- **Blockers:** FSDP2 owns copy-in/copy-out and the all-gather boundary; the operand copy exists because _build_operands refills FSDP-managed buffers. The backward instance re-does the forward work because blocks reshard after forward (keeping fp8 operands costs ~20 MB/layer).
- **Tier T1:** Cast and quantize are exact functions of the inputs; 32x32 amax is order-insensitive.
- **Notes:** 96 instances = 48 forward + 48 backward unshards. Member kernels: 1344; ideal from 72.1 MB external bytes per instance.

### R-19 QKV split/contiguous copies + q_norm + k_norm forward
- **What:** The wqkv output [32768,5120] is split into q/k/v and copied contiguous (3 copies), then q_norm and k_norm (RMSNorm over head_dim 128) run as separate kernels.
- **Where:** `torchtitan/models/common/attention/attention.py:169-197,287-302; torchtitan/models/common/norm.py:36-51`; 5.0 kernels x 48 instances; qkv [32768,5120] bf16 viewed [32768,4,10,128] -> q [32768,32,128], k,v [32768,4,128] bf16; norm weights [128] bf16; rstd [32768,36] fp32.
- **Numbers:** 202.2 us measured vs 97.6 us ideal per instance (675.8 MB external); gap 5.02 ms.
- **Dataflow:** q_norm/k_norm read only the split views of qkv; v needs only a strided copy. A single kernel can read each qkv row once and emit normalized q, k and contiguous v. q_norm and k_norm are siblings over the same input row.
- **Blockers:** The contiguous copies exist for vLLM KV-cache layout compatibility (comment in QKVLinear.forward); a fused kernel must still produce contiguous outputs. The RMSNorm backward saves its input, which would become a view of qkv.
- **Tier T1:** RMSNorm reduces over 128 elements per (token, head); the fused kernel must reproduce the quack RMSNorm reduction order.
- **Notes:** Member kernels: 240; ideal from 675.8 MB external bytes per instance.

### R-20 Dist-MoE forward routing-plan metadata kernels
- **What:** Per layer, ten small Dist-MoE kernels build the expert-parallel dispatch plan from the gathered top-k expert ids (routing split, prefix, combine, metadata gather, block-scaled plan, two GEMM scheduler preparations, small cat/copies).
- **Where:** `dist_moe/kernels/triton/dist_dispatch_routing_kernel.py:421-470; dist_moe/_blockscaled.py:1318,3369-3483`; 10.0 kernels x 48 instances; expert ids [8,32768,8] int64 (all EP ranks) -> gather/scatter index tables per local expert, token counts.
- **Numbers:** 87.3 us measured vs 3.3 us ideal per instance (23.1 MB external); gap 4.03 ms.
- **Dataflow:** Each stage consumes the previous stage's counts/offsets; routing split -> prefix -> combine is a classic histogram/scan/scatter that fits one persistent kernel with grid-wide sync or a decoupled look-back scan.
- **Blockers:** Inside the dist_moe library; the stages are separated by cross-rank metadata exchange in the real job (gathered ids), and the routing split kernel itself is 55 us of the 87 us.
- **Tier T2:** Slot assignment uses prefix sums over tokens; outputs are integer indices, so bitwise equality requires the same slot order (order-sensitive data movement).
- **Notes:** With round-robin routing the plan is identical every layer and every step; caching it would remove the region (configuration-specific). Member kernels: 480; ideal from 23.1 MB external bytes per instance.

### R-21 Dist-MoE backward: broadcast and scale dy into [T,K,D] with score gradients
- **What:** Broadcasts the MoE output gradient to every routed slot scaled by its router score and computes the score gradient as a dot product with the saved route outputs.
- **Where:** `dist_moe/_postprocess.py:552-616; dist_moe/kernels/triton/broadcast_n_reduction.py:386-533`; 1.0 kernels x 48 instances; dy [32768,2048] bf16, scores [32768,8] fp32, saved route outputs [32768,8,2048] bf16 -> d route [32768,8,2048] bf16, d scores [32768,8] fp32.
- **Numbers:** 409.9 us measured vs 330.0 us ideal per instance (2283.8 MB external); gap 3.84 ms.
- **Dataflow:** Single kernel already; 80% of roofline. T3 extension: apply the per-row score in the dgrad dispatch GEMM prologue so the 1.07 GB scaled-dy buffer is never written.
- **Blockers:** The dgrad dispatch GEMM reads the scaled gradient through remote symmetric memory; scaling in its prologue is a Dist-MoE kernel change.
- **Tier T1:** Score gradient is a 2048-long dot product per (token, k); order must match.
- **Notes:** Member kernels: 48; ideal from 2283.8 MB external bytes per instance.

### R-22 HiMidLo gate weight gradient: three per-piece GEMMs + adds -> one GEMM
- **What:** The router gate weight gradient is computed as three [128,32768]x[32768,2048] split-K GEMMs (one per bf16 piece) with split-K reduce kernels, then two fp32 adds and a bf16 cast.
- **Where:** `torchtitan/models/common/linear/hi_mid_lo_linear.py:363-377`; 9.0 kernels x 48 instances; pieces 3x[32768,128] bf16, x [32768,2048] bf16 -> wgrad [128,2048] fp32 -> bf16.
- **Numbers:** 105.8 us measured vs 26.7 us ideal per instance (159.9 MB external); gap 3.79 ms.
- **Dataflow:** All three GEMMs read the same x; one GEMM over K=3x32768 (or one kernel with three accumulators) reads x once.
- **Blockers:** Bitwise equality with the eager per-piece rounding; cuBLAS split-K heuristics.
- **Tier T3:** Accumulating the three pieces inside one GEMM changes the fp32 rounding order relative to three separately rounded GEMM results plus two adds.
- **Notes:** Overlaps the HiMidLo glue region on the adds and cast. Member kernels: 432; ideal from 159.9 MB external bytes and 51.5 GFLOP per instance.

### R-23 Forward-backward: column-wise MXFP8 of the wqkv/wo inputs produced in the forward
- **What:** The wqkv and wo backward re-read the saved bf16 inputs (normed x [32768,2048], attention output [32768,4096]) to quantize them column-wise (32x1) for the weight-gradient GEMMs.
- **Where:** `torchtitan/quantization/mxfp8/linear.py:138-170,290-310`; 4.0 kernels x 48 instances; saved x [32768,2048] bf16 -> fp8 colwise + scales; saved O [32768,4096] bf16 -> fp8 colwise + scales.
- **Numbers:** 103.9 us measured vs 30.0 us ideal per instance (207.6 MB external); gap 3.55 ms.
- **Dataflow:** The column-wise quantization depends only on forward values; the forward norm (R-27) and attention epilogue (R-28) already hold these tiles.
- **Blockers:** The capture saves inputs in bf16 (96 bf16, 0 mxfp8 saved formats); for the attention output the bf16 copy must stay for the flash-attention backward, so this adds 138 MB/layer of saved fp8.
- **Tier T1:** 32x1 amax is order-insensitive; requires the forward producers (attention_norm, attention epilogue) to emit the column-wise blocks.
- **Notes:** Ideal = incremental forward writes of the column-wise operands. Member kernels: 192; ideal from 207.6 MB external bytes per instance.

### R-24 attention_norm backward + residual-grad add
- **What:** RMSNorm backward of attention_norm and the autograd add_ that accumulates it into the residual-stream gradient.
- **Where:** `torchtitan/models/qwen3/model.py:57-60; torchtitan/models/common/norm.py:36-51`; 3.0 kernels x 48 instances; d normed [32768,2048] bf16, saved x bf16, rstd fp32, residual grad [32768,2048] bf16 -> grad x [32768,2048] bf16.
- **Numbers:** 137.6 us measured vs 77.6 us ideal per instance (537.0 MB external); gap 2.88 ms.
- **Dataflow:** The norm backward output is consumed only by the accumulation add_.
- **Blockers:** Autograd accumulation is a separate kernel; T3 extension into the wqkv dgrad GEMM epilogue would also remove the d normed round trip.
- **Tier T1:** RMSNorm backward row reduction; the add of the residual grad happens after the bf16 rounding of the norm grad.
- **Notes:** Member kernels: 144; ideal from 537.0 MB external bytes per instance.

### R-25 Attention residual add + ffn_norm + Dist-MoE dispatch copy
- **What:** x + attn_out, the ffn_norm RMSNorm, and the Dist-MoE kernel that copies the normalized tokens (and routing ids/scores) into the expert-parallel dispatch buffer.
- **Where:** `torchtitan/models/qwen3/model.py:57-67; torchtitan/models/common/dist_moe/routed_experts.py:166-243; dist_moe/kernels/triton/conditional_copy.py:303`; 3.0 kernels x 48 instances; x, attn_out [32768,2048] bf16 -> x [32768,2048] bf16, normed [32768,2048] bf16 in the symmetric dispatch buffer; ids/scores [32768,8].
- **Numbers:** 137.4 us measured vs 78.5 us ideal per instance (543.2 MB external); gap 2.83 ms.
- **Dataflow:** The normed tokens are consumed by the router gate GEMM and copied verbatim into the dispatch buffer; writing them once directly into the dispatch buffer removes one [32768,2048] read+write.
- **Blockers:** Dist-MoE owns the symmetric-memory dispatch buffer and its conditional (activation-slot) placement; the router gate GEMM and saved activations would read from that buffer.
- **Tier T1:** RMSNorm row reduction order.
- **Notes:** Member kernels: 144; ideal from 543.2 MB external bytes per instance.

### R-26 HiMidLo router-gate backward glue: bf16 piece split, concats, weight-grad adds
- **What:** Splits the fp32 gate gradient [32768,128] into three bf16 pieces (casts, shift, subtracts), concatenates pieces and weight copies for the stacked GEMMs, adds the per-piece weight gradients and casts to bf16.
- **Where:** `torchtitan/models/common/linear/hi_mid_lo_linear.py:342-415`; 14.0 kernels x 48 instances; d logits [32768,128] fp32 -> pieces [32768,384] bf16; weight [128,2048] bf16 -> [384,2048]; wgrad [128,2048] fp32 -> bf16.
- **Numbers:** 60.4 us measured vs 6.9 us ideal per instance (47.7 MB external); gap 2.57 ms.
- **Dataflow:** All pieces derive elementwise from the fp32 gradient; writing them directly in the stacked layout removes the cats.
- **Blockers:** Disabled "fp32_to_bf16_split" local_compile region. T3 extension: split in registers inside the GEMM prologue (noted as TODO in the source).
- **Tier T0:** Exact bit manipulations (bf16 round, shift-upcast, subtract) and copies.
- **Notes:** Member kernels: 672; ideal from 47.7 MB external bytes per instance.

### R-27 Cross-layer: ffn residual add + next attention_norm + MXFP8 quantize of the wqkv input
- **What:** Layer i-1's trailing residual add (x + ffn_out), layer i's attention_norm RMSNorm and the 1x32 MXFP8 quantization (+ scale swizzle) of the normalized activations for the wqkv GEMM.
- **Where:** `torchtitan/models/qwen3/model.py:47-73; torchtitan/models/common/norm.py:36-51; torchtitan/quantization/mxfp8/linear.py:77-170`; 3.98 kernels x 48 instances; x, ffn_out [32768,2048] bf16 -> x [32768,2048] bf16, normed [32768,2048] bf16 (saved), fp8 e4m3 [32768,2048] + e8m0 scales [32768,64] (swizzled).
- **Numbers:** 140.3 us measured vs 87.6 us ideal per instance (606.1 MB external); gap 2.53 ms.
- **Dataflow:** The add output is the residual stream x consumed row-wise by the norm; the norm output row is consumed only by the MXFP8 quantizer (and saved in bf16 for the wgrad). One row-parallel kernel can do add -> rmsnorm -> amax/scale/cast per 32-element block -> swizzled scale write.
- **Blockers:** Crosses the FSDP block boundary (the add is in block i-1, the norm in block i; FSDP pre-forward hooks of block i run in between). The quantizer is inside the MXFP8Linear autograd Function, so the norm output must be passed in pre-quantized. Layer 0 has no preceding add.
- **Tier T1:** RMSNorm row reduction over 2048 must match the quack kernel order; the 1x32 amax is order-insensitive (max).
- **Notes:** 47 adds + 48 norms + 48 quantize + 48 swizzle kernels. Member kernels: 191; ideal from 606.1 MB external bytes per instance.

### R-28 MXFP8 quantize of the attention output in the flash-attention forward epilogue
- **What:** The wo input (attention output [32768,4096] bf16) is quantized to MXFP8 1x32 plus scale swizzle by two separate kernels right after flash attention writes it.
- **Where:** `torchtitan/models/common/attention/attention.py:318-324; torchtitan/quantization/mxfp8/linear.py:77-170`; 2.0 kernels x 48 instances; O [32768,4096] bf16 -> fp8 e4m3 [32768,4096] + e8m0 scales [32768,128] swizzled.
- **Numbers:** 70.3 us measured vs 20.0 us ideal per instance (138.4 MB external); gap 2.41 ms.
- **Dataflow:** Each output tile of the attention kernel is a complete set of 32-element blocks along head_dim (128), so the per-block amax/scale can be computed in the epilogue from registers.
- **Blockers:** Flash attention is a library kernel (CuTe DSL flash_attn); the quantized output must be handed to the MXFP8Linear Function. Ideal counts only the extra fp8+scale bytes written by the epilogue (attention kernel time is excluded since document lengths are unknown).
- **Tier T3:** The quantization itself is exact given O; but it requires changes inside the flash-attention kernel epilogue (bf16 O must still be written for the attention backward).
- **Notes:** Ideal = incremental bytes only (138 MB/layer), not the full attention. Member kernels: 96; ideal from 138.4 MB external bytes per instance.

### R-29 Router gate dgrad GEMM accumulates into the expert input gradient (beta=1 epilogue)
- **What:** The HiMidLo gate input-gradient GEMM [32768,384]x[384,2048] writes a fresh bf16 [32768,2048] tensor that autograd then adds into the routed-experts input gradient with a separate add_.
- **Where:** `torchtitan/models/common/linear/hi_mid_lo_linear.py:342-362`; 2.0 kernels x 48 instances; pieces [32768,384] bf16 x stacked W [384,2048] bf16 -> += d ffn_norm-output [32768,2048] bf16.
- **Numbers:** 88.8 us measured vs 42.6 us ideal per instance (295.2 MB external); gap 2.21 ms.
- **Dataflow:** The GEMM output is consumed only by the accumulation into the gradient buffer that the expert backward already produced.
- **Blockers:** Autograd owns gradient accumulation; the Function would need the existing gradient buffer as an in/out operand (the code already does this for weight gradients).
- **Tier T3:** Eager rounds the GEMM result to bf16, then adds in fp32 and rounds again; the epilogue must round the accumulator to bf16 before adding C to stay bitwise.
- **Notes:** Overlaps R-14 on the add_ kernel. Member kernels: 96; ideal from 295.2 MB external bytes and 51.5 GFLOP per instance.

### R-30 wo MXFP8 GEMM epilogue adds the attention residual
- **What:** The wo projection GEMM [32768,4096]x[4096,2048] writes attn_out, then a separate kernel computes x + attn_out.
- **Where:** `torchtitan/models/common/attention/attention.py:322-324; torchtitan/models/qwen3/model.py:55-60`; 2.0 kernels x 48 instances; O fp8 [32768,4096] x W fp8 [4096,2048] + x [32768,2048] bf16 -> x [32768,2048] bf16.
- **Numbers:** 190.2 us measured vs 152.1 us ideal per instance (415.3 MB external); gap 1.83 ms.
- **Dataflow:** attn_out is consumed only by the residual add (it is not saved for backward); the add is elementwise on the GEMM output tile.
- **Blockers:** Custom epilogue on the MXFP8 GEMM inside the MXFP8Linear Function; the residual lives outside the attention module. The following ffn_norm needs whole 2048-wide rows, which a tile epilogue does not see.
- **Tier T3:** The epilogue must round the GEMM accumulator to bf16 (attn_out) before the bf16 residual add, as eager does.
- **Notes:** Overlaps R-25 on the residual add. Member kernels: 96; ideal from 415.3 MB external bytes and 549.8 GFLOP per instance.

### R-31 Dist-MoE backward plan/metadata kernels
- **What:** Backward plan construction, activation-to-dispatch copy and four GEMM scheduler preparation kernels per layer.
- **Where:** `dist_moe/_blockscaled.py (moe_backward_plan, moe_forward_recompute stages)`; 8.0 kernels x 48 instances; ids [32768,8], counts -> backward dispatch/scheduling tables.
- **Numbers:** 33.0 us measured vs 1.5 us ideal per instance (10.5 MB external); gap 1.51 ms.
- **Dataflow:** The backward plan is a function of the forward routing; it could be produced together with the forward plan.
- **Blockers:** Inside the dist_moe library.
- **Tier T2:** Integer index tables; same slot order required.
- **Notes:** Member kernels: 384; ideal from 10.5 MB external bytes per instance.

### R-32 Root FSDP groups (tok_embeddings, norm+lm_head): unshard and reduce-scatter glue
- **What:** Cast/copy-in, copy-out of the all-gathered [151936,2048] embedding and lm_head weights (lm_head re-gathered for the decoder backward), chunk_cat into the reduce-scatter buffer and fp32 cast of the reduced shard.
- **Where:** `torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:36,259-319`; 10.5 kernels x 2 instances; [151936,2048] params, shards [18992,2048]; bf16 all-gather [8,38895616].
- **Numbers:** 773.7 us measured vs 157.3 us ideal per instance (1089.1 MB external); gap 1.23 ms.
- **Dataflow:** For single-parameter groups the dim-0 all-gather output already has the parameter layout, so copy-out and chunk_cat are layout no-ops; casts fold into the copy-in and the post-reduce copy.
- **Blockers:** FSDP2 generic buffer management; norm weight is packed with lm_head so that group needs a strided copy.
- **Tier T0:** Pure copies and casts.
- **Notes:** Ideal assumes a copy-out is still needed for the packed (norm, lm_head) group only. Member kernels: 21; ideal from 1089.1 MB external bytes per instance.

### R-33 Embedding backward (deterministic sort-based scatter-add)
- **What:** Gradient of tok_embeddings: zero-fill of [151936,2048], radix sort of token ids, segment reduction and scatter.
- **Where:** `torchtitan/models/common/embedding.py`; 21.0 kernels x 1 instances; grad [32768,2048] bf16, ids [32768] -> grad weight [151936,2048] bf16.
- **Numbers:** 526.7 us measured vs 109.3 us ideal per instance (756.8 MB external); gap 0.42 ms.
- **Dataflow:** Zero-fill and scatter of the same buffer can be one kernel; sort + segment reduce is the deterministic contract.
- **Blockers:** Single instance; library implementation.
- **Tier T2:** Segment sums over duplicate token ids must follow the sorted order.
- **Notes:** Member kernels: 21; ideal from 756.8 MB external bytes per instance.

### R-34 Last-layer residual add + final norm forward, final norm backward
- **What:** Layer 47 trailing residual add fused with the final RMSNorm, and the final RMSNorm backward with its weight-grad reduction.
- **Where:** `torchtitan/models/qwen3/model.py:69-73; torchtitan/models/common/decoder.py:352`; 4.0 kernels x 1 instances; x, ffn_out [32768,2048] bf16 -> h [32768,2048] bf16; backward dy [32768,2048] bf16.
- **Numbers:** 188.6 us measured vs 155.1 us ideal per instance (1073.7 MB external); gap 0.03 ms.
- **Dataflow:** Same as the per-layer residual+norm pattern; single instance.
- **Blockers:** Final norm is a separate FSDP root module; negligible size.
- **Tier T1:** Row reductions of RMSNorm forward/backward and the column reduction for the weight gradient.
- **Notes:** Member kernels: 4; ideal from 1073.7 MB external bytes per instance.

## 5. No opportunity

| what | time (ms) | reason |
|---|---:|---|
| Zero-fill of the Dist-MoE peer-scatter buffers [32768,8,2048] bf16 (1 per layer forward, 2 per layer backward) | 22.884 | Fake-process-group emulation artifact: dist_moe/_buffers.py:501-514 (_initialize_fake_peer_scatter_output) zeroes the buffer only under fake symmetric memory; native symmetric memory has one physical writer per routed row, so the real job has no such kernel. |
| Flash-attention forward semaphore/counter memsets (3 per layer, <1 us each) | 0.118 | Library-internal 8-16 byte resets; nothing worth fusing (0.12 ms in total). |
| tok_embeddings forward gather [32768,2048] | 0.035 | Single kernel close to its roofline (35 us vs 19 us); only candidate is folding into layer 0 attention_norm, negligible. |

Every other memory-bound kernel class of the forward and backward is a member of at least one region (checked mechanically: `regions.py` prints the unclaimed glue classes, and only the three rows above remain). GEMM, attention and communication kernels are outside the region scope except where a T3 region names them.

## 6. Observations about working from these artifacts

**What the artifacts made easy**
- Forward attribution was nearly free: every kernel's correlation id leads to its launching CPU op, and the `## mod <path> [<class>] ##` scopes nest cleanly around those ops (no partially overlapping host events), so a single stack sweep gives module path and layer index. Kineto's `Input Dims` / `Input type` / `Concrete Inputs` give shapes, dtypes and flags (for example rowwise vs columnwise MXFP8 quantization) for every aten and custom op.
- Memcpy and memset activities carry their byte counts, so their ideal times needed no shape reasoning.
- Python stacks resolved launches that have no CPU op (raw Triton launches inside FSDP hooks and inside the MXFP8 linear Function) and exposed the origin of the largest false lead (below).
- The `moe_*` and `FSDP::*` annotations split library-owned work into stages (dispatch plan, main, postprocess; pre-forward, all-gather, copy-out, post-backward reduce) and identify the FSDP group and process-group size (`[pg=8]` vs `[pg=1]`).

**What was hard**
- Kineto forward ops carry no sequence number, only the backward `evaluate_function` nodes do. Mapping backward kernels to the forward op required the execution trace (`seq_id` on forward nodes, outermost node per `seq_id`) and the `rf_id` -> `Record function id` join back into Kineto, then the forward op's module scope.
- Kineto puts the autograd thread's ops on the same tid as the main thread; nesting still works by time, but nothing in Kineto alone marks which ops ran on the backward thread.
- Wall-clock phase boundaries are wrong for this program: the chunked loss interleaves 8 lm_head/CE forwards with their backwards, so ops of chunks 2-8 occur after the first backward op. Phase had to come from `evaluate_function` ancestry plus FSDP backward annotations, with the gradient-accumulator slice copies (outside autograd) assigned to backward by hand.
- Custom ops that launch many kernels (`dist_moe::block_scaled_forward/backward` launch ~13 kernels including GEMMs; flash-attention backward launches preprocess, main, three postprocess kernels and seven memsets) record only the op's inputs, so per-kernel bytes had to be derived from kernel names and library source rather than from the trace.
- Values are not recorded: `cu_seqlens` (72 documents) and positions are unknown, so attention FLOPs could not be computed and attention-adjacent ideals (R-28) count only the incremental bytes; the round-robin routing (exactly balanced) had to be read from source.
- Expert-parallel communication in Dist-MoE happens inside kernels through symmetric memory (metadata gathers and the dispatch/combine-fused grouped GEMMs), not under collective CPU ops, so by the stated rule it counts as compute; under the fake process group those kernels read local memory, so their times differ from the real job.
- Templated kernel names are long and truncated differently per kernel; classifying GEMM vs glue by name needed care (see false leads).

**Evidence that was missing or needed workarounds**
- No explicit link from a backward kernel to the forward module: built from ET `seq_id` + `rf_id` as above. Some `evaluate_function: AddBackward0` events lack a sequence number; the kernels under them are FSDP pre-backward hooks, attributed through the `FSDP::* (layers.N)` annotation instead. `EmbeddingBackward0` and the final `FusedRmsNormBackward0` had no matching forward node and were attributed by node type.
- Region external bytes were computed analytically from the model shapes (per-region formulas in `regions.py`), not from ET storage ids. The ET tensor and storage ids would allow a mechanical producer/consumer graph; I used module structure plus source reading instead, which is faster but less exhaustive.
- Op-level bytes in `kernel_table.csv` come from ET input/output tensors (numel x itemsize) and are approximate for in-place and multi-kernel ops; raw Triton launches have no op record, so their op-level ideal is blank (0).

**How phase and module were identified for backward ops**
- For each kernel: innermost enclosing `autograd::engine::evaluate_function: <Node>` -> its `Sequence number` -> ET forward node with that `seq_id` (`fw_tid` 0, outermost node of that `seq_id`) -> its `rf_id` -> Kineto CPU op with that `Record function id` -> innermost enclosing `## mod` scope. Kernels under FSDP backward hooks take the module from the FSDP annotation (`layers.N`, `tok_embeddings`, `norm, lm_head`). Loss ops sit outside any module (module "").

**False leads discarded**
- The three `[32768,8,2048]` bf16 zero-fills per layer in Dist-MoE (22.9 ms, the largest single memset class) looked like a zero-fill + combine fusion target. The Python stack shows `dist_moe/_buffers.py:_initialize_fake_peer_scatter_output`, which zeroes the peer-scatter buffer only under fake symmetric memory; the real job has no such kernel. Listed under no-opportunity.
- The 9.7 s step with 90% GPU idle looked like synchronization stalls; the gap histogram shows thousands of 0.1-10 ms gaps, i.e. host launch overhead under heavy instrumentation, not a few syncs.
- Device-to-device copies inside `FSDP::pre_forward (layers.N) [pg=1]` looked like communication but run under no collective op (the size-1 expert group copies through its all-gather buffer); they are glue (R-08), not communication.
- A name-based GEMM classifier (`cutlass_kernel` substring) put the Dist-MoE weight quantizer and GEMM scheduler-preparation kernels into GEMM time; they are memory-bound glue.
- A wall-clock forward/backward split put chunks 2-8 of the loss forward into the backward; fixed as described above.

**Kinds of opportunity I may have missed**
- Producer/consumer links that only show up in tensor storage ids (buffer reuse across layers, views shared across modules), since regions were found from module structure and source rather than an ET dataflow graph.
- Inefficiency inside library kernels that is not a fusion problem: the Dist-MoE dispatch- and combine-fused grouped GEMMs run at roughly 20-45% of the MXFP8 peak (their wgrad GEMMs at 85-95%), and the deterministic flash-attention backward takes 4.3x its forward.
- Cross-step reuse: with round-robin routing the dispatch plan is identical for every layer and step, and expert/attention weight quantization only changes after the optimizer step; I noted these but did not size them as separate regions.
- Communication-glue in the real job: emulated collectives are local copies here; real NCCL/symmetric-memory kernels and their packing may differ.
- Opportunities in the optimizer (27.2 ms), which was out of scope.

**Effort**
- Wall time: started 2026-11-10T16:47:46-08:00, deliverables written 2026-11-10T17:13:15-08:00.
- Scripts written (all under `out/`): `load_trace.py`, `load_et.py`, `attribute.py`, `classify.py`, `analyze.py`, `regions.py`, `regions_meta.py`, `build_outputs.py`, `idle.py`, `obs_text.py`, `make_report.py`, `log_region.py`; exploration helpers `explore1-6.py`, `dump_layer.py`.

## 7. Reproduction

Run from `/home/bahuang/local/step_analysis/qwen3_30b_eager` (CPU only, isolated Python):

```bash
cd /home/bahuang/local/step_analysis/qwen3_30b_eager
python -I out/load_trace.py      # Kineto trace -> out/trace.pkl
python -I out/load_et.py         # execution trace -> out/et.pkl
python -I out/attribute.py       # kernel -> launching op, module, phase, backward seq -> forward op (out/kernels.pkl)
python -I out/classify.py        # phase fix for chunked loss, module normalization, categories (out/kernels2.pkl, out/class_summary.txt)
python -I out/analyze.py         # step budget (out/budget.json), per-op bytes/flops/ideal (out/kernel_table.csv, out/kernels3.pkl)
python -I out/regions.py         # region membership, measured/ideal/gap, coverage check (out/regions_eval.pkl)
python -I out/build_outputs.py   # out/regions.json, out/summary.json
python -I out/idle.py            # out/idle_hist.json
python -I out/make_report.py "<start time>" "<end time>"   # out/report.md
```

`out/regions_log.jsonl` is the timestamped log of regions as they were identified (written with `out/log_region.py`). `out/kernel_table.csv` holds one row per GPU activity with module, layer, phase, launching op, backward node, forward op/module, input shapes/dtypes, op-level bytes, flops, ideal time and measured/ideal ratio.
