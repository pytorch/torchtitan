# Kernel-fusion opportunities in one Llama 3 8B training step (eager, FSDP 8)

## 1. Setup

- **Program.** torchtitan eager `Trainer` running `torchtitan_recipes.models.llama3.llama3_8b(seq_len=8192)` in BF16: 32 decoder layers (dim 4096, 32 query heads / 8 KV heads of 128, SwiGLU hidden 14336 with a fused `w13` [2,14336,4096], fused `wqkv` [6144,4096]), vocab 128256, complex RoPE, quack/cuteDSL RMSNorm (`aten::rms_norm`), varlen flash attention (flash_attn.cute SM100) on packed documents, `ChunkedLossWrapper` (8 chunks of 1024 tokens: lm_head + CE + per-chunk backward, then one decoder backward). FSDP2 over 8 ranks (bf16 params, fp32 reduce), reshard after forward, one 8192-token microbatch. One process with a fake process group runs rank 0; collectives are emulated by local copies. No activation recompute, `local_compile` off (so the default `loss`, `fused_binary_activation` and `rope` compiled regions are not used), deterministic algorithms on, CUDA graphs off. Step 10 profiled after 3 warmup steps.
- **Roofline numbers** (`hw_profile.json`, NVIDIA GB300): HBM copy bandwidth 6921.3 GB/s; BF16 GEMM peak 1929.5 TFLOP/s (max of the measured shapes). Ideal time = max(bytes / hbm_copy_gbps, flops / BF16 peak). No MXFP8 GEMMs run in this program.
- GEMMs in this step run at 1961 TFLOP/s on average (368.9 TFLOP in 188.07 ms, ideal 191.17 ms), i.e. at or above the random-data peak (low-entropy data in the fake-PG run), so GEMM-containing (T3) regions have slightly pessimistic ideals.

## 2. Step budget

| category | step ms | forward ms | backward ms | notes |
|---|---|---|---|---|
| gemm | 188.07 | 59.32 | 128.75 | aten::mm / addmm (nvjet) |
| attention | 23.30 | 3.93 | 19.37 | flash-attention fwd/bwd incl. pre/post-process and memsets |
| communication | 11.91 | 5.44 | 6.47 | kernels under c10d:: collectives (fake-PG copies); excluded from regions |
| memory_bound_glue | 125.25 | 40.71 | 84.54 | everything else in forward + backward |
| optimizer | 4.23 | - | - | fused AdamW (out of scope) |
| other | 1.94 | - | - | grad-norm clipping after backward, batch H2D copies (out of scope) |
| **sum of kernel time** | 354.70 | | | |

- GPU busy in the step (union over streams): **354.39 ms**; GPU idle inside the step's kernel span: **1106.89 ms** (span 1461.3 ms, so the GPU is idle 76% of the step).
- Forward + backward: busy 348.23 ms over a 1399.9 ms span (idle 1051.7 ms). The step is CPU/launch bound under the profiler (e.g. layer 5 forward: 19.0 ms wall for 3.1 ms of kernels; backward 41.3 ms wall for 6.9 ms).
- **Attribution coverage: 100.0%** of forward+backward kernel time is mapped to a launching op, a phase and a module path (97.4% to a non-empty module; the rest is the loss, the chunked-loss gradient buffer and the varlen-metadata builder, which run outside any module and use module ""). Coverage is 100.09% of the busy union because a few kernels overlap across streams.
- Excluded from the step: 3 multi-tensor-apply kernels (1.74 ms) of the previous step's optimizer still running when the capture started (no CPU launch in the trace).
- Memory-bound glue in forward+backward: 125.25 ms; 123.86 ms of it is inside at least one region, the remaining 1.39 ms is listed under No opportunity. The flash-attention auxiliary kernels (counted under attention) are also covered by R-16.

## 3. Ranked regions

Regions overlap (alternative fusions of the same kernels are listed separately and cross-referenced), so gaps must not be summed blindly. A member-disjoint selection (R-25 + R-07 + R-26 + R-18 + R-09 + R-02 + R-10 + R-15 + R-19 + R-27 + R-22 + R-16 + R-24 + R-12 + R-13 + R-21 + R-23; checked disjoint by op instance) totals **110.1 ms** of gap (31.1% of GPU-busy step time). Restricted to non-GEMM tiers T0-T2 (R-01 + R-05 + R-04 + R-17 + R-08 + R-02 + R-10 + R-15 + R-27 + R-22 + R-20 + R-24 + R-12 + R-13 + R-21 + R-23; R-11 is contained in R-08 + R-10), the disjoint total is **92.2 ms** (26.0%).

| id | title | phases | scope | kind | inst | measured ms | ideal ms | gap ms | % of step | tier |
|---|---|---|---|---|---|---|---|---|---|---|
| R-25 | Layer weight-gradient GEMMs writing fp32 directly into the reduce-scatter buffer | backward | gemm-adjacent, communication-glue | GEMM epilogue into collective buffer | 32 | 90.619 | 59.263 | 31.357 | 8.85 | T3 |
| R-07 | w2 dgrad GEMM with SwiGLU-backward epilogue | backward | gemm-adjacent, intra-module, forward-backward | GEMM epilogue | 32 | 41.569 | 15.955 | 25.614 | 7.23 | T3 |
| R-01 | FSDP reduce-scatter copy-in: w13 grad re-layout + bf16->fp32 chunk_cat | backward | communication-glue, intra-module | copy+cast into collective buffer | 32 | 31.003 | 6.051 | 24.953 | 7.04 | T2 |
| R-05 | SwiGLU backward: two muls, silu_backward and the gate/up gradient cat in one pass | backward | intra-module, forward-backward | pointwise | 32 | 26.672 | 5.430 | 21.242 | 5.99 | T0 |
| R-26 | SwiGLU computed in the w2 GEMM A-operand prologue | forward | gemm-adjacent, intra-module, forward-backward | GEMM prologue | 32 | 34.077 | 15.955 | 18.122 | 5.11 | T3 |
| R-04 | SwiGLU forward: silu(gate) * up in one pass | forward | intra-module | pointwise | 32 | 17.849 | 3.258 | 14.591 | 4.12 | T0 |
| R-06 | w13 GEMM with SwiGLU epilogue | forward | gemm-adjacent, intra-module | GEMM epilogue | 32 | 46.200 | 31.911 | 14.289 | 4.03 | T3 |
| R-18 | Fused linear cross-entropy (lm_head forward + CE + lm_head dgrad) per chunk | forward, backward | gemm-adjacent, forward-backward, cross-module | GEMM + online softmax + GEMM | 8 | 16.933 | 8.921 | 8.012 | 2.26 | T3 |
| R-17 | Cross-entropy forward + backward per loss chunk | forward, backward | forward-backward | row reduction + pointwise (online softmax) | 8 | 8.284 | 0.607 | 7.677 | 2.17 | T1 |
| R-09 | wqkv GEMM with split + RoPE epilogue | forward | gemm-adjacent, cross-module | GEMM epilogue | 32 | 12.441 | 6.838 | 5.603 | 1.58 | T3 |
| R-08 | QKV split copies + complex RoPE forward in one pass | forward | cross-module, horizontal | pointwise + gather | 32 | 5.953 | 0.951 | 5.003 | 1.41 | T0 |
| R-02 | FSDP all-gather copy-out: split_with_sizes_copy + w13 Shard(1) re-assembly | forward, backward | communication-glue, intra-module | copy-out with layout permutation | 64 | 13.026 | 8.067 | 4.959 | 1.40 | T2 |
| R-10 | RoPE backward + dQKV assembly in one pass | backward | cross-module, horizontal | pointwise + concat | 32 | 5.482 | 0.950 | 4.531 | 1.28 | T0 |
| R-27 | FSDP all-gather copy-in: cast each layer's fp32 shard once per step, reuse it for the backward re-gather | forward, backward | communication-glue, forward-backward | cross-phase reuse (cast once) | 32 | 3.483 | 0.756 | 2.726 | 0.77 | T0 |
| R-15 | RMSNorm backward + residual-gradient accumulation | backward | intra-module, cross-module | row reduction + pointwise | 64 | 4.910 | 2.483 | 2.428 | 0.69 | T1 |
| R-19 | lm_head weight gradient: one GEMM over all chunks writing the reduce-scatter buffer | backward | horizontal, gemm-adjacent, communication-glue | sibling GEMM batching + output aliasing | 1 | 6.753 | 4.461 | 2.292 | 0.65 | T3 |
| R-03 | FSDP all-gather copy-in: fp32->bf16 cast fused with copy into all-gather buffer | forward, backward | communication-glue | cast + copy into collective buffer | 67 | 3.825 | 1.683 | 2.142 | 0.60 | T0 |
| R-14 | Residual adds as wo / w2 GEMM epilogues | forward | gemm-adjacent, intra-module | GEMM epilogue (C += residual) | 64 | 22.112 | 20.514 | 1.598 | 0.45 | T3 |
| R-22 | Embedding backward writing fp32 straight into the reduce-scatter buffer | backward | communication-glue, cross-module | scatter-add + cast into collective buffer | 1 | 1.808 | 0.313 | 1.495 | 0.42 | T2 |
| R-16 | Flash-attention backward prologue/epilogue kernels folded into neighbours | forward, backward | gemm-adjacent, cross-module | attention prologue/epilogue | 32 | 4.141 | 2.647 | 1.494 | 0.42 | T3 |
| R-20 | lm_head grad accumulator aliased to the reduce-scatter buffer (last chunk GEMM + copy) | backward | communication-glue, gemm-adjacent | output aliasing | 1 | 1.738 | 0.646 | 1.091 | 0.31 | T0 |
| R-24 | Varlen attention metadata (cu_seqlens) from positions in one kernel | forward | intra-module | scan + scatter | 1 | 0.723 | 0.000 | 0.723 | 0.20 | T2 |
| R-11 | RoPE cache gather, position bounds check and cache conj computed once per step | forward, backward | cross-layer | hoisting (common subexpression across layers) | 32 | 0.641 | 0.002 | 0.638 | 0.18 | T0 |
| R-12 | Attention residual add + ffn_norm RMSNorm | forward | cross-module | pointwise + row reduction | 32 | 1.608 | 1.241 | 0.367 | 0.10 | T1 |
| R-13 | FFN residual add + next layer's attention_norm (or final norm) | forward | cross-layer | pointwise + row reduction | 32 | 1.575 | 1.241 | 0.333 | 0.09 | T1 |
| R-21 | Final norm backward fused with the chunked-loss gradient accumulation | forward, backward | cross-module | row reduction + copy elimination | 1 | 0.104 | 0.029 | 0.075 | 0.02 | T1 |
| R-23 | Token embedding gather + layer-0 attention_norm | forward | cross-module | gather + row reduction | 1 | 0.038 | 0.029 | 0.009 | 0.00 | T1 |

Regions per scope tag: gemm-adjacent 10, intra-module 10, cross-module 10, communication-glue 8, forward-backward 6, horizontal 3, cross-layer 2 (27 regions).

## 4. Region details

### R-25 Layer weight-gradient GEMMs writing fp32 directly into the reduce-scatter buffer

- **What / where:** The four weight-gradient GEMMs per layer write bf16 gradients that FSDP immediately re-reads to build the fp32 reduce-scatter input; their epilogues can write (bf16-rounded, widened) fp32 values at each rank's offset of the reduce-scatter buffer. Source: `torchtitan/models/common/linear/linear.py:210-237; torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py`. Shapes: wqkv [6144,4096], wo [4096,4096], w13 [28672,4096], w2 [4096,14336] grads -> fp32 [8,27264000].
- **Numbers:** 32 instances x 6.0 kernels; 2831.8 us measured vs 1852.0 us ideal per instance (2013.3 MB external); gap 31.357 ms (8.85% of GPU-busy step). Evidence: measured.
- **Dataflow:** Each weight gradient is consumed only by the reduce-scatter copy-in (AccumulateGrad is a no-op hand-off).
- **Blockers:** Requires FSDP to expose its reduce-scatter buffer before backward and GEMMs with a row-blocked scattered output layout (each rank's slice is a separate address range).
- **Tier T3:** Epilogue must round each gradient to bf16 before widening to fp32 to match eager (wgrad in bf16, then chunk_cat upcast).
- **Notes:** overlaps R-01.

### R-07 w2 dgrad GEMM with SwiGLU-backward epilogue

- **What / where:** The w2 input-gradient GEMM produces dh [8192,14336]; its epilogue can load gate_up and write the stacked [8192,28672] SwiGLU gradient directly. Source: `torchtitan/models/common/linear/linear.py:210-237; activation.py:39-62`. Shapes: dY bf16 [8192,4096], w2 bf16 [4096,14336], gate_up bf16 -> d_gate_up bf16 [8192,28672].
- **Numbers:** 32 instances x 5.0 kernels; 1299.0 us measured vs 498.6 us ideal per instance (1124.1 MB external); gap 25.614 ms (7.23% of GPU-busy step). Evidence: measured.
- **Dataflow:** dh is consumed only by the SwiGLU backward.
- **Blockers:** Custom GEMM epilogue with two extra operand loads; bitwise main loop.
- **Tier T3:** GEMM output must be rounded to bf16 (dh) before the pointwise backward math.
- **Notes:** overlaps R-05.

### R-01 FSDP reduce-scatter copy-in: w13 grad re-layout + bf16->fp32 chunk_cat

- **What / where:** Per layer, the bf16 weight gradients (wqkv, wo, w13 [2,14336,4096], w2, two norms) are first re-laid out (w13 chunked along dim 1 and concatenated to [16,1792,4096]) and then copied with an fp32 upcast into the [8, 27.26M] reduce-scatter input; one pass can write the fp32 buffer straight from the gradients. Source: `torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py (foreach_reduce, chunk_cat); torchtitan/models/common/feed_forward.py:47-75 (w13 [2,F,D])`. Shapes: grads bf16: [6144,4096],[4096,4096],[2,14336,4096],[4096,14336],[4096]x2 -> fp32 [8,27264000].
- **Numbers:** 32 instances x 2.0 kernels; 968.9 us measured vs 189.1 us ideal per instance (1308.7 MB external); gap 24.953 ms (7.04% of GPU-busy step). Evidence: measured.
- **Dataflow:** cat output [16,1792,4096] bf16 is consumed only by _chunk_cat, whose output is consumed only by the reduce-scatter; both are index permutations plus a cast of the same 218M gradient elements.
- **Blockers:** Owned by FSDP2 (foreach_reduce / chunk_cat); the Shard(1) w13 path is a separate cat before the chunk_cat. Needs a custom copy-in kernel or a chunk_cat that accepts a per-parameter shard dim.
- **Tier T2:** Pure data movement with an exact bf16->fp32 widening; bitwise identical as long as every element lands in the same rank-major slot.
- **Notes:** overlaps R-25. chunk_cat runs at ~2.2 TB/s and the Shard(1) cat at ~1.2 TB/s; the fused ideal is 189 us/layer.

### R-05 SwiGLU backward: two muls, silu_backward and the gate/up gradient cat in one pass

- **What / where:** Backward of SwiGLU computes d_up = dh * silu(gate), d_silu = dh * up, d_gate = silu_backward(d_silu, gate) and stacks d_gate and d_up into [8192,28672]; one kernel reads dh and gate_up and writes the stacked gradient directly. Source: `torchtitan/models/common/feed_forward.py:71-75; activation.py:39-62,175-176`. Shapes: dh bf16 [8192,14336], gate_up bf16 [8192,2,14336] -> d_gate_up bf16 [8192,28672].
- **Numbers:** 32 instances x 4.0 kernels; 833.5 us measured vs 169.7 us ideal per instance (1174.4 MB external); gap 21.242 ms (5.99% of GPU-busy step). Evidence: measured.
- **Dataflow:** The two mul outputs and silu_backward output feed only the UnbindBackward0 cat, whose output feeds only w13's backward GEMMs. Recomputing silu(gate) from gate_up removes the forward's saved silu output (forward-backward pairing).
- **Blockers:** Autograd materializes each gradient separately and stacks them with cat; needs a custom autograd Function (as the local_compile region would provide).
- **Tier T0:** Pointwise with bf16 roundings at the same points as eager (d_up = dh*silu(gate), d_silu = dh*up, silu_backward in fp32).
- **Notes:** overlaps R-07.

### R-26 SwiGLU computed in the w2 GEMM A-operand prologue

- **What / where:** Instead of writing h = silu(gate)*up [8192,14336] and reading it back, the w2 projection can compute h tile by tile while loading its A operand from gate_up. Source: `torchtitan/models/common/feed_forward.py:73-75; activation.py:39-62,175-176`. Shapes: gate_up bf16 [8192,2,14336], w2 bf16 [4096,14336] -> out bf16 [8192,4096].
- **Numbers:** 32 instances x 3.0 kernels; 1064.9 us measured vs 498.6 us ideal per instance (654.3 MB external); gap 18.122 ms (5.11% of GPU-busy step). Evidence: measured.
- **Dataflow:** h is consumed by the w2 GEMM and saved for w2's weight gradient; recomputing h from the saved gate_up in backward removes the saved copy (forward-backward pairing).
- **Blockers:** h is also needed by w2's weight-gradient GEMM in backward: either still write h (costs 235 MB, stays flop-bound) or recompute it in the backward prologue. Needs a GEMM with a fused elementwise A-prologue.
- **Tier T3:** Prologue must produce the same bf16 h values (silu in fp32, bf16 round, multiply, bf16 round) before feeding the MMA.
- **Notes:** overlaps R-04, R-06, R-14. Alternative to R-06 (epilogue of w13); only one of the two can own the SwiGLU.

### R-04 SwiGLU forward: silu(gate) * up in one pass

- **What / where:** In each feed_forward, silu reads the strided gate half of w13's [8192,2,14336] output and writes a [8192,14336] bf16 tensor that mul immediately reads with the up half; one kernel reads gate_up once and writes the product. Source: `torchtitan/models/common/feed_forward.py:71-75; activation.py:39-62,175-176`. Shapes: gate_up bf16 [8192,2,14336] -> h bf16 [8192,14336].
- **Numbers:** 32 instances x 2.0 kernels; 557.8 us measured vs 101.8 us ideal per instance (704.6 MB external); gap 14.591 ms (4.12% of GPU-busy step). Evidence: measured.
- **Dataflow:** silu output is consumed by mul (and saved for MulBackward0); gate and up come from the same w13 output.
- **Blockers:** Eager runs two ATen kernels on a non-contiguous gate view (non-vectorized path, ~1.7 TB/s). The recipe's default local_compile region 'fused_binary_activation' covers this but is off in this run.
- **Tier T0:** Pointwise; reproduce silu in fp32 then bf16 rounding of silu(gate) before the multiply.
- **Notes:** overlaps R-06, R-26. Ideal assumes the backward recomputes silu(gate) instead of reading a saved copy (see R-05).

### R-06 w13 GEMM with SwiGLU epilogue

- **What / where:** The w13 projection [8192,4096]x[4096,28672] can apply silu(gate)*up in its epilogue, writing gate_up (saved for backward) and h without a separate pass. Source: `torchtitan/models/common/feed_forward.py:73-75; linear/linear.py:193-208`. Shapes: x bf16 [8192,4096], w13 bf16 [28672,4096] -> gate_up bf16 [8192,28672], h bf16 [8192,14336].
- **Numbers:** 32 instances x 3.0 kernels; 1443.8 us measured vs 997.2 us ideal per instance (1006.6 MB external); gap 14.289 ms (4.03% of GPU-busy step). Evidence: measured.
- **Dataflow:** w13 output feeds only the SwiGLU (and is saved for its backward).
- **Blockers:** Needs a GEMM with a paired-column epilogue (gate and up columns 14336 apart) and bitwise-identical main loop to cuBLAS/nvjet.
- **Tier T3:** Epilogue must round the GEMM accumulator to bf16 before silu (as eager does) and keep the same GEMM K-order.
- **Notes:** overlaps R-04, R-26. Measured w13 GEMM already runs above the random-data peak (low-entropy fake-PG data), so most of the gap is the SwiGLU pass.

### R-18 Fused linear cross-entropy (lm_head forward + CE + lm_head dgrad) per chunk

- **What / where:** Per chunk, lm_head [1024,4096]x[4096,128256] produces logits that only feed the CE, and the CE gradient feeds only lm_head's input-gradient GEMM; a fused linear-CE never writes the 263 MB bf16 logits or the fp32 intermediates. Source: `torchtitan/components/loss.py:918-964; linear/linear.py:193-237`. Shapes: h chunk bf16 [1024,4096], W bf16 [128256,4096] -> dh chunk bf16 [1024,4096], loss.
- **Numbers:** 8 instances x 13.25 kernels; 2116.7 us measured vs 1115.2 us ideal per instance (1067.5 MB external); gap 8.012 ms (2.26% of GPU-busy step). Evidence: measured.
- **Dataflow:** logits -> CE -> dlogits -> dgrad GEMM; nothing else reads logits/dlogits (dlogits also feeds the weight-gradient GEMM, R-19).
- **Blockers:** dlogits is also needed by the lm_head weight-gradient GEMM; a fused kernel must either recompute logits for it or emit dlogits. Bitwise is very hard.
- **Tier T3:** Needs both GEMMs and the vocab reduction inside; logits are never materialized, so bitwise needs the same per-tile reduction order as the unfused path (hard).
- **Notes:** overlaps R-17.

### R-17 Cross-entropy forward + backward per loss chunk

- **What / where:** For each 1024-token chunk, the [1024,128256] bf16 logits are upcast to fp32, log_softmax'd, gathered by nll, then backward zeroes a fp32 gradient, scatters, runs log_softmax backward and casts back to bf16; a single fused CE kernel reads the bf16 logits once and writes bf16 dlogits plus the scalar loss. Source: `torchtitan/components/loss.py:54-80,345-397,918-964`. Shapes: logits bf16 [1024,128256] (+fp32 copy, log-probs, fp32 grad), labels int64 [1024] -> dlogits bf16 [1024,128256].
- **Numbers:** 8 instances x 11.25 kernels; 1035.5 us measured vs 75.9 us ideal per instance (525.3 MB external); gap 7.677 ms (2.17% of GPU-busy step). Evidence: measured.
- **Dataflow:** The fp32 logits, log-probs and fp32 dlogits are each consumed by exactly the next op; the chunk's backward runs immediately after its forward, so forward and backward can share one kernel.
- **Blockers:** CrossEntropyLoss is compiled by the 'loss' local_compile region in the default recipe (off here); bitwise reduction order over the vocab.
- **Tier T1:** logsumexp over V=128256 in fp32 must reproduce PyTorch's log_softmax reduction tree; the nll gather/scatter and the 1/num_tokens scaling are pointwise.
- **Notes:** overlaps R-18. Members include the per-chunk scalar ops (div by token count, loss accumulation, ones_like seed) and the total_loss zero init.

### R-09 wqkv GEMM with split + RoPE epilogue

- **What / where:** The fused qkv projection can write q, k (rotated) and v in their separate contiguous layouts from its epilogue. Source: `torchtitan/models/common/attention/attention.py:169-197; rope.py:268-328`. Shapes: x bf16 [8192,4096], wqkv bf16 [6144,4096] -> q [8192,32,128], k,v [8192,8,128].
- **Numbers:** 32 instances x 14.0 kernels; 388.8 us measured vs 213.7 us ideal per instance (222.3 MB external); gap 5.603 ms (1.58% of GPU-busy step). Evidence: measured.
- **Dataflow:** wqkv output feeds only the split/rope chain.
- **Blockers:** Custom epilogue with a position-indexed cache load and head-dependent output routing.
- **Tier T3:** Epilogue rounds to bf16 before the fp32 rotation, matching eager.
- **Notes:** overlaps R-08, R-11.

### R-08 QKV split copies + complex RoPE forward in one pass

- **What / where:** After wqkv, q is made contiguous, k and v are copied, and RoPE upcasts q and k to fp32, multiplies by the complex cache gathered at positions, and casts back; one kernel can read the fused [8192,6144] qkv once and write rotated q, rotated k and v. Source: `torchtitan/models/common/attention/attention.py:169-197,287-302; rope.py:161-171,268-328`. Shapes: qkv bf16 [8192,8,6,128] -> q bf16 [8192,32,128], k,v bf16 [8192,8,128]; cache complex64 [8192,1,64].
- **Numbers:** 32 instances x 13.0 kernels; 186.0 us measured vs 29.7 us ideal per instance (205.6 MB external); gap 5.003 ms (1.41% of GPU-busy step). Evidence: measured.
- **Dataflow:** Each copy/cast output is consumed only by the next op in the chain; q and k use the same gathered cache rows (horizontal), v only needs the copy.
- **Blockers:** Eager ComplexRoPE uses view_as_complex/complex mul; the 'rope' local_compile region is off here. Bitwise needs matching the complex multiply arithmetic.
- **Tier T0:** Pointwise; the fp32 complex multiply must use the same operation order/FMA contraction as the eager complex-mul kernel (the compiled real-arithmetic path is only within 1 bf16 ulp).
- **Notes:** overlaps R-09, R-11.

### R-02 FSDP all-gather copy-out: split_with_sizes_copy + w13 Shard(1) re-assembly

- **What / where:** After each layer's all-gather (forward and backward prefetch), the [8, 27.26M] bf16 buffer is split into per-parameter tensors and w13 is then re-assembled by a second cat from its eight [2,1792,4096] shards; the copy-out can write w13 in its final [2,14336,4096] layout directly. Source: `torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py (foreach_all_gather_copy_out); _fsdp_param.py (Shard(1) chunk_cat)`. Shapes: bf16 [8,27264000] -> wqkv [6144,4096], wo [4096,4096], w13 [2,14336,4096], w2 [4096,14336], norms [4096].
- **Numbers:** 64 instances x 2.0 kernels; 203.5 us measured vs 126.0 us ideal per instance (872.4 MB external); gap 4.959 ms (1.40% of GPU-busy step). Evidence: measured.
- **Dataflow:** split_with_sizes_copy writes w13's per-rank pieces into a temporary [8,2*7340032] view that only the following cat reads.
- **Blockers:** FSDP2 copy-out path (split_with_sizes_copy + Shard(dim!=0) fix-up) is library-owned; needs a copy-out that handles non-zero shard dims in one pass.
- **Tier T2:** Order-sensitive copy only (no arithmetic); bitwise trivially once indices match.
- **Notes:** 64 instances = 32 forward + 32 backward re-gathers (reshard after forward).

### R-10 RoPE backward + dQKV assembly in one pass

- **What / where:** Backward of RoPE upcasts dq and dk to fp32, multiplies by the conjugated cache, casts back to bf16, and SplitWithSizesBackward concatenates dq, dk, dv into [8192,8,6,128]; one kernel can read dq/dk/dv and write the fused dqkv. Source: `torchtitan/models/common/rope.py:318-328; attention/attention.py:183-197`. Shapes: dq bf16 [8192,32,128], dk,dv bf16 [8192,8,128] -> dqkv bf16 [8192,8,6,128].
- **Numbers:** 32 instances x 9.0 kernels; 171.3 us measured vs 29.7 us ideal per instance (205.5 MB external); gap 4.531 ms (1.28% of GPU-busy step). Evidence: measured.
- **Dataflow:** Each cast/mul output is consumed only by the next op; the final cat feeds only wqkv's backward GEMMs.
- **Blockers:** Autograd materializes each step; needs a custom autograd Function for the rope+split.
- **Tier T0:** Pointwise conj-multiply in fp32 plus casts; same arithmetic-order caveat as R-08.
- **Notes:** overlaps R-11, R-16. Could additionally absorb the flash-attention backward postprocess (fp32 dq/dk/dv accumulators -> bf16), see R-16.

### R-27 FSDP all-gather copy-in: cast each layer's fp32 shard once per step, reuse it for the backward re-gather

- **What / where:** Each layer's fp32 parameter shard is cast to bf16 and copied into the all-gather input twice per step (forward gather and backward re-gather) with identical results; casting once into a persistent per-layer bf16 all-gather input removes the backward pair of copies. Source: `torch/distributed/fsdp/_fully_shard/_fsdp_param.py (all_gather_inputs); _fsdp_collectives.py (foreach_all_gather)`. Shapes: layer shard fp32 [27264000] -> bf16 all-gather input [27264000], 2 gathers per step.
- **Numbers:** 32 instances x 4.0 kernels; 108.8 us measured vs 23.6 us ideal per instance (163.6 MB external); gap 2.726 ms (0.77% of GPU-busy step). Evidence: measured.
- **Dataflow:** Forward and backward copy-in read the same unchanged fp32 shard and write the same bf16 all-gather input values.
- **Blockers:** Keeping the bf16 all-gather input (54.5 MB per layer, 1.7 GB for 32 layers) alive from forward to backward trades memory for the copies; FSDP2 frees it after each all-gather.
- **Tier T0:** The fp32 shard does not change between forward and backward of a step, so the cached bf16 values are bitwise identical.
- **Notes:** overlaps R-03. Covers the 32 decoder layers only (the 64 layer instances of R-03); ideal is one cast+copy per layer per step.

### R-15 RMSNorm backward + residual-gradient accumulation

- **What / where:** Each norm backward writes dx [8192,4096] and then add_ accumulates it into the residual gradient; the norm backward can add the incoming residual gradient in its epilogue and write the sum once. Source: `torchtitan/models/llama3/model.py:57-61; torchtitan/models/common/nn_modules.py:135 (RMSNorm)`. Shapes: dy, x bf16 [8192,4096], rstd fp32 [8192], residual grad bf16 [8192,4096] -> dx bf16 [8192,4096], dw [4096].
- **Numbers:** 64 instances x 3.0 kernels; 76.7 us measured vs 38.8 us ideal per instance (268.5 MB external); gap 2.428 ms (0.69% of GPU-busy step). Evidence: measured.
- **Dataflow:** The norm's dx is consumed only by the add_ into the residual-gradient buffer.
- **Blockers:** Autograd accumulates the two gradient contributions of the residual stream with a separate add_; needs a fused norm-backward-with-residual Function.
- **Tier T1:** Row reductions (dot(dy*w, x)) and the weight-gradient column reduction must follow the quack kernel's order.

### R-19 lm_head weight gradient: one GEMM over all chunks writing the reduce-scatter buffer

- **What / where:** The eight chunk weight-gradient GEMMs ([128256,1024]x[1024,4096], first mm then 7 addmm) each read and write the full fp32 [128256,4096] accumulator (4.2 GB per chunk), which is then copied unchanged into the reduce-scatter buffer; one K=8192 GEMM (or a persistent accumulator that is the reduce-scatter buffer) avoids both. Source: `torchtitan/components/loss.py:913-964; linear/linear.py:210-250; torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py`. Shapes: dlogits bf16 8x[1024,128256], h bf16 [8192,4096] -> grad fp32 [128256,4096] -> RS buffer fp32 [8,65667072].
- **Numbers:** 1 instances x 9.0 kernels; 6753.2 us measured vs 4460.7 us ideal per instance (4269.8 MB external); gap 2.292 ms (0.65% of GPU-busy step). Evidence: measured.
- **Dataflow:** All eight GEMMs write the same accumulator, consumed only by the reduce-scatter copy-in, whose dim-0 chunking of a contiguous [128256,4096] tensor is a no-op layout change.
- **Blockers:** Batching needs all eight chunks' bf16 dlogits (2.1 GB) alive at once, against the purpose of the chunked loss; aliasing the accumulator to FSDP's reduce-scatter buffer needs FSDP API support.
- **Tier T3:** Changes the fp32 accumulation order from 8 sequential K=1024 GEMMs (addmm into fp32) to one K=8192 GEMM; bitwise only if the GEMM keeps the same per-chunk accumulation order.
- **Notes:** overlaps R-20.

### R-03 FSDP all-gather copy-in: fp32->bf16 cast fused with copy into all-gather buffer

- **What / where:** Before every all-gather the fp32 sharded parameters are cast to bf16 into temporaries and then copied into the flat all-gather input buffer; a single foreach kernel can cast straight into the buffer. Source: `torch/distributed/fsdp/_fully_shard/_fsdp_param.py (all_gather_inputs); _fsdp_collectives.py (all_gather_copy_in)`. Shapes: layer shard fp32 27.26M elems -> bf16 all-gather input; tok_embeddings / (norm, lm_head) shards 65.7M elems.
- **Numbers:** 67 instances x 2.0 kernels; 57.1 us measured vs 25.1 us ideal per instance (173.9 MB external); gap 2.142 ms (0.60% of GPU-busy step). Evidence: measured.
- **Dataflow:** The bf16 temporaries written by the first _foreach_copy_ are read only by the all_gather_copy_in _foreach_copy_.
- **Blockers:** FSDP2 mixed-precision cast (FSDPParam.all_gather_inputs) and copy-in are separate library steps.
- **Tier T0:** Elementwise round-to-nearest bf16 cast and a copy; bitwise identical.
- **Notes:** overlaps R-27. Instances: 34 forward (tok_embeddings, 32 layers, norm+lm_head) + 33 backward (32 layer prefetches + norm+lm_head re-gather).

### R-14 Residual adds as wo / w2 GEMM epilogues

- **What / where:** Both residual adds of each block (x + wo(...), h + w2(...)) can be done in the epilogue of the GEMM producing the branch output, avoiding the separate read of the branch output and the residual. Source: `torchtitan/models/llama3/model.py:57-66; models/common/attention/attention.py:323-325; feed_forward.py:75`. Shapes: wo: [8192,4096]x[4096,4096]; w2: [8192,14336]x[14336,4096]; residual bf16 [8192,4096].
- **Numbers:** 64 instances x 2.0 kernels; 345.5 us measured vs 320.5 us ideal per instance (360.7 MB external); gap 1.598 ms (0.45% of GPU-busy step). Evidence: measured.
- **Dataflow:** wo and w2 outputs are consumed only by the residual adds.
- **Blockers:** Custom epilogue; alternative to R-12/R-13 (fuse the add into the following norm instead).
- **Tier T3:** Epilogue must round the accumulator to bf16 first and then add the bf16 residual in fp32 to match eager's two roundings.
- **Notes:** overlaps R-12, R-13, R-26. Per-instance values are averages of the wo and w2 instances; ideal is flop-bound so the gap is mostly the add kernel plus GEMM-vs-peak differences.

### R-22 Embedding backward writing fp32 straight into the reduce-scatter buffer

- **What / where:** The embedding backward zero-fills a 1.05 GB bf16 [128256,4096] gradient, scatter-adds the 8192 token gradients (sort + segment sum), and FSDP then copies it with an fp32 upcast into the reduce-scatter buffer; one pass can zero-fill and scatter directly into the fp32 buffer. Source: `torchtitan/models/common/embedding.py:37; torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py (foreach_reduce)`. Shapes: dh bf16 [8192,4096], ids int64 [8192] -> fp32 [8,65667072] (grad [128256,4096]).
- **Numbers:** 1 instances x 22.0 kernels; 1808.0 us measured vs 313.3 us ideal per instance (2168.5 MB external); gap 1.495 ms (0.42% of GPU-busy step). Evidence: measured.
- **Dataflow:** The bf16 embedding gradient is consumed only by the reduce-scatter copy-in.
- **Blockers:** embedding_dense_backward (deterministic) and FSDP's chunk_cat are separate library steps; bitwise requires the bf16 segment-sum rounding before widening.
- **Tier T2:** Deterministic sorted segment sums in bf16 (same token grouping and order), then an exact widening to fp32.

### R-16 Flash-attention backward prologue/epilogue kernels folded into neighbours

- **What / where:** Around each attention backward, accumulators are zeroed by memsets, a preprocess kernel computes rowsum(dO*O) (and zeroes dQ accumulation), and three postprocess kernels convert fp32 dQ/dK/dV accumulators to bf16; these can be folded into the wo dgrad epilogue (rowsum) and the RoPE-backward pass (conversion). Source: `torchtitan/models/common/attention/attention.py:426-536 (varlen_attn)`. Shapes: O, dO bf16 [8192,32,128]; dQ accum fp32 [8192,32,128]; dK/dV accum fp32.
- **Numbers:** 32 instances x 14.0 kernels; 129.4 us measured vs 82.7 us ideal per instance (572.5 MB external); gap 1.494 ms (0.42% of GPU-busy step). Evidence: estimated.
- **Dataflow:** dO (wo dgrad output) and O feed the preprocess; the postprocess outputs feed only RoPE backward / the dqkv cat.
- **Blockers:** Library-owned (flash_attn.cute via torch varlen_attn); changing the kernel boundary needs the library to expose fp32 accumulators or accept a fused epilogue.
- **Tier T3:** Inside the attention library; the dQ fp32 accumulation order is set by the main kernel and must be preserved.
- **Notes:** overlaps R-10. Measured times are the memset, preprocess and postprocess kernels only (main attention kernels excluded). Ideal bytes are estimated from the kernels' tensor shapes (accumulator layout assumed unpadded).

### R-20 lm_head grad accumulator aliased to the reduce-scatter buffer (last chunk GEMM + copy)

- **What / where:** The fp32 lm_head gradient [128256,4096] is copied by chunk_cat into an [8,65667072] buffer with the identical memory layout; letting the last chunk's addmm accumulate in (or the accumulator be) that buffer removes the 4.2 GB copy. Source: `torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py (foreach_reduce); linear/linear.py:238-250`. Shapes: grad fp32 [128256,4096] -> fp32 [8,65667072].
- **Numbers:** 1 instances x 2.0 kernels; 1737.8 us measured vs 646.4 us ideal per instance (4473.7 MB external); gap 1.091 ms (0.31% of GPU-busy step). Evidence: measured.
- **Dataflow:** chunk_cat input is the final accumulator and its output feeds only the reduce-scatter.
- **Blockers:** Requires FSDP to accept a pre-placed gradient in its reduce-scatter buffer (or to skip chunk_cat when the layout already matches).
- **Tier T0:** Removes a byte-for-byte fp32 copy; the GEMM is unchanged, so numerics are unchanged.
- **Notes:** overlaps R-19.

### R-24 Varlen attention metadata (cu_seqlens) from positions in one kernel

- **What / where:** Before the forward, the decoder builds packed cu_seqlens from positions with ~25 tiny ops (eq, cumsum, where, full, arange, scatter_, sums and async asserts); with deterministic algorithms scatter_ becomes a sort-based index_put taking ~0.65 ms for 8192 elements. Source: `torchtitan/models/common/attention/attention.py:1002-1100; decoder.py (preprocess_inputs)`. Shapes: positions int64 [8192] -> cu_seqlens int32 [19].
- **Numbers:** 1 instances x 40.0 kernels; 723.4 us measured vs 0.0 us ideal per instance (0.1 MB external); gap 0.723 ms (0.20% of GPU-busy step). Evidence: measured.
- **Dataflow:** All intermediates are consumed within the metadata builder; output is a [19] int32 cu_seqlens.
- **Blockers:** Deterministic mode routes scatter_ through index_put_ with sorting; a dedicated kernel (or a non-scatter formulation) is needed.
- **Tier T2:** Integer cumsum/scatter; exact. scatter_ order only matters for the discarded overflow slot.

### R-11 RoPE cache gather, position bounds check and cache conj computed once per step

- **What / where:** Every layer re-checks positions <= max position, gathers cache[positions] into [8192,1,64] complex64, and in backward re-materializes the conjugated cache twice; these depend only on positions and are identical for all 32 layers. Source: `torchtitan/models/common/rope.py:25-37,268-290,318-319; decoder.py:300-315 (one shared cache)`. Shapes: positions int64 [8192]; cache complex64 [8192,64].
- **Numbers:** 32 instances x 6.0 kernels; 20.0 us measured vs 0.1 us ideal per instance (0.5 MB external); gap 0.638 ms (0.18% of GPU-busy step). Evidence: measured.
- **Dataflow:** positions and the rope cache are the same tensors for every layer; the gathered cache and its conj have no layer-dependent input.
- **Blockers:** Each layer owns its RoPE module and cache buffer; hoisting needs the decoder to precompute the gathered cache (or the fused rope kernel to gather inline, which R-08/R-10 already assume).
- **Tier T0:** Pure gathers/compares/conj; bitwise identical.
- **Notes:** overlaps R-08, R-09, R-10. Ideal per instance is the once-per-step cost divided by 32.

### R-12 Attention residual add + ffn_norm RMSNorm

- **What / where:** h = x + attn_out is written to memory and immediately re-read by ffn_norm; a fused add+RMSNorm reads x and attn_out once and writes h (residual) and the normalized [8192,4096] output. Source: `torchtitan/models/llama3/model.py:57-61; torchtitan/models/common/nn_modules.py:135 (RMSNorm)`. Shapes: x, attn_out bf16 [8192,4096] -> h bf16 [8192,4096], normed bf16 [8192,4096], rstd fp32 [8192].
- **Numbers:** 32 instances x 2.0 kernels; 50.2 us measured vs 38.8 us ideal per instance (268.5 MB external); gap 0.367 ms (0.10% of GPU-busy step). Evidence: measured.
- **Dataflow:** h feeds ffn_norm and the next residual add; attn_out (wo output) feeds only the add.
- **Blockers:** Bitwise match of the quack RMSNorm reduction; the add sits in the block's forward between two modules.
- **Tier T1:** RMSNorm's row sum of squares must use the same reduction order as the existing quack RMSNorm kernel.
- **Notes:** overlaps R-14.

### R-13 FFN residual add + next layer's attention_norm (or final norm)

- **What / where:** The block output h + ffn_out of layer i is written and immediately normalized by layer i+1's attention_norm (the final model norm after layer 31); fusing the add into the next norm saves a write+read of [8192,4096]. Source: `torchtitan/models/llama3/model.py:57,64-66; torchtitan/models/common/decoder.py:338-352`. Shapes: h, ffn_out bf16 [8192,4096] -> out bf16 [8192,4096], normed bf16 [8192,4096], rstd fp32 [8192].
- **Numbers:** 32 instances x 2.0 kernels; 49.2 us measured vs 38.8 us ideal per instance (268.5 MB external); gap 0.333 ms (0.09% of GPU-busy step). Evidence: measured.
- **Dataflow:** The block output feeds the next block's attention_norm and its residual add only.
- **Blockers:** Crosses the transformer-block (and FSDP unit) boundary; eager module structure keeps them in different module calls.
- **Tier T1:** Same reduction-order requirement as R-12.
- **Notes:** overlaps R-14.

### R-21 Final norm backward fused with the chunked-loss gradient accumulation

- **What / where:** The chunked loss zero-fills an [8192,4096] gradient buffer and copies each chunk's hidden-state gradient into it, then the final RMSNorm backward reads it; the norm backward can consume the chunk gradients directly (row-wise), so the buffer fill and copies disappear. Source: `torchtitan/components/loss.py:625-672,879-902,957-981`. Shapes: chunk grads bf16 8x[1024,4096], x bf16 [8192,4096], rstd fp32 [8192] -> dx bf16 [8192,4096], dw fp32 [4096].
- **Numbers:** 1 instances x 12.0 kernels; 104.0 us measured vs 29.1 us ideal per instance (201.4 MB external); gap 0.075 ms (0.02% of GPU-busy step). Evidence: measured.
- **Dataflow:** The accumulation buffer is written by 8 copies and read once by the final norm's backward.
- **Blockers:** The chunk loop and the decoder backward are separate autograd graphs joined by _DecoderOutputGradientBackProp; the norm's weight gradient needs a cross-chunk reduction.
- **Tier T1:** Norm weight-gradient reduction over 8192 rows must keep the quack kernel's order; row-wise dx is order-free.

### R-23 Token embedding gather + layer-0 attention_norm

- **What / where:** The embedding gather writes h0 [8192,4096] which layer 0's attention_norm immediately re-reads; the norm can gather rows directly (still writing h0 for the residual). Source: `torchtitan/models/common/decoder.py:332; embedding.py:37; torchtitan/models/llama3/model.py:57`. Shapes: ids int64 [8192], table bf16 [128256,4096] -> h0, normed bf16 [8192,4096].
- **Numbers:** 1 instances x 2.0 kernels; 38.5 us measured vs 29.1 us ideal per instance (201.4 MB external); gap 0.009 ms (0.00% of GPU-busy step). Evidence: measured.
- **Dataflow:** h0 feeds attention_norm and the first residual add.
- **Blockers:** Crosses the embedding/decoder-layer and FSDP unit boundary.
- **Tier T1:** RMSNorm reduction order.

## 5. No opportunity

| what | ms | reason |
|---|---|---|
| backward layers.* host-to-device metadata memcpy (FSDP) x64 | 0.3294 | Host-to-device copies of FSDP split-size / chunk metadata (int64 [3831], [53288]); launch-latency bound, no device dataflow to fuse with. Removable only by caching the metadata on device. |
| forward lm_head split_with_sizes_copy x1 | 0.3095 | Single-kernel FSDP all-gather copy-out for groups without Shard(dim!=0) parameters (tok_embeddings, norm+lm_head); already one pass at ~6.8 TB/s (ideal 304 us vs 309 us measured). The backward re-gather of (norm, lm_head) only serves the norm weight -- a grouping issue, not a fusion one. |
| backward lm_head split_with_sizes_copy x1 | 0.3088 | Single-kernel FSDP all-gather copy-out for groups without Shard(dim!=0) parameters (tok_embeddings, norm+lm_head); already one pass at ~6.8 TB/s (ideal 304 us vs 309 us measured). The backward re-gather of (norm, lm_head) only serves the norm weight -- a grouping issue, not a fusion one. |
| forward tok_embeddings split_with_sizes_copy x1 | 0.3087 | Single-kernel FSDP all-gather copy-out for groups without Shard(dim!=0) parameters (tok_embeddings, norm+lm_head); already one pass at ~6.8 TB/s (ideal 304 us vs 309 us measured). The backward re-gather of (norm, lm_head) only serves the norm weight -- a grouping issue, not a fusion one. |
| forward layers.* host-to-device metadata memcpy (FSDP) x32 | 0.1000 | Host-to-device copies of FSDP split-size / chunk metadata (int64 [3831], [53288]); launch-latency bound, no device dataflow to fuse with. Removable only by caching the metadata on device. |
| backward lm_head host-to-device metadata memcpy (FSDP) x3 | 0.0154 | Host-to-device copies of FSDP split-size / chunk metadata (int64 [3831], [53288]); launch-latency bound, no device dataflow to fuse with. Removable only by caching the metadata on device. |
| backward tok_embeddings host-to-device metadata memcpy (FSDP) x1 | 0.0130 | Host-to-device copies of FSDP split-size / chunk metadata (int64 [3831], [53288]); launch-latency bound, no device dataflow to fuse with. Removable only by caching the metadata on device. |
| forward lm_head host-to-device metadata memcpy (FSDP) x1 | 0.0029 | Host-to-device copies of FSDP split-size / chunk metadata (int64 [3831], [53288]); launch-latency bound, no device dataflow to fuse with. Removable only by caching the metadata on device. |
| forward tok_embeddings host-to-device metadata memcpy (FSDP) x1 | 0.0028 | Host-to-device copies of FSDP split-size / chunk metadata (int64 [3831], [53288]); launch-latency bound, no device dataflow to fuse with. Removable only by caching the metadata on device. |
| backward lm_head _chunk_cat x1 | 0.0024 | Final-norm weight-gradient reduce-scatter copy-in (fp32 [4096]); 2 us, launch bound. |
| out of scope (grad clipping / metrics after backward): _foreach_mul_ | 1.2165 | Outside the forward/backward scope of this analysis (reported in the step budget as 'other'). |
| out of scope (grad clipping / metrics after backward): _foreach_norm | 0.6409 | Outside the forward/backward scope of this analysis (reported in the step budget as 'other'). |
| out of scope (batch H2D copy / token-count all-reduce before forward): _to_copy | 0.0112 | Outside the forward/backward scope of this analysis (reported in the step budget as 'other'). |
| out of scope (grad clipping / metrics after backward): _local_scalar_dense | 0.0107 | Outside the forward/backward scope of this analysis (reported in the step budget as 'other'). |
| out of scope (grad clipping / metrics after backward): clone | 0.0094 | Outside the forward/backward scope of this analysis (reported in the step budget as 'other'). |
| out of scope (grad clipping / metrics after backward): mul | 0.0078 | Outside the forward/backward scope of this analysis (reported in the step budget as 'other'). |

## 6. Observations about working from these artifacts

**What the artifacts made easy**

- Kernel -> launching op was mechanical: every GPU activity carries a CUDA-runtime `correlation` id, and nesting the CPU ops and `record_function`s by time gives the full op chain (e.g. `aten::copy_ < aten::_to_copy < aten::to < aten::type_as`). Only 3 of 3,310 GPU activities had no launch event.
- The `## mod <path> [<class>] ##` annotations give the forward module path and layer index directly, and the `FSDP::all_gather (layers.N)` / `FSDP::post_backward_reduce (layers.N)` annotations name the FSDP group for all communication glue, including backward prefetches.
- Joining Kineto ops to the execution trace by `rf_id` gave exact shapes, dtypes, strides and storage ids. Strides explained several slow kernels without any guessing: silu / silu_backward read `gate` with row stride 28672 (a strided half of `w13`'s [8192,2,14336] output, non-vectorized path at ~1.7 TB/s), and the FSDP `cat` before `_chunk_cat` reads eight [2,1792,4096] strided views of w13's gradient (~1.2 TB/s).
- Python stacks (`with_stack`) resolved the ops that run outside any module: the ~25 tiny ops plus a 0.65 ms deterministic `scatter_` before the root module call are `Decoder.preprocess_inputs -> create_varlen_metadata_for_document`.

**What was hard / needed workarounds**

- Backward module attribution. Kineto's `Sequence number` on `autograd::engine::evaluate_function: XBackward0` links to forward ops, but the number is shared by a whole forward op chain (e.g. `aten::rms_norm` and `aten::_fused_rms_norm`), so I took the outermost forward op with that number and its innermost module annotation. Two backward nodes (the final norm's `FusedRmsNormBackward0` and `EmbeddingBackward0`) carry no sequence number in either trace; they are unique per step, so I mapped them by node type. The execution trace's `seq_id` is -1 on the `evaluate_function` records and did not help.
- FSDP's backward hooks run inside whichever autograd node triggers them, so backward all-gathers show up under `AddBackward0` or `FusedRmsNormBackward0`, and the reduce-scatter copy-in under `RegisterPostBackwardFunctionBackward`, whose forward op is at the root module. Sequence-number mapping would attribute them to the wrong module; I overrode the module with the group named in the enclosing `FSDP::` annotation.
- All CPU ops (forward, autograd backward, the chunked loss's eight inner backwards) are on one thread, so thread ids could not separate phases. Phase = "has an `evaluate_function` (or FSDP backward) ancestor". The chunked loss's `GradAccumulator.add` copies run outside autograd nodes after each inner backward; I classified them as backward by their position in the loop.
- Byte accounting. In the execution trace, out= style ops (`split_with_sizes_copy`, `_chunk_cat`, `_foreach_copy_`, `addmm(out=)`) list their destination tensors among the inputs and have no outputs, and multi-kernel ops (flash-attention backward = memsets + preprocess + main + 3 postprocess kernels) share one op record. I therefore computed region external bytes analytically from the model dimensions and FSDP shard sizes (cross-checked against the op shapes), not by summing per-op bytes. My first flop parser misread `addmm(out=)` operands and showed the lm_head weight-gradient GEMM at 229 TFLOP/s; fixed (it runs at ~1.46 PFLOP/s, memory-bound on its 4.2 GB fp32 accumulator read+write per chunk).
- The tensor tuple layout in the execution trace (`[tensor_id, storage_id, offset, numel, itemsize, device]`) had to be inferred from examples.
- The capture begins with three multi-tensor-apply kernels (1.74 ms) of the previous step's optimizer that have no CPU launch in the window; excluded from the step.

**Evidence that was missing**

- No measured bytes per kernel (no hardware counters), so every ideal is analytic; L2 effects are invisible (the embedding gather appears faster than the HBM roofline, presumably from L2 hits on repeated tokens).
- Flash-attention's internal buffers (dQ/dK/dV fp32 accumulators, which are memset to [8,1605632] fp32, i.e. padded) are only visible as memset sizes; R-16's ideal is estimated.
- Communication is emulated by local copies, so real collective time and its overlap with these copy kernels are unknown; the GPU is also idle ~76% of the step (CPU/launch bound under the profiler: 19 ms wall per layer forward for 3.1 ms of kernels), so kernel timings are of isolated kernels and the measured gaps translate into wall time only once the run is GPU bound. In this regime, the kernel-count reduction of each fusion (e.g. 13 -> 1 for R-08, 40 -> 1 for R-24) matters as much as bytes.
- GEMMs run above the random-data peak (low-entropy fake-PG data), which makes GEMM-containing ideals pessimistic and hides some T3 fusions (below).

**How phase and module were identified for backward ops**

`kernel -> correlation -> runtime launch -> innermost CPU op`, then walk up: an `evaluate_function: X` ancestor marks backward; its `Sequence number` selects the forward op(s) with the same number, and the forward op's innermost `## mod ##` annotation is the module (layer index from the path). FSDP glue takes the group from the `FSDP::` annotation; the two sequence-less nodes are mapped by type; chunked-loss gradient copies by loop position.

**False leads discarded**

- The 7 x 54.5 MB device copies per all-gather and the reduce-scatter copy are the fake process group emulating collectives: excluded as communication.
- `aten::cat` inside `RegisterPostBackwardFunctionBackward` looked like an autograd gradient concat; it is FSDP re-laying out w13's [2,14336,4096] gradient because w13 is sharded on dim 1 (12.2 ms per step, part of R-01). The matching forward/backward `cat` after all-gather is the inverse (R-02).
- A slow-looking lm_head weight-gradient GEMM (parser bug above) -- really an fp32-accumulator-bound GEMM, kept as R-19/R-20.
- Horizontal fusion of each linear's dgrad and wgrad GEMMs (both read dY): both are compute bound, ~0 gain at roofline; dropped.
- Horizontally fusing independent FSDP copy kernels of different layers that run back to back: same bytes, no roofline gain; dropped.
- The backward re-gather of the whole (norm, lm_head) FSDP group (1 GB of lm_head weight) only to get the 4096-element norm weight for the final norm's backward: a grouping/resharding issue, not a fusion; noted under No opportunity.
- `out_THV.contiguous()` and the attention input `.to(bf16)` casts launch nothing.

**Opportunities I may have missed or did not list**

- GEMM-prologue fusions whose computed gap is negative because the GEMMs already beat the "peak": RMSNorm folded into the wqkv / w13 A-operand load, SwiGLU backward as a prologue of the w13 dgrad/wgrad GEMMs. They are real byte savings (one [8192,4096] or [8192,28672] write+read each) but not representable with this ideal.
- Fusions inside flash attention beyond its pre/post-processing; stream/overlap opportunities with real communication; CPU launch overhead (CUDA graphs), which dominates this capture.
- Fusions that change numerics (e.g. keeping weight gradients in fp32 without the bf16 round trip, fp32 logits straight from the lm_head GEMM) -- I only listed bitwise-reproducible designs, noting the required rounding points.
- The optimizer step and grad-norm clipping (out of scope; 4.2 ms fused AdamW, 1.9 ms clipping).

**Effort**

- Wall time: started 16:46:51 PST; attribution done by ~16:56, first region inventory logged at 16:56, regions.json / report.md first written ~17:06, final version 17:09 -- about 22 minutes, well inside the 80/90-minute limits.
- Scripts: `out/attribute.py` (attribution, op table, budget), `out/regions.py` (region selection, ideals, footprints, no-opportunity), `out/make_report.py` (this report). Intermediate: `out/kernels.jsonl`, `out/ops.jsonl`, `out/classes_raw.txt`, `out/regions_members.json`.


## 7. Reproduction

All scripts are CPU-only and read only the artifacts and `hw_profile.json`:

```bash
cd /home/bahuang/local/step_analysis/llama3_8b_eager
python out/attribute.py    # kernels.jsonl (one row per GPU activity), ops.jsonl (op instances + ET shapes), budget_raw.json
python out/regions.py      # regions.json (+ console summary, saved as out/regions_summary.txt)
python out/make_report.py  # report.md (uses out/observations.md for section 6)
```
- `out/attribute.py`: builds the CPU op/annotation nesting from the Kineto trace, maps each kernel through its CUDA-runtime correlation id to the launching op, takes the module from the innermost `## mod ... ##` annotation (forward) or from the forward op with the same autograd sequence number (backward), the FSDP group from `FSDP::` annotations, joins op instances to the execution trace by `rf_id` for shapes/dtypes/strides/storages, and computes the step budget.
- `out/regions.py`: selects each region's member op instances, computes measured time from kernel durations, external bytes / flops from the model dimensions, ideal and gap, footprints, overlaps and the no-opportunity list.
- `out/regions_log.jsonl`: append-only log of regions as they were identified. `out/classes_raw.txt`: first-pass kernel-class table.
