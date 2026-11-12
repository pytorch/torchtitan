# DeepSeek-V3 671B (eager, MXFP8, Dist-MoE): kernel-fusion opportunities in one training step

## 1. Setup

- **Program.** torchtitan eager `Trainer`, recipe `deepseek_v3_671b_dist_moe_mxfp8` with parallelism overridden to FSDP 256 x EP 64 (TP 1, PP 1, CP 1), one 4,096-token microbatch per step (packed varlen documents). 61 decoder blocks: blocks 0-2 dense SwiGLU FFN (F = 18,432), blocks 3-60 MoE (256 routed experts, top-8 with 8 groups / top-4 groups, sigmoid scores, 1 shared expert F = 2,048, 4 local experts per EP rank via Dist-MoE MXFP8 grouped GEMMs). MLA attention (q_lora 1536, kv_lora 512, 128 heads, qk 128+64, v 128) with the recipe's fused MLA RoPE/KV Triton kernels and FA4 varlen attention; fused SwiGLU Triton kernel; MXFP8 linears (torchao quantize + cuBLAS nvjet `_scaled_mm`) for attention, shared experts, dense FFN and lm_head (vocab 129,280); router gate in bf16 with fp32 output (HiMidLo backward); MicrobatchWiseLoadBalanceLoss aux loss; fp32 master weights, bf16 params/reduce; FSDP2 with 32x32-block MXFP8 weight quantization after all-gather.
- **Capture.** One process with a fake process group emulating rank 0 (collectives become local copies on side streams); deterministic algorithms; no activation recompute; CUDA graphs off; step 10 profiled after warmup.
- **Roofline.** GPU NVIDIA GB300; HBM 6921 GB/s (copy); GEMM peaks MXFP8 3615 TFLOP/s, BF16 1930 TFLOP/s (max of the measured shapes). Ideal = max(bytes / HBM, flops / peak). Region ideal = external bytes only (inputs read once, outputs written once). For GEMM/attention epilogue regions (tier T3, except the horizontal wq_a/wkv_a GEMM region), the host GEMM or attention kernel is not counted in measured or ideal; the ideal is only the bytes the epilogue/prologue adds, so the gap is what the fusion removes.

## 2. Step budget

| category | ms (kernel-time sum) | forward | backward |
|---|---:|---:|---:|
| gemm | 126.33 | 39.86 | 86.48 |
| attention | 88.76 | 17.88 | 70.88 |
| communication | 178.71 | 89.52 | 89.18 |
| memory_bound_glue | 294.28 | 112.19 | 182.09 |
| optimizer | 24.46 |  |  |
| other | 0.41 |  |  |

- Step GPU busy (union over all streams): **712.92 ms**; GPU idle inside the step's GPU span: **16470.22 ms**. The profiled step spans 17.2 s of wall time against 0.71 s of GPU busy time: it is host-bound under Python-stack capture and the execution-trace observer (train.log's step-10 throughput, 676 tok/s, is also host-bound), so idle time and stream overlap are capture artifacts, not properties of the program.
- Forward+backward busy: **688.04 ms** (26023 forward and 29582 backward GPU activities). Attribution coverage (phase + module + op resolved): **100.0%** of forward+backward busy time.
- Step window = GPU activity launched inside ProfilerStep#9 (GPU span 17183.1 ms; the profiled step was CPU-bound by the profiler, hence the idle time). by_category_ms sums kernel durations (streams overlap, so the sum exceeds busy time). Forward+backward GPU window 16588.1 ms with 15900.0 ms idle; forward+backward busy excluding emulated communication 509.34 ms. attributed_pct = union of kernels with resolved phase, module and op over forward+backward busy time; by kernel-duration sum it is 100.0%. 12.33 ms of FusedAdam kernels from the previous step's optimizer spill into the trace window and are excluded. optimizer includes gradient-norm clipping (4.98 ms) and AdamW.
- Communication here is the fake-PG emulation (device-to-device copies under `c10d::` ops); its time is reported but excluded from regions. GEMM time is MXFP8 nvjet GEMMs (near or above the random-data peak because of low-entropy data) plus Dist-MoE grouped GEMMs, which under the fake PG process only rank 0's locally routed rows and are therefore much cheaper than in a real EP run.
- Regions cover 287.2 ms of kernel time (memory-bound glue 294.28 ms, plus 7.9 ms of GEMM-category kernels in R-17/R-20); no-opportunity classes 15.0 ms. Total gap 164.5 ms (23.1% of step GPU busy time). Gaps are additive: every kernel belongs to exactly one region.

## 3. Ranked regions

| id | title | phases | scope | kind | inst | measured ms | ideal ms | gap ms | % step | tier |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| R-01 | FSDP all-gather copy-out fused with MXFP8 weight quantization of each block's unsharded weights | for+bac | communication-glue, cross-module, horizontal | copy elimination: gather-buffer de-interleave + block quantize + operand store in one multi-tensor kernel | 123 | 92.83 | 53.17 | 39.66 | 5.56 | T2 |
| R-02 | MoE router forward: sigmoid, biased group-limited top-k, gather/normalize, routing map, token counts and load-balance statistics | for | intra-module, cross-module, horizontal | vertical row-wise fusion with a cross-token reduction epilogue | 58 | 18.50 | 0.08 | 18.42 | 2.58 | T2 |
| R-03 | FSDP reduce-scatter copy-in: dim-1 shard reorder cat folded into chunk_cat | bac | communication-glue, horizontal | copy elimination (reorder + pack in one pass) | 62 | 39.70 | 22.94 | 16.76 | 2.35 | T2 |
| R-04 | Router + aux-loss backward to the gate-logit gradient | bac | intra-module, cross-module | vertical row-wise backward fusion | 58 | 10.45 | 0.08 | 10.37 | 1.45 | T1 |
| R-05 | FSDP pre-all-gather fp32->bf16 parameter cast fused with copy-in (multi-tensor) | for+bac | communication-glue, horizontal | vertical cast+copy and horizontal multi-tensor launch | 124 | 13.30 | 4.54 | 8.77 | 1.23 | T0 |
| R-06 | Routed-expert backward: fold score gradient and dy broadcast into the expert GEMMs (no [T,K,D] materialisation) | bac | gemm-adjacent, communication-glue | GEMM epilogue (dot-product reduction) + GEMM prologue (scaled broadcast) | 58 | 9.70 | 0.98 | 8.72 | 1.22 | T3 |
| R-07 | MLA KV backward pack fused with MXFP8 quantization of the packed KV gradient for wkv_b backward | bac | cross-module | vertical producer-consumer (pack -> quantize) | 61 | 11.93 | 5.40 | 6.53 | 0.92 | T1 |
| R-08 | ffn_norm backward with fused MoE input-gradient combine/accumulation, residual add and MXFP8 dy for wo | bac | cross-module, intra-module | vertical row-wise fusion (reduction over K + norm backward + quantize epilogue) | 61 | 13.79 | 7.53 | 6.27 | 0.88 | T1 |
| R-09 | Residual-stream update + RMSNorm + shared MXFP8 quantization of the next projections (cross-layer) | for | cross-layer, cross-module, horizontal | vertical row-wise fusion + horizontal CSE of duplicate quantization | 62 | 12.30 | 6.30 | 6.01 | 0.84 | T1 |
| R-10 | Forward-backward: produce the colwise MXFP8 copy of bf16-saved linear inputs in the forward quantize instead of requantizing in backward | for+bac | forward-backward, horizontal | forward-backward reuse (extra epilogue output saved for backward) | 62 | 7.11 | 1.21 | 5.91 | 0.83 | T1 |
| R-11 | attention_norm backward with fused wq_a/wkv_a gradient sum, residual add and the previous layer's w2 dy quantization (cross-layer) | bac | cross-layer, cross-module | vertical row-wise fusion with quantize epilogue | 62 | 7.48 | 3.15 | 4.33 | 0.61 | T1 |
| R-12 | Fused cross-entropy forward/backward on bf16 logits, emitting MXFP8 dlogits for lm_head backward | for+bac | forward-backward, cross-module | vertical fusion with recompute-in-backward (online softmax) | 1 | 4.19 | 0.46 | 3.73 | 0.52 | T1 |
| R-13 | MLA query RoPE applied in the wq_b GEMM epilogue instead of a full-tensor copy | for | gemm-adjacent, cross-module | GEMM epilogue | 61 | 3.59 | 0.00 | 3.59 | 0.50 | T3 |
| R-14 | Dist-MoE forward dispatch staging: padding mask, routing/activation copies and planning in fewer kernels | for | communication-glue, intra-module | vertical + horizontal fusion of copies and planning | 58 | 4.64 | 1.48 | 3.16 | 0.44 | T2 |
| R-15 | MLA key assembly (K-nope copy + shared rotated k_pe broadcast) in the wkv_b GEMM epilogue | for | gemm-adjacent, cross-module | GEMM epilogue with layout change | 61 | 3.09 | 0.60 | 2.50 | 0.35 | T3 |
| R-16 | Inverse query RoPE (backward) fused with MXFP8 quantization of dq for wq_b backward | bac | cross-module | vertical producer-consumer | 61 | 5.99 | 3.60 | 2.39 | 0.34 | T1 |
| R-17 | Dist-MoE grouped-GEMM prepare kernels folded into their GEMMs | for+bac | gemm-adjacent, intra-module | GEMM prologue | 116 | 2.03 | 0.00 | 2.03 | 0.29 | T3 |
| R-18 | HiMidLo router-gate backward: split, stack and weight-gradient accumulation in one pass | bac | intra-module, gemm-adjacent | vertical elementwise fusion + GEMM accumulate | 58 | 2.20 | 0.30 | 1.90 | 0.27 | T1 |
| R-19 | Flash-attention forward epilogue writes the MXFP8 copy of the attention output for wo | for | gemm-adjacent, cross-module | attention epilogue (extra quantized output) | 61 | 2.30 | 0.61 | 1.69 | 0.24 | T3 |
| R-20 | Horizontal GEMM fusion of wq_a and wkv_a (shared input) in forward and input-gradient | for+bac | horizontal, gemm-adjacent | horizontal GEMM concatenation | 61 | 5.87 | 4.18 | 1.69 | 0.24 | T3 |
| R-21 | Attention residual add + ffn_norm + w13 MXFP8 quantization | for | cross-module, intra-module | vertical row-wise fusion with quantize epilogue | 61 | 3.85 | 2.34 | 1.51 | 0.21 | T1 |
| R-22 | SwiGLU backward fused with MXFP8 quantization of d(h13) for w13 backward | bac | cross-module | vertical producer-consumer | 61 | 2.45 | 1.04 | 1.41 | 0.20 | T1 |
| R-23 | kv_norm backward + split backward cat + wkv_a dy quantization | bac | cross-module, intra-module | vertical row-wise fusion | 61 | 1.39 | 0.12 | 1.26 | 0.18 | T1 |
| R-24 | Embedding backward scattered directly into the reduce-scatter buffer | bac | communication-glue, intra-module | copy elimination (scatter into packed layout) | 1 | 1.43 | 0.28 | 1.15 | 0.16 | T2 |
| R-25 | SwiGLU forward fused with MXFP8 quantization of the w2 input | for | cross-module | vertical producer-consumer | 61 | 1.65 | 0.62 | 1.02 | 0.14 | T1 |
| R-26 | q_norm backward fused with wq_a dy quantization | bac | cross-module | vertical row-wise fusion | 61 | 1.34 | 0.34 | 1.00 | 0.14 | T1 |
| R-27 | kv_norm forward (without the contiguous clone) fused with wkv_b MXFP8 quantization | for | cross-module, intra-module | vertical row-wise fusion | 61 | 0.84 | 0.07 | 0.77 | 0.11 | T1 |
| R-28 | tok_embeddings: embedding lookup directly from the gathered buffer (skip the full-table copy-out) | for | communication-glue | copy elimination (gather from packed layout) | 1 | 0.56 | 0.02 | 0.55 | 0.08 | T0 |
| R-29 | Dist-MoE backward staging: backward plan + activation-to-dispatch copy | bac | communication-glue, intra-module | horizontal fusion of planning with a copy | 58 | 1.45 | 0.98 | 0.47 | 0.07 | T0 |
| R-30 | RoPE position bound check hoisted out of the per-layer attention (cross-layer) | for | cross-layer | cross-layer common-subexpression elimination | 61 | 0.46 | 0.00 | 0.46 | 0.06 | T1 |
| R-31 | q_norm forward fused with wq_b MXFP8 quantization | for | cross-module | vertical row-wise fusion | 61 | 0.72 | 0.28 | 0.45 | 0.06 | T1 |

`% step` is relative to step GPU busy time. Scope counts: cross-module 19, intra-module 11, communication-glue 8, horizontal 7, gemm-adjacent 7, cross-layer 3, forward-backward 2.

## 4. Region details

### R-01: FSDP all-gather copy-out fused with MXFP8 weight quantization of each block's unsharded weights

After each transformer block's parameter all-gather (forward unshard and backward re-unshard), the rank-major gathered bf16 buffer is de-interleaved into per-parameter bf16 tensors, w13 is reassembled and made contiguous, then every MXFP8 linear weight is quantized to 32x32-block MXFP8 into a temporary and copied into the persistent operand storage (expert w13/w2 likewise via Dist-MoE weight preparation). One kernel per block could read the gathered buffer once and write the persistent MXFP8 operands (and bf16 only for the unquantized norm/router/bias parameters). Where: torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:467-510 (_default_all_gather_output_fn); torchtitan/quantization/_fsdp_tensor.py:411-460 (fsdp_post_all_gather); torchtitan/quantization/mxfp8/tensor.py:71-110 (_build_operands copy_ refill); dist_moe/_blockscaled.py:3014 (_prepare_blockscaled_weight); 123 x 42.21 kernels, 754.7 us measured vs 432.29 us ideal per instance. Dataflow: all_gather output -> split_with_sizes_copy -> (cat reassembly, clone for dim-1 shards) -> triton 32x32 quantize/swizzle (temp) -> copy_ into persistent operands; FSDP releases the bf16 per-parameter tensors right after quantization, so the bf16 round trip and the temp->persistent copy have no other consumer. Blockers: Spans FSDP2 internals (copy-out runs before fsdp_post_all_gather), torchtitan's MXFP8 FSDP tensor subclass and Dist-MoE's weight preparation; needs a format-aware all-gather output hook. Same work is repeated for the backward re-unshard (reshard_after_forward); keeping the MXFP8 operands from forward to backward would remove the backward half entirely at a memory cost. Tier T2: Pure data movement plus per-32x32-block amax (max is order-independent, exact). Bitwise identity only needs the same block-scale rounding as the torchao/Dist-MoE quantizers and correct rank-major addressing. Adjacent: R-05, R-28.

### R-02: MoE router forward: sigmoid, biased group-limited top-k, gather/normalize, routing map, token counts and load-balance statistics

From the fp32 gate logits [4096, 256], the router runs ~47 small kernels per MoE layer: sigmoid, expert-bias add, per-group top-2 and group sum, group top-4, mask, top-8, gather of unbiased scores, normalisation and scaling, one-hot routing map, token-per-expert counts, and the microbatch load-balance loss statistics (counts, L1-normalised score sums, f*p). One row-wise kernel (plus a small cross-token reduction) produces all of them. Where: torchtitan/models/common/moe/moe.py:233-330 (TokenChoiceTopKRouter.forward), 499-608 (MicrobatchWiseLoadBalanceLoss); torchtitan/models/deepseek_v3/moe.py:31-83 (_select_experts); 58 x 47.0 kernels, 319.04 us measured vs 1.42 us ideal per instance. Dataflow: gate mm (fp32 out) -> sigmoid -> +bias -> topk(2) per group -> sum -> topk(4 groups) -> scatter mask -> masked_fill -> topk(8) -> gather(scores) -> sum/div/mul; scores also -> aux-loss normalize/sum; routing_map -> sum -> tokens_per_expert add_. All intermediates are [4096,<=256] and consumed only inside the router. Blockers: torch.topk on [4096,256] costs ~72 us per call (radix-select gatherTopK) and three top-k calls dominate; deterministic mode; the aux-loss reduction crosses tokens (needs a second pass or atomics with fixed order). Tier T2: Top-k selection with sorted=False must reproduce torch.topk's tie-breaking and output order (order-sensitive); the aux-loss column sums over 4096 tokens need the reduce_kernel's order (T1). Adjacent: R-04.

### R-03: FSDP reduce-scatter copy-in: dim-1 shard reorder cat folded into chunk_cat

In each block's post-backward, the shared-expert (or dense FFN) w13 gradient [2, F, 7168], sharded on dim 1, is chunked into 256 pieces and concatenated (four batched-cat launches over 512 strided inputs) before chunk_cat copies all gradients into the rank-major reduce-scatter buffer. A chunk_cat that addresses each gradient along its own shard dimension writes the buffer in one pass. Where: torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:544-556 (_default_reduce_scatter_input_fn), 870 (foreach_reduce_scatter_copy_in -> fsdp::chunk_cat); 62 x 7.81 kernels, 640.34 us measured vs 370.03 us ideal per instance. Dataflow: unsharded grads (autograd outputs) -> torch.cat(chunks along dim 1) -> _chunk_cat into flat RS input -> reduce_scatter; the cat output is consumed only by chunk_cat. Blockers: FSDP2 owns the copy-in; needs per-tensor shard-dim support in fsdp::chunk_cat (upstream change). The chunk_cat kernel itself runs at 0.65-0.9x of its own roofline for the 1.85 GB lm_head gradient. Tier T2: Order-defined data movement, no arithmetic.

### R-04: Router + aux-loss backward to the gate-logit gradient

The backward of the top-k score masking, aux-loss injection, normalisation, scaling, the gather of scores (deterministic scatter-add implemented with a radix sort), the L1-normalisation of the aux loss and the sigmoid runs ~59 kernels per MoE layer to produce d(gate logits) [4096,256] fp32 from d(topk scores) [4096,8]. Each token row's gradient depends only on that row (plus the scalar aux-loss gradient), so one row-wise kernel suffices. Where: torchtitan/models/common/moe/moe.py:233-330, 499-608; torchtitan/models/common/aux_loss.py (inject); 58 x 59.0 kernels, 180.13 us measured vs 1.42 us ideal per instance. Dataflow: d topk_scores -> MaskedFillBackward -> MulBackward(route_scale) -> DivBackward(norm) -> SumBackward -> GatherBackward (scatter into [4096,256]) + aux-loss path (Mul/Div/Expand/ClampMin/LinalgVectorNorm backward over [4096,256]) -> add_ -> SigmoidBackward -> d gate logits. Blockers: Spread over autograd nodes of several modules (routed_experts mask, router, aux_loss); requires a custom autograd Function for the router; deterministic scatter_add path for gather_backward. Tier T1: Row sums over K=8 and over E=256 (normalize backward) must keep the reduce order; the gather backward is collision-free (top-k ids are unique per row), so the deterministic sort-based scatter-add can become a direct store with identical results. Adjacent: R-02, R-18.

### R-05: FSDP pre-all-gather fp32->bf16 parameter cast fused with copy-in (multi-tensor)

Before each block's all-gather (forward and backward), every fp32 parameter shard is cast to bf16 by its own kernel (about 10 per block, plus a contiguous clone for dim-1-sharded w13), then _foreach_copy_ copies the bf16 shards into the flat all-gather input. One multi-tensor kernel can read the fp32 shards and write bf16 directly into the all-gather input. Where: torchtitan/quantization/_fsdp_tensor.py:352-409 (fsdp_pre_all_gather: movedim/.to(dtype).contiguous()); torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:440-452 (_foreach_copy_ into flat buffer); 124 x 12.69 kernels, 107.29 us measured vs 36.57 us ideal per instance. Dataflow: fp32 sharded params -> _to_copy(bf16) -> (clone for movedim'd shards) -> _foreach_copy_ into flat all-gather input -> all_gather; intermediates have no other consumer. Blockers: The cast happens inside torchtitan's FSDP tensor-subclass pre-all-gather hook while the copy-in is FSDP2's; requires the hook to return a dtype-converting view or FSDP to cast during copy-in. Also repeated in the backward re-unshard although parameters did not change. Tier T0: Elementwise round-to-nearest cast and copies only. Adjacent: R-01.

### R-06: Routed-expert backward: fold score gradient and dy broadcast into the expert GEMMs (no [T,K,D] materialisation)

broadcast_and_scale reads dy [4096,7168], the recomputed per-slot expert outputs x [4096,8,7168] and the scores, writing dx = dy*score for every (token, k) slot [4096,8,7168] and dscore = <dy, x>. The score gradient can be computed in the epilogue of the recompute fprop GEMM that produces x, and the scaled broadcast can be generated in the dgrad GEMM's operand load, so neither 470 MB tensor needs to be written and re-read. Where: dist_moe/_postprocess.py:552-615 (_reduction_backward -> broadcast_and_scale); dist_moe/kernels/triton/broadcast_n_reduction.py:386; 58 x 1.0 kernels, 167.31 us measured vs 16.99 us ideal per instance. Dataflow: recompute fprop GEMM -> x[T,K,D] (written to the combine buffer) -> broadcast_and_scale(dy, scores, x) -> dx[T,K,D] (dispatch buffer) -> dgrad GEMM reads dx from peers. Blockers: Dist-MoE owns the kernels; in real EP the expert GEMM runs on the expert-owning rank, so its epilogue needs the source token's dy (remote read over symmetric memory). Fake-PG GEMM times here process only rank 0's local rows, so the GEMM side is not representative. Tier T3: Needs grouped GEMM epilogue/prologue changes; dscore is a 7168-long dot product whose summation order must match the Triton kernel. Adjacent: R-11.

### R-07: MLA KV backward pack fused with MXFP8 quantization of the packed KV gradient for wkv_b backward

The fused KV-backward kernel packs dK-nope and dV into the bf16 kv gradient [4096,128,256] and reduces/inverse-rotates dK-rope; wkv_b's backward then re-reads that 268 MB tensor to quantize it rowwise and colwise to MXFP8 (plus two scale swizzles). The pack kernel can emit both MXFP8 layouts directly; the bf16 kv gradient has no other consumer. Where: torchtitan_recipes/overrides/fused_mla.py:409-536 (_fused_kv_backward_kernel), 911-922; torchtitan/quantization/mxfp8/linear.py:233 (_MXFP8LinearFunction.backward); 61 x 4.0 kernels, 195.65 us measured vs 88.55 us ideal per instance. Dataflow: flash-attn backward -> dK, dV -> _fused_kv_backward -> grad_kv (bf16) -> mxfp8_quantize (rowwise+colwise) -> wkv_b dgrad and wgrad GEMMs. Blockers: The KV pack is a recipe Triton kernel and the quantizer is torchao's; MXFP8LinearFunction would need to accept a pre-quantized grad_output. Tier T1: dK-rope reduction over 128 heads must keep the existing kernel's order; MXFP8 amax per 32 elements is exact.

### R-08: ffn_norm backward with fused MoE input-gradient combine/accumulation, residual add and MXFP8 dy for wo

The gradient of ffn_norm's output arrives as three partial gradients: the routed experts' per-slot dgrad [4096,8,7168] summed over K, the shared-expert w13 dgrad and the router gate dgrad, accumulated by two separate adds. Then RMSNorm backward, a separate residual add_, and wo's backward re-reads the result to quantize it. One row-wise kernel can sum the partials, run the norm backward, add the residual gradient and write both bf16 and MXFP8 (rowwise+colwise) outputs. Where: dist_moe/_postprocess.py:502-551 (_scale_and_sum in backward); torchtitan/models/deepseek_v3/model.py:41-63; torchtitan/quantization/mxfp8/linear.py:233; 61 x 7.85 kernels, 226.13 us measured vs 123.35 us ideal per instance. Dataflow: dgrad combine buffer -> scale_and_sum -> + shared w13 dgrad (add) -> + gate dgrad (add_) -> rmsnorm_backward -> + residual grad (add_) -> mxfp8_quantize -> wo dgrad/wgrad GEMMs; the bf16 result also flows to attention_norm backward's residual add. Blockers: Partial gradients come from different autograd nodes (Dist-MoE custom op, MXFP8 linear, HiMidLo linear), so fusing needs a custom backward for the whole MoE block; scale_and_sum is a Dist-MoE kernel. Tier T1: K-sum order of scale_and_sum, the autograd accumulation order ((shared + routed) + gate), the RMSNorm-backward row reduction and the weight-gradient column partials must be matched. Adjacent: R-11.

### R-09: Residual-stream update + RMSNorm + shared MXFP8 quantization of the next projections (cross-layer)

At the end of each MoE block, scale_and_sum combines the 8 routed expert outputs [4096,8,7168] with the router scores, a separate add adds the shared-expert output and another adds the residual; the next block's attention_norm then normalises the result, and wq_a and wkv_a each quantize the same normed tensor to MXFP8 (two identical quantizations). One row-wise kernel can do the combine, both adds, the RMSNorm and a single quantization (also the final norm feeding lm_head). Where: dist_moe/_postprocess.py:502-551 (_scale_and_sum); torchtitan/models/common/moe/moe.py:823-857 (shared add); torchtitan/models/deepseek_v3/model.py:41-63; torchtitan_recipes/overrides/fused_mla.py:1026-1070; torchtitan/quantization/mxfp8/linear.py:77; 62 x 7.82 kernels, 198.44 us measured vs 101.55 us ideal per instance. Dataflow: combine buffer -> scale_and_sum -> + shared (add) -> + residual (add) -> [next block] rmsnorm -> mxfp8_quantize (wq_a) and mxfp8_quantize (wkv_a) of the same tensor -> GEMMs. Blockers: Crosses the block boundary and the FSDP unit boundary (next block's pre-forward all-gather hooks fire in between); scale_and_sum lives in Dist-MoE; the two quantizations are inside two separate MXFP8Linear modules (needs a shared-input quantization cache). Tier T1: K-sum order of scale_and_sum, the add order (routed + shared, then residual) and the RMSNorm row reduction must be matched; quantize amax is exact. Adjacent: R-21, R-10.

### R-10: Forward-backward: produce the colwise MXFP8 copy of bf16-saved linear inputs in the forward quantize instead of requantizing in backward

wq_a, wkv_a, wq_b, wo, shared/dense w13 and lm_head save their bf16 input and re-read it in backward to quantize colwise for the weight-gradient GEMM; wq_a and wkv_a even quantize the same normed tensor twice. The forward quantization kernel already reads that input and can write the colwise MXFP8 layout too, which is saved instead of being recomputed. Where: torchtitan_recipes/models/deepseek_v3.py:64-87 (linears_saving_inputs_for_backward_in_mxfp8); torchtitan/quantization/mxfp8/linear.py:77-233; 62 x 14.81 kernels, 114.71 us measured vs 19.44 us ideal per instance. Dataflow: forward: normed/attention output -> rowwise quantize -> fprop GEMM, bf16 saved; backward: saved bf16 -> colwise quantize (+2 swizzles) -> wgrad GEMM. Blockers: Changes the saved-activation format (memory and recompute policy); the recipe keeps bf16 for shared-input and attention-output projections on purpose; the bf16 normed tensor is not needed by the norm backward, so saving MXFP8 instead reduces memory. Tier T1: Same quantizer on the same bf16 tensor gives identical bytes; only the place it runs changes. Adjacent: R-09, R-19.

### R-11: attention_norm backward with fused wq_a/wkv_a gradient sum, residual add and the previous layer's w2 dy quantization (cross-layer)

wq_a's and wkv_a's input gradients are summed by an add, attention_norm's RMSNorm backward runs as two kernels, a separate add_ accumulates the residual gradient, and the previous block's shared-expert (or dense) w2 backward then re-reads the result to quantize it to MXFP8. One row-wise kernel can produce the bf16 residual gradient and its MXFP8 layouts. Where: torchtitan/models/deepseek_v3/model.py:41-63; torchtitan/quantization/mxfp8/linear.py:233; 62 x 6.92 kernels, 120.57 us measured vs 50.75 us ideal per instance. Dataflow: wq_a dgrad + wkv_a dgrad (add) -> rmsnorm_backward -> + residual (add_) -> [previous block] shared_experts.w2 mxfp8_quantize(dy) -> w2 dgrad/wgrad; the bf16 result also feeds the previous block's routed-expert broadcast_and_scale and its ffn_norm residual add. Blockers: Crosses the block/FSDP-unit boundary (post-backward reduce-scatter and pre-backward all-gather hooks fire in between); partial gradients come from two MXFP8Linear autograd nodes. Tier T1: Accumulation order of the two dgrads and RMSNorm-backward reductions must be matched. Adjacent: R-08, R-06.

### R-12: Fused cross-entropy forward/backward on bf16 logits, emitting MXFP8 dlogits for lm_head backward

The loss upcasts the bf16 logits [4096,129280] to fp32, runs log_softmax (writes a 2.1 GB fp32 tensor saved for backward) and nll_loss; backward zero-fills a 2.1 GB gradient, scatters -1/N, runs the log-softmax backward, downcasts to bf16 and lm_head's backward re-reads it to quantize. A fused cross-entropy keeps only the per-row logsumexp from the forward and writes the MXFP8 (rowwise+colwise) logit gradient directly in the backward. Where: torchtitan/components/loss.py:54-92 (cross_entropy_loss: pred.float()), 345; torchtitan/quantization/mxfp8/linear.py:233; 1 x 12.0 kernels, 4192.61 us measured vs 463.84 us ideal per instance. Dataflow: lm_head GEMM -> logits -> _to_copy fp32 -> _log_softmax -> nll_loss -> loss; backward: zero_ + nll_loss_backward -> _log_softmax_backward_data -> _to_copy bf16 -> mxfp8_quantize -> lm_head dgrad/wgrad. Blockers: Crosses the loss/model boundary and lm_head's MXFP8 autograd Function (needs pre-quantized grad_output); could also be the lm_head GEMM epilogue (T3). Tier T1: Row max/sum-exp reductions over 129280 must reproduce the softmax kernel's order for bitwise identity.

### R-13: MLA query RoPE applied in the wq_b GEMM epilogue instead of a full-tensor copy

fused_mla_q rotates the 64 rope dims of each of 128 heads but, being out of place, copies the whole query [4096,128,192] bf16 (201 MB read + 201 MB write). The wq_b GEMM epilogue can rotate the rope columns in registers before storing q. Where: torchtitan_recipes/overrides/fused_mla.py:833-856 (_FusedMLAQ.forward, deliberately out of place), 1050-1062; 61 x 1.0 kernels, 58.83 us measured vs 0.0 us ideal per instance. Dataflow: wq_b scaled_mm -> q (bf16) -> fused_mla_q_rope (copy + rotate) -> flash attention. Blockers: cuBLAS nvjet MXFP8 GEMM has no custom epilogue; requires a CUTLASS/CuTe GEMM. An in-place rotation of only the rope columns (134 MB, ~19 us) is the non-GEMM fallback but the recipe avoids it because autograd's CopySlices would materialise the projection. Tier T3: Needs the MXFP8 GEMM epilogue; the rotation itself is pointwise (fp32 math then bf16 round must match the Triton kernel).

### R-14: Dist-MoE forward dispatch staging: padding mask, routing/activation copies and planning in fewer kernels

Before the routed-expert GEMMs, two masked_fills apply the padding mask to the top-k ids and scores, the normed tokens and routing are copied into the symmetric dispatch buffer, five small planning kernels (metadata gather, routing split/prefix/combine, forward plan) run, the dispatch buffer is copied into the activation buffer and a cat packs scalars. These read the same routing tensors and the same 58.7 MB token block. Where: torchtitan/models/common/dist_moe/routed_experts.py:166-243; dist_moe/kernels/triton/conditional_copy.py:303-530; dist_moe/kernels/triton/dist_dispatch_routing_kernel.py:421; 58 x 13.0 kernels, 80.02 us measured vs 25.51 us ideal per instance. Dataflow: router outputs -> masked_fill x2 -> copy_routing_and_dispatch (x -> dispatch buffer) -> routing split/prefix/combine -> plan -> copy_dispatch_to_activation -> grouped GEMMs. Blockers: Dist-MoE internals and the EP synchronisation protocol (barriers between planning steps); under the fake PG some kernels only see rank 0's rows. Tier T2: Slot assignment uses prefix sums over routed tokens (integer, order-defined). Adjacent: R-21, R-02.

### R-15: MLA key assembly (K-nope copy + shared rotated k_pe broadcast) in the wkv_b GEMM epilogue

fused_mla_kv materialises K [4096,128,192] by copying the nope half of each head from the wkv_b output [4096,128,256] and broadcasting the rotated shared k_pe; V stays a view. The wkv_b GEMM epilogue can write K (with the rope tail) and V directly. Where: torchtitan_recipes/overrides/fused_mla.py:892-909 (_FusedMLAKV.forward), 982-998; 61 x 1.0 kernels, 50.72 us measured vs 9.77 us ideal per instance. Dataflow: wkv_b scaled_mm -> kv -> fused_mla_k_rope (+ k_pe from wkv_a) -> K, V -> flash attention. Blockers: cuBLAS GEMM epilogue; K and V would become separate allocations (V no longer a free view); alternatively an MLA-aware attention kernel reading K-nope and k_pe directly. Tier T3: Copy + pointwise rotation; needs a GEMM with custom epilogue.

### R-16: Inverse query RoPE (backward) fused with MXFP8 quantization of dq for wq_b backward

The query RoPE backward rotates dq [4096,128,192] in place, then wq_b's backward re-reads it to quantize rowwise and colwise (three scale swizzles). One kernel can rotate and quantize in a single pass; the bf16 dq has no other consumer. Where: torchtitan_recipes/overrides/fused_mla.py:858-871 (_FusedMLAQ.backward); torchtitan/quantization/mxfp8/linear.py:233; 61 x 3.0 kernels, 98.26 us measured vs 59.09 us ideal per instance. Dataflow: flash-attn backward postprocess -> dq -> fused_mla_q_rope_ (in place) -> mxfp8_quantize -> wq_b dgrad/wgrad. Blockers: Recipe kernel + torchao quantizer; MXFP8LinearFunction must accept pre-quantized grad_output. Extending into the flash-attention dQ postprocess (fp32 accumulator -> MXFP8) would also remove the bf16 dq write (T3). Tier T1: Pointwise rotation + exact amax.

### R-17: Dist-MoE grouped-GEMM prepare kernels folded into their GEMMs

Each routed-expert grouped GEMM (forward fprop dispatch and SwiGLU-combine GEMMs, their backward recompute, dgrad and wgrad GEMMs) is preceded by a separate ~5 us 'prepare' kernel launched by Dist-MoE. Folding the preparation into the persistent GEMM's prologue removes 2 launches per MoE layer in forward and 6 in backward. Where: dist_moe/kernels/dist_blockscaled_grouped_gemm.py, dist_moe/kernels/blockscaled_grouped_gemm.py (prepare_* CuTe kernels); 116 x 4.0 kernels, 17.53 us measured vs 0.0 us ideal per instance. Dataflow: Dist-MoE plan -> prepare kernel -> grouped GEMM (consumes the prepared state only). Blockers: Dist-MoE CuTe kernels; the prepare step may exist to publish state to peers before the dispatch GEMM starts (cross-rank ordering). Tier T3: Inside the grouped GEMM implementation; no floating-point arithmetic is reordered if the prepare work moves into the GEMM prologue. Adjacent: R-14, R-29.

### R-18: HiMidLo router-gate backward: split, stack and weight-gradient accumulation in one pass

The gate's fp32 output gradient [4096,256] is split into three bf16 pieces with eight elementwise kernels and concatenated, the bf16 weight is stacked three times, and the weight gradient of three GEMMs is accumulated with two add_ kernels and a final cast. A single split kernel writing the stacked operand directly (and GEMM accumulation for dW) removes most of these launches. Where: torchtitan/models/common/linear/hi_mid_lo_linear.py:121-380 (_split_into_bf16_pieces); 58 x 14.0 kernels, 38.0 us measured vs 5.23 us ideal per instance. Dataflow: router backward -> d gate (fp32) -> _to_copy/__lshift__/sub x2 -> cat -> mm (dgrad); cat(weight x3); mm x3 + add_ x2 -> _to_copy -> weight grad. Blockers: Small tensors: launch-bound (~2-4 us per kernel). Tier T1: Piece split is exact bit manipulation; the dW sum over three pieces must keep its order (or move into GEMM beta=1 accumulation, T3). Adjacent: R-04.

### R-19: Flash-attention forward epilogue writes the MXFP8 copy of the attention output for wo

wo's forward re-reads the attention output [4096,16384] bf16 to quantize it to MXFP8 rowwise and swizzle its scales. The attention forward already holds each output tile and can write the MXFP8 layout next to the bf16 output (which backward still needs). Where: torchtitan_recipes/overrides/fused_mla.py:1099-1108; torchtitan/quantization/mxfp8/linear.py:77; 61 x 2.0 kernels, 37.72 us measured vs 10.0 us ideal per instance. Dataflow: flash-attn forward -> O (bf16, saved) -> mxfp8_quantize -> wo scaled_mm. Blockers: FA4 kernel is external (flash_attn.cute); per-head output tiles must cover whole 32-element blocks (they do: head dim 128). Tier T3: Requires changing the FA4 CuTe kernel epilogue; amax exact. Adjacent: R-10.

### R-20: Horizontal GEMM fusion of wq_a and wkv_a (shared input) in forward and input-gradient

wq_a [7168->1536] and wkv_a [7168->576] read the same normed input; their forward GEMMs can be one GEMM with N=2112, and their input-gradient GEMMs (whose outputs are summed) one GEMM with K=2112 over the concatenated output gradients, which also removes the summing add. Where: torchtitan_recipes/overrides/fused_mla.py:1033-1066; torchtitan/quantization/mxfp8/linear.py:77-233; 61 x 4.0 kernels, 96.27 us measured vs 68.61 us ideal per instance. Dataflow: attention_norm -> quantize (shared) -> wq_a GEMM, wkv_a GEMM; backward: q_norm-bwd output -> wq_a dgrad, kv_norm-bwd/cat output -> wkv_a dgrad -> add. Blockers: Two separate MXFP8Linear modules and FSDP parameters (weights would need a stacked parameter or a grouped GEMM); the dgrad merge changes numerics unless done as a grouped GEMM with a fused bf16 add. Tier T3: The merged dgrad accumulates both products inside one GEMM K-loop, which changes rounding versus two GEMMs plus a bf16 add (not bitwise without a split-K order match); the forward concatenation is bitwise-safe. Adjacent: R-09, R-11.

### R-21: Attention residual add + ffn_norm + w13 MXFP8 quantization

After wo, the block adds the attention output to the residual, ffn_norm normalises it and shared_experts.w13 (or the dense w13) re-reads the normed tensor to quantize it. One row-wise kernel can write the residual, the bf16 normed tensor (saved, and read by the router gate and the dispatch copy) and its MXFP8 layout. Where: torchtitan/models/deepseek_v3/model.py:51-60; torchtitan/quantization/mxfp8/linear.py:77; 61 x 4.0 kernels, 63.05 us measured vs 38.31 us ideal per instance. Dataflow: wo scaled_mm -> + residual (add) -> rmsnorm -> mxfp8_quantize (w13), normed also -> router gate mm, Dist-MoE dispatch copy. Blockers: Three modules (block, ffn_norm, MXFP8Linear); the add could alternatively be wo's GEMM epilogue (T3). Tier T1: RMSNorm row reduction order; add and amax exact. Adjacent: R-09, R-14.

### R-22: SwiGLU backward fused with MXFP8 quantization of d(h13) for w13 backward

silu_and_mul_backward writes d(h13) [4096, 2F] bf16 and w13's backward re-reads it to quantize rowwise and colwise (three swizzles). The SwiGLU backward kernel can emit the MXFP8 layouts directly; the bf16 gradient has no other consumer. Where: torchtitan_recipes/overrides/fused_swiglu.py:94-160, 274-300; torchtitan/quantization/mxfp8/linear.py:233; 61 x 3.0 kernels, 40.16 us measured vs 17.1 us ideal per instance. Dataflow: w2 dgrad GEMM -> dh -> silu_and_mul_backward(h13) -> d h13 -> mxfp8_quantize -> w13 dgrad/wgrad. Blockers: Recipe Triton kernel + torchao quantizer; MXFP8LinearFunction must accept pre-quantized grad_output. Tier T1: Pointwise math + exact amax.

### R-23: kv_norm backward + split backward cat + wkv_a dy quantization

kv_norm's backward clones its non-contiguous input slice, runs two kernels, the split backward concatenates d(kv_latent) [4096,512] with d(k_pe) [4096,64], and wkv_a's backward quantizes the [4096,576] result. One kernel can read the slice in place, write into the [4096,576] gradient and emit MXFP8. Where: torchtitan_recipes/overrides/fused_mla.py:1064-1090; torchtitan/quantization/mxfp8/linear.py:233; 61 x 6.0 kernels, 22.73 us measured vs 1.99 us ideal per instance. Dataflow: wkv_b dgrad -> d kv_normed -> clone(x) + rmsnorm_backward (+reduce_rows) -> cat with dk_pe -> mxfp8_quantize -> wkv_a GEMMs. Blockers: Launch-bound small tensors spread over three autograd nodes. Tier T1: RMSNorm-backward reductions (row and weight column) order.

### R-24: Embedding backward scattered directly into the reduce-scatter buffer

The deterministic embedding backward zero-fills a [129280, 7168] bf16 gradient (1.85 GB) and accumulates the 4096 token rows with a sort-based scatter, then FSDP's chunk_cat copies the whole table into the reduce-scatter buffer. Zero-filling the reduce-scatter buffer and scattering the rows into its rank-major layout removes the full-table copy. Where: torchtitan/models/common/embedding.py:37-45; torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:870; 1 x 23.0 kernels, 1427.05 us measured vs 276.27 us ideal per instance. Dataflow: d embedding output -> embedding_backward (memset + sort + scatter-add) -> table grad -> chunk_cat -> reduce_scatter. Blockers: FSDP2 owns the copy-in; the autograd embedding backward allocates its own gradient. Tier T2: Duplicate token ids accumulate; the deterministic sorted accumulation order must be preserved. Adjacent: R-03.

### R-25: SwiGLU forward fused with MXFP8 quantization of the w2 input

silu_and_mul writes h [4096,F] bf16 whose only consumer is w2's forward quantization (w2 saves its input in MXFP8, rowwise for fprop and colwise for wgrad). The SwiGLU kernel can write both MXFP8 layouts directly. Where: torchtitan_recipes/overrides/fused_swiglu.py:45-93; torchtitan/quantization/mxfp8/linear.py:77; 61 x 4.0 kernels, 27.02 us measured vs 10.24 us ideal per instance. Dataflow: w13 scaled_mm -> h13 -> silu_and_mul -> h -> mxfp8_quantize (rowwise+colwise, colwise saved) -> w2 scaled_mm. Blockers: Recipe Triton kernel + torchao quantizer; MXFP8LinearFunction must accept pre-quantized input. Tier T1: Pointwise + exact amax.

### R-26: q_norm backward fused with wq_a dy quantization

q_norm's RMSNorm backward (two kernels) writes d(q_a) [4096,1536] and wq_a's backward re-reads it to quantize it. One kernel can emit MXFP8 directly; the bf16 gradient has no other consumer. Where: torchtitan_recipes/overrides/fused_mla.py:1036-1043; torchtitan/quantization/mxfp8/linear.py:233; 61 x 4.0 kernels, 21.96 us measured vs 5.51 us ideal per instance. Dataflow: wq_b dgrad -> rmsnorm_backward (+reduce_rows) -> mxfp8_quantize -> wq_a GEMMs. Blockers: Launch-bound; two autograd nodes. Tier T1: RMSNorm-backward reduction order.

### R-27: kv_norm forward (without the contiguous clone) fused with wkv_b MXFP8 quantization

kv_latent is a strided slice of the wkv_a output [4096,576]; it is cloned to be contiguous, normalised, and wkv_b quantizes it rowwise and colwise (its input is saved in MXFP8). One kernel can read the slice in place and write only the MXFP8 layouts. Where: torchtitan_recipes/overrides/fused_mla.py:1064-1090; 61 x 5.0 kernels, 13.82 us measured vs 1.23 us ideal per instance. Dataflow: wkv_a scaled_mm -> split -> clone -> rmsnorm -> mxfp8_quantize (+2 swizzles) -> wkv_b scaled_mm. Blockers: Launch-bound small tensors; the norm backward needs its input saved (can be the strided view). Tier T1: RMSNorm row reduction order.

### R-28: tok_embeddings: embedding lookup directly from the gathered buffer (skip the full-table copy-out)

FSDP copies the gathered 1.85 GB embedding table out of the rank-major all-gather buffer before the lookup reads only 4096 rows. The lookup can index the gathered buffer directly. Where: torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:467-510; torchtitan/models/common/embedding.py:37-45; 1 x 3.0 kernels, 564.13 us measured vs 16.97 us ideal per instance. Dataflow: all_gather -> split_with_sizes_copy (full table) -> embedding (4096 rows). Blockers: FSDP2 materialises unsharded parameters before forward; the embedding module would need the rank-major layout. Tier T0: Pure gather. Adjacent: R-01.

### R-29: Dist-MoE backward staging: backward plan + activation-to-dispatch copy

The routed-expert backward recomputes a plan and copies the saved routed input [4096,7168] back into the symmetric dispatch buffer for the forward recompute. Folding the plan into the copy kernel (or keeping the forward dispatch buffer) removes a launch and part of the traffic. Where: dist_moe/_blockscaled.py:1972 (_backward_impl); dist_moe/kernels/triton/conditional_copy.py:522; 58 x 2.0 kernels, 24.98 us measured vs 16.97 us ideal per instance. Dataflow: saved routed input -> copy_activation_to_dispatch -> recompute fprop GEMMs. Blockers: Dist-MoE internals; the symmetric buffer is shared across layers, so retaining the forward copy needs per-layer buffers. Tier T0: Copy + integer planning. Adjacent: R-06.

### R-30: RoPE position bound check hoisted out of the per-layer attention (cross-layer)

Every attention layer re-checks that all 4096 positions are below the rope cache size with three kernels (le, all, _assert_async) on the same positions tensor. One check per step suffices. Where: torchtitan/models/common/rope.py:25-37; torchtitan_recipes/overrides/fused_mla.py:1045-1049; 61 x 3.0 kernels, 7.52 us measured vs 0.0 us ideal per instance. Dataflow: positions -> le -> all -> _assert_async, repeated 61 times on the same input. Blockers: Lives in each attention module's forward; hoisting needs the check at the model or data level. Tier T1: Boolean all-reduction (exact).

### R-31: q_norm forward fused with wq_b MXFP8 quantization

q_norm normalises the wq_a output [4096,1536] and wq_b re-reads the result to quantize it. One kernel can write the bf16 normed tensor (saved for wq_b's wgrad) and its MXFP8 layout. Where: torchtitan_recipes/overrides/fused_mla.py:1036-1043; 61 x 3.0 kernels, 11.87 us measured vs 4.57 us ideal per instance. Dataflow: wq_a scaled_mm -> rmsnorm -> mxfp8_quantize -> wq_b scaled_mm. Blockers: Launch-bound small tensors. Tier T1: RMSNorm row reduction order.

## 5. No opportunity

| what | ms | reason |
|---|---:|---|
| Dist-MoE peer-scatter output zero-fill (zero_ memset of [4096, 8, 7168] bf16 dispatch/combine buffers), 1 forward + 2 backward per MoE layer | 12.3 | Fake process-group artifact: dist_moe _buffers._initialize_fake_peer_scatter_output zeroes rows that no virtual peer writes; a native symmetric-memory run has one writer per row and launches no memset. |
| Flash-attention workspace memsets and zeros (3 per forward call, 3 per backward call) | 0.4 | Library-internal 0.7-1.7 us workspace/semaphore initialisation inside the FA4 call; would need a change in the attention kernel to fold, and the total is negligible. |
| Post-reduce-scatter gradient upcast bf16 -> fp32 of each FSDP group (reduce-scatter stream) | 2.32 | Its producer is the reduce-scatter (communication) and its only consumer is the optimizer step, which is out of scope; the cast belongs in the AdamW kernel (read bf16 shard grad directly), outside the forward/backward. |

Every non-GEMM, non-attention, non-communication kernel class in the forward and backward is assigned to exactly one region above or to this table (the assignment script reports any unassigned class; none remain). Excluded by definition: GEMMs (118.4 ms outside R-17/R-20), attention kernels (88.8 ms, including the FA4 backward pre/post-process), and emulated communication (178.7 ms). The optimizer step (AdamW 19.5 ms) and gradient clipping (5.0 ms) are out of scope.


<!-- Sections 6-7 below are the agent's own text from its final reply (the harness blocked it from writing
this .md file); sections 1-5 above were generated by its out/write_report.py. -->

## 6. Observations about working from these artifacts

**Easy.** Forward module from the `## mod <path> [<class>] ##` annotations enclosing each kernel launch
(correlation-id join). Shapes/dtypes from Kineto `Input Dims` and ET tensor tuples. FSDP and Dist-MoE annotate
their phases (`FSDP::all_gather_copy_out`, `FSDP::post_backward_reduce`, `moe_forward_recompute`), so glue is easy
to locate. The collective rule cleanly separates emulated communication.

**Hard.** (1) Backward module: `evaluate_function` events carry no sequence number; the nested node record does,
which joins to the forward op and its innermost module annotation (a first pass used the wrong event and left
196 ms of backward kernels unmapped). (2) FSDP pre-/post-backward runs inside whichever autograd node triggers
it; attributed to the FSDP unit by parsing its annotation. (3) Kernels launched from Python without an aten op
(torchao scale swizzle, FSDP 32x32 quantize, most Dist-MoE kernels) have no shapes or bytes. (4) Python stacks
are slow to resolve (~4M `python_function` events) and some stop at custom-op boundaries. (5) Size: 520 MB ET
and 4.7M Kineto events (cached as pickles). (6) ET byte accounting is approximate (nested lists, overlapping
views, in-place aliasing); region bytes use hand formulas. (7) Fake-PG distortions; host-bound capture;
low-entropy data (some kernels beat the roofline).

**Missing evidence.** No DRAM counters; saved bf16 vs MXFP8 inputs came from the recipe; no bytes for kernels
without an aten op; 20 ms of non-collective DtoD memcpys under FSDP needed Python stacks (MXFP8 operand refills,
`mxfp8/tensor.py:_build_operands`, Dist-MoE `_prepare_blockscaled_weight`).

**Phase and module in backward.** Phase by launch time against CPU markers (forward = root module up to the first
`evaluate_function`; backward up to the last `evaluate_function` / FSDP root post-backward callback; clipping and
AdamW follow). Module via node sequence number -> forward op -> module annotation; FSDP work -> FSDP unit; loss
ops -> "".

**False leads discarded.** DtoD memcpys under FSDP pre-forward (operand refills, now in R-01); the 70 us zero_ of
[4096,8,7168] (fake-PG artifact); four post-backward cats per block (one cat over 512 strided inputs); ET bytes for
that cat and function-level swizzle bytes (replaced with shape-based bytes); grad clipping first counted as
backward; FA4 at 3x/6x of its flop roofline (kernel quality, not fusion).

**Possibly missed.** Fusions in or around FA4 (backward preprocess in the wo dgrad epilogue; dQ postprocess writing
MXFP8); GEMM epilogues only in notes (residual add in wo, sigmoid in the router gate, dW accumulation with
beta=1, routed combine in the grouped-GEMM epilogue); glue that exists only with real EP; cross-step fusions;
megakernels spanning R-08..R-11; anything depending on overlap with communication.

**Effort.** About 40 minutes (17:04:37-17:44).

## 7. Reproduction

    cd /home/bahuang/local/step_analysis/dsv3_eager
    python -I out/load_kineto.py; python -I out/load_et.py
    python -I out/attribute_stage1.py
    python -I out/attribute_stage2.py
    python -I out/build_regions.py
    python -I out/write_outputs.py
    python -I out/write_report.py > out/report.md
