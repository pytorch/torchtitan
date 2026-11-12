# Kernel-fusion opportunities in one DeepSeek-V3 671B training step (GraphTrainer, fake PG, rank 0)

## 1. Setup

- **Program:** DeepSeek-V3 671B (61 layers: 3 dense-MLP layers, 58 MoE layers with 256 routed experts, top-8, 1 shared expert; MLA attention with q_lora 1536 / kv_lora 512 / 128 heads x (128 nope + 64 rope), vocab 129280, dim 7168). MXFP8 linears (torchao `mxfp8_quantize` + `triton_mx_block_rearrange` + `_scaled_mm_v2`), routed experts in Dist-MoE (`dist_moe.block_scaled_forward/backward`, MXFP8 grouped GEMMs, 4 local experts per rank at EP 64), the recipe's fused MLA RoPE kernels and fused SwiGLU.
- **Execution:** GraphTrainer traced joint forward+backward FX graph (66,238 nodes, 13,321 launch kernels), `memory_policy=none` (no recompute, parameters stay unsharded through backward), FSDP 256 with per-layer bucketed all-gather / reduce-scatter, EP 64, one 4096-token microbatch, deterministic algorithms, regional Inductor found no regions, so every kernel is an eager aten / custom-op kernel. One stream, one CPU thread.
- **Fake PG:** one process runs rank 0; collectives are emulated by local copies (excluded as communication). dist_moe under fake PG only computes rank-local routes and zero-fills its peer buffers (see no-opportunity).
- **Roofline:** NVIDIA GB300, HBM 6921.3 GB/s (copy), MXFP8 GEMM peak 3615.4 TFLOP/s, BF16 GEMM peak 1929.5 TFLOP/s (max of the measured shapes). Ideal = max(bytes / HBM, flops / peak).

## 2. Step budget

| category | GPU ms | % of step GPU-busy |
|---|---:|---:|
| gemm | 127.32 | 22.0 |
| attention | 89.02 | 15.4 |
| communication | 90.28 | 15.6 |
| memory_bound_glue | 233.76 | 40.5 |
| optimizer | 31.77 | 5.5 |
| other | 5.35 | 0.9 |
| **total GPU-busy (step)** | **577.48** | 100 |

- Forward+backward GPU-busy: **540.37 ms**; attributed to a launching op: **100.0%** (99.12% to graph nodes via their record-function labels, the remaining 0.88% = 848 `aten::clone` copies launched by the runtime's `param.grad` initialization right after the graph).
- gemm = `_scaled_mm_v2`, `mm`, and the CuTe grouped-GEMM kernels inside dist_moe; attention = the flash-attention forward/backward ops (incl. their preprocess/postprocess kernels); communication = kernels under `_c10d_functional::` / `c10d::` CPU ops; optimizer = AdamW (incl. 12.3 ms of the previous step's optimizer kernels that execute at the start of the window); other = post-backward grad-norm clipping.
- **GPU idle:** 932.0 s of the 932.6 s profiled step (forward+backward window idle 931.6 s). The step's wall time is dominated by the CPU: fake-PG emulation plus Python-stack profiling (3.9 M Python events) of a 66k-node graph, so idle time is a profiling artifact, not a GPU-side signal. All region numbers use kernel durations only.
- Memory-bound glue in forward+backward: 233.8 ms = 43% of forward+backward GPU time; the regions below cover all of it except the fake-PG memsets.

## 3. Ranked regions

33 regions; total gap 149.3 ms (25.9% of step GPU-busy; regions marked as overlapping describe alternative / chained fusions of neighbouring kernels, their kernel sets are disjoint so gaps add up). % of step = gap / step GPU-busy.

| id | title | phases | scope | kind | inst | measured ms | ideal ms | gap ms | % step | tier |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| R-01 | FSDP reduce-scatter copy-in: shard-interleave cat of w13 grads + bucket pack | bac | communication-glue, horizontal | multi-tensor copy / layout pack | 63 | 40.40 | 15.82 | 24.57 | 4.26 | T0 |
| R-02 | FSDP all-gather copy-out fused with MXFP8 weight quantization | for | communication-glue, cross-module, horizontal | copy-out + quantize (producer-consumer) | 63 | 35.20 | 19.29 | 15.91 | 2.76 | T1 |
| R-03 | MoE router forward: sigmoid + bias + group-limited top-k + normalize + expert counts | for | intra-module, cross-module | selection / top-k chain | 58 | 14.79 | 0.09 | 14.70 | 2.55 | T2 |
| R-04 | MoE router + aux-loss backward and gate-grad hi/mid/lo split | bac | intra-module, cross-module | pointwise + scatter chain | 58 | 11.99 | 0.13 | 11.86 | 2.05 | T2 |
| R-05 | ffn_norm backward block: grad sum + top-k reduce + RMSNorm backward + residual add + wo dY quantize | bac | cross-module | reduction + pointwise chain -> quantize | 61 | 14.08 | 7.04 | 7.04 | 1.22 | T1 |
| R-06 | Attention dQ postprocess + q RoPE backward + wq_b dY MXFP8 quantize | bac | cross-module | producer-consumer (cast -> rope -> quantize) | 61 | 12.37 | 5.39 | 6.99 | 1.21 | T1 |
| R-07 | Attention-backward preprocess (rowsum(dO*O), dQaccum zero) in wo dgrad GEMM epilogue | bac | gemm-adjacent, cross-module | GEMM epilogue | 61 | 22.88 | 16.23 | 6.64 | 1.15 | T3 |
| R-08 | MLA kv backward + wkv_b dY MXFP8 quantize | bac | cross-module | producer-consumer (pointwise/reduce -> quantize) | 61 | 11.91 | 5.41 | 6.50 | 1.13 | T1 |
| R-09 | Reduce-scatter output cast to fp32 + param.grad clone | bac | communication-glue, horizontal | multi-tensor cast / copy | 63 | 8.52 | 2.27 | 6.25 | 1.08 | T0 |
| R-10 | Colwise re-quantization of saved bf16 linear inputs in backward | bac | forward-backward, horizontal | quantize (fold into forward) | 62 | 6.30 | 1.47 | 4.83 | 0.84 | T1 |
| R-11 | Router aux-loss forward statistics | for | intra-module, horizontal | reduction chain | 58 | 4.07 | 0.04 | 4.02 | 0.70 | T1 |
| R-12 | q RoPE as wq_b GEMM epilogue | for | gemm-adjacent, cross-module | GEMM epilogue | 61 | 9.20 | 5.22 | 3.98 | 0.69 | T3 |
| R-13 | Fused cross-entropy forward + backward with lm_head dY quantize | for+bac | forward-backward, cross-module | online-softmax reduction + pointwise | 1 | 4.19 | 0.31 | 3.88 | 0.67 | T1 |
| R-14 | attention_norm backward + residual-grad add (layer boundary) | bac | cross-module, cross-layer | reduction + pointwise chain | 62 | 5.88 | 2.61 | 3.27 | 0.57 | T1 |
| R-15 | MoE top-k combine + shared-expert add + residual add | for | cross-module | reduction + pointwise chain | 58 | 8.52 | 5.42 | 3.10 | 0.54 | T1 |
| R-16 | FSDP all-gather copy-in with fp32->bf16 cast and pad | for | communication-glue, horizontal | multi-tensor cast / pack | 62 | 4.89 | 2.27 | 2.62 | 0.45 | T0 |
| R-17 | Layer-output gradient: routed-expert broadcast-and-scale + shared w2 dY quantize | bac | horizontal, cross-module | sibling consumers of one input | 61 | 11.50 | 8.93 | 2.58 | 0.45 | T1 |
| R-18 | dist_moe routing metadata and plan kernels | for+bac | forward-backward, communication-glue | histogram / prefix-sum metadata | 58 | 2.39 | 0.01 | 2.38 | 0.41 | T2 |
| R-19 | k RoPE / k-v split as wkv_b GEMM epilogue | for | gemm-adjacent, cross-module | GEMM epilogue | 61 | 6.71 | 4.33 | 2.38 | 0.41 | T3 |
| R-20 | Router gate backward GEMM operand packing and fp32 weight-grad accumulation | bac | gemm-adjacent, intra-module | GEMM prologue/epilogue | 58 | 3.37 | 1.36 | 2.01 | 0.35 | T3 |
| R-21 | attention_norm + duplicate MXFP8 quantize for wq_a / wkv_a | for | cross-module, horizontal, cross-layer | producer-consumer (norm -> quantize) + dedupe | 61 | 3.77 | 1.89 | 1.88 | 0.33 | T1 |
| R-22 | wq_a / wkv_a sibling GEMMs on the same normed input | for+bac | horizontal, gemm-adjacent | horizontal GEMM concatenation | 61 | 5.89 | 4.18 | 1.71 | 0.30 | T3 |
| R-23 | Attention residual add + ffn_norm + w13 MXFP8 quantize | for | cross-module | pointwise + reduction -> quantize | 61 | 3.98 | 2.34 | 1.64 | 0.28 | T1 |
| R-24 | SwiGLU backward + w13 dY MXFP8 quantize | bac | cross-module | producer-consumer (pointwise -> quantize) | 61 | 2.59 | 1.04 | 1.54 | 0.27 | T1 |
| R-25 | kv_norm backward + k_pe grad cat + wkv_a dY quantize | bac | cross-module | reduction + concat -> quantize | 61 | 1.51 | 0.12 | 1.39 | 0.24 | T1 |
| R-26 | dist_moe dispatch-buffer copies and activation save/restore | for+bac | forward-backward, communication-glue, horizontal | copy elimination | 58 | 3.32 | 1.97 | 1.35 | 0.23 | T0 |
| R-27 | q_norm backward + wq_a dY quantize | bac | cross-module | reduction -> quantize | 61 | 1.48 | 0.34 | 1.14 | 0.20 | T1 |
| R-28 | SwiGLU + w2 MXFP8 quantize | for | cross-module | producer-consumer (pointwise -> quantize) | 61 | 1.66 | 0.62 | 1.04 | 0.18 | T1 |
| R-29 | kv_norm (strided slice) + wkv_b MXFP8 quantize | for | cross-module | reduction -> quantize | 61 | 0.85 | 0.07 | 0.77 | 0.13 | T1 |
| R-30 | Attention output MXFP8 quantize + scale swizzle | for | intra-module | quantize + layout | 61 | 2.38 | 1.79 | 0.59 | 0.10 | T1 |
| R-31 | q_norm + wq_b MXFP8 quantize | for | cross-module | reduction -> quantize | 61 | 0.73 | 0.28 | 0.45 | 0.08 | T1 |
| R-32 | Embedding dense backward | bac | intra-module | index accumulate | 1 | 0.56 | 0.28 | 0.28 | 0.05 | T2 |
| R-33 | Final norm + lm_head input MXFP8 quantize | for | cross-module | reduction -> quantize | 1 | 0.04 | 0.02 | 0.02 | 0.00 | T1 |

Regions per scope tag: cross-module 22, horizontal 10, communication-glue 6, intra-module 6, gemm-adjacent 5, forward-backward 4, cross-layer 2.

Evidence: every member kernel time comes from the trace (all regions `measured`). Ideal bytes come from graph tensor metadata, except for kernels inside dist_moe / flash-attention ops, whose bytes are estimated from the op's tensor shapes and the library source (route-wise [4096,8,7168] buffers, fp32 dQ accumulator).

## 4. Region details

### R-01 FSDP reduce-scatter copy-in: shard-interleave cat of w13 grads + bucket pack  (T0, gap 24.57 ms)

- **What:** Before each layer's gradient reduce-scatter, every weight gradient is chunked along its shard axis and copied into one flat bf16 bucket; the dim-1-sharded w13 gradients ([2, 2048|18432, 7168]) are first re-laid out by a 256-input cat into [512, 8|72, 7168] and then packed again.
- **Where:** torchtitan/experiments/graph_trainer/simple_fsdp.py:366 (Shard redistribute backward, w13 interleave cat); bucketing._pre_bucket_reduce_scatter from joint_transformer_block_bucketing_reordering_pass; 63 instances x 7.7 kernels; 641.2 us measured vs 251.1 us ideal per instance (1738.2 MB external). Shapes: grads bf16 (e.g. wq_b [24576,1536], wo [7168,16384], experts [4,2,2048,7168]+[4,7168,2048]); w13 grad [2,2048,7168]->[512,8,7168] (dense [2,18432,7168]->[512,72,7168]); bucket bf16 [234M] (attn+shared) / [176M] (experts) / [585M] (dense).
- **Dataflow:** Each wgrad GEMM output and norm-weight grad is read once by chunk_cat and written once into the reduce-scatter input bucket; the w13 grad is read and rewritten twice (cat, then chunk_cat). A single multi-tensor pack kernel that applies the dim-1 shard interleave while packing reads each grad once and writes the bucket once.
- **Blockers:** Bucketing pass (bucketing._pre_bucket_reduce_scatter) and DTensor redistribute own the layout; the dim-1 interleave comes from Shard(1) placement of w13. Further step (T3): let wgrad GEMMs write straight into bucket offsets.
- **Tier:** T0 -- Pure copies into fixed offsets; any order of writes gives identical bytes.
- **Notes:** The 256-input CatArrayBatchedCopy is the slowest piece: 184 us (shared w13) and 922 us (dense w13) per layer vs 17/153 us of bytes. [working id C3]

### R-02 FSDP all-gather copy-out fused with MXFP8 weight quantization  (T1, gap 15.91 ms)

- **What:** After each layer's parameter all-gather, the gathered bf16 bucket is split into per-parameter bf16 weights (plus a cat for dim-1-sharded w13), and then every linear weight is re-read to produce MXFP8 data and 32x32 block scales (expert weights through a separate quantizer).
- **Where:** simple_fsdp.py:268-366; 63 instances x 14.4 kernels; 558.8 us measured vs 306.2 us ideal per instance (2119.6 MB external). Shapes: gathered bucket bf16 [256, 915520] / [4, 44040192] (experts) / [256, 2284608] (dense); weights e.g. wo [7168,16384], wkv_b [32768,512], experts [4,4096,7168] -> fp8 e4m3 + e8m0 scales.
- **Dataflow:** split_with_sizes_copy writes every unsharded bf16 weight, which is consumed only by the weight quantizer (except norm weights and the bf16 router gate). Quantizing straight from the gathered buffer removes one bf16 write+read of every weight.
- **Blockers:** Copy-out belongs to the FSDP bucketing pass; weight quantization belongs to the MXFP8 linear (torchao) and dist_moe.prepare_mxfp8_weight; the gathered layout is shard-major (256 x shard) so the quantizer must read 32x32 tiles across shard boundaries.
- **Tier:** T1 -- Block amax is a max-reduction (order-insensitive, exact), so a fused kernel can be bitwise identical; layout of the swizzled scales must match.
- **Overlaps / chains with:** R-16
- **Notes:** Instances: 61 layers + embedding/norm/lm_head buckets. [working id C2]

### R-03 MoE router forward: sigmoid + bias + group-limited top-k + normalize + expert counts  (T2, gap 14.70 ms)

- **What:** Per MoE layer, the router turns gate logits [4096,256] fp32 into 8 expert ids and weights per token via sigmoid, bias add, group top-2 sums, top-4 groups, masked top-8, gather, normalization and route scaling, then counts tokens per expert; about 28 small kernels including three top-k calls.
- **Where:** moe.py:257-314; routed_experts.py:195-196; 58 instances x 28.0 kernels; 254.9 us measured vs 1.5 us ideal per instance (10.2 MB external). Shapes: logits fp32 [4096,256] -> view [4096,8,32]; topk [4096,8,32]k2, [4096,8]k4, [4096,256]k8; ids int64 [4096,8], weights fp32 [4096,8], counts [256].
- **Dataflow:** All ops consume the same [4096,256] score row of a token; a one-CTA-per-token-block kernel can do sigmoid, bias, group selection, final top-8, gather, normalization and the masked_fill of padded rows in registers and write only ids, weights, the selection mask and counts.
- **Blockers:** Must reproduce torch.topk ordering bitwise; the per-expert count update (add_ into a buffer) needs an atomic-free reduction for determinism.
- **Tier:** T2 -- Top-k tie-breaking and index order must match torch.topk; the per-expert count is an integer reduction (exact).
- **Overlaps / chains with:** R-11, R-18
- **Notes:** The gate GEMM (bf16 x bf16 -> fp32, 17 us vs 9.6 us) could additionally take sigmoid + bias as an epilogue (T3). [working id F10]

### R-04 MoE router + aux-loss backward and gate-grad hi/mid/lo split  (T2, gap 11.86 ms)

- **What:** Per MoE layer, the backward of the routing weights (normalize, gather), of the load-balance aux loss and of the sigmoid produces the gate-logit gradient [4096,256] fp32, which is then split into hi/mid/lo bf16 parts and concatenated for the 3-way bf16 router-gate GEMMs; about 70 small kernels including a 31-kernel deterministic scatter_add.
- **Where:** hi_mid_lo_linear.py:117; local_compile.py:96; moe.py:257-607; routed_experts.py:196; 58 instances x 70.0 kernels; 206.8 us measured vs 2.2 us ideal per instance (15.2 MB external). Shapes: grad weights fp32 [4096,8]; scores fp32 [4096,256]; scatter_add [4096,8]->[4096,256]; hi/mid/lo bf16 [4096,256] x3 -> cat [4096,768].
- **Dataflow:** Everything is per-token over a 256-wide row: grad of the 8 selected weights -> scatter into 256 columns -> add aux-loss grad -> sigmoid_backward -> split into three bf16 terms written directly into the [4096,768] GEMM operand.
- **Blockers:** Deterministic scatter_add launches a sort-based path (31 kernels); aux-loss pieces come from separate functions; the hi/mid/lo split uses int16 bit tricks saved from forward.
- **Tier:** T2 -- scatter_add order (deterministic mode) must be reproduced; with only 8 distinct columns per row and no duplicate indices per row, a per-row scatter is collision-free and therefore order-independent.
- **Overlaps / chains with:** R-20
- **Notes:** [working id R7]

### R-05 ffn_norm backward block: grad sum + top-k reduce + RMSNorm backward + residual add + wo dY quantize  (T1, gap 7.04 ms)

- **What:** Per layer, the gradient of the ffn_norm output is the sum of the router-gate, routed-expert (top-k reduced over 8 routes) and shared-expert input gradients; it feeds RMSNorm backward, whose input gradient is added to the residual gradient and then quantized to MXFP8 for the wo backward GEMMs.
- **Where:** _blockscaled.py:4377; linear.py:129; nn_modules.py:155; 61 instances x 8.9 kernels; 230.8 us measured vs 115.3 us ideal per instance (798.3 MB external). Shapes: route grads bf16 [4096,8,7168] -> [4096,7168]; 3x bf16 [4096,7168] grads; rstd fp32 [4096,1]; norm weight [7168]; fp8 [4096,7168] x2 + scales.
- **Dataflow:** reduce_from_topk -> add -> add -> rms_norm_backward (+ weight-grad column reduction kernel) -> add residual grad -> mxfp8_quantize -> scale swizzle; every intermediate [4096,7168] tensor has exactly one consumer inside the chain.
- **Blockers:** reduce_from_topk lives inside dist_moe.block_scaled_backward; the RMSNorm weight-grad needs a cross-token reduction (second kernel today).
- **Tier:** T1 -- Sum over K routes and the RMSNorm row/column reductions must keep their order; the separate bf16 roundings of each add must be reproduced.
- **Overlaps / chains with:** R-17
- **Notes:** Dense layers (0-2) have only rms bwd + residual add + quantize. [working id B3]

### R-06 Attention dQ postprocess + q RoPE backward + wq_b dY MXFP8 quantize  (T1, gap 6.99 ms)

- **What:** The attention backward accumulates dQ in fp32 [4096,128,192] and a postprocess kernel converts it to bf16; RoPE backward is then applied in place on the rope part, and the result is re-read to quantize it to MXFP8 for the wq_b backward GEMMs.
- **Where:** attention.py:457; fused_mla.py:847; linear.py:129; 61 instances x 5.0 kernels; 202.8 us measured vs 88.3 us ideal per instance (611.4 MB external). Shapes: dQaccum fp32 [4096,128,192] -> dq bf16 [4096,128,192] -> fp8 [4096,24576] x2 + e8m0 [4096,768].
- **Dataflow:** Postprocess writes dq bf16 (201 MB), q_rope_ reads/writes it, quantize reads it again; one kernel reading the fp32 accumulator and writing fp8 rowwise+colwise removes three bf16 passes.
- **Blockers:** The postprocess kernel is internal to the flash-attention backward (torch_attn._varlen_attn_backward); needs a hook to replace its epilogue or to expose dQaccum.
- **Tier:** T1 -- Pointwise cast and rotation plus exact amax; the dQ accumulation itself is untouched.
- **Overlaps / chains with:** R-07
- **Notes:** [working id B5]

### R-07 Attention-backward preprocess (rowsum(dO*O), dQaccum zero) in wo dgrad GEMM epilogue  (T3, gap 6.64 ms)

- **What:** The flash-attention backward preprocess re-reads the attention output O and its gradient dO [4096,128,128] to compute the per-row dot product and zero the fp32 dQ accumulator; dO is produced by the wo dgrad GEMM immediately before.
- **Where:** attention.py:457; linear.py:129; 61 instances x 2.0 kernels; 375.0 us measured vs 266.1 us ideal per instance (826.7 MB external). Shapes: wo dgrad: fp8 [4096,7168] x [7168,16384] -> dO bf16 [4096,16384]; O bf16 [4096,128,128]; dpsum fp32 [128,4096]; dQaccum fp32 [4096,128,192].
- **Dataflow:** Each GEMM output tile holds dO for complete head rows (128 columns = one head), so the epilogue can load the matching O tile and emit rowsum(dO*O) per (token, head) while storing dO.
- **Blockers:** cuBLAS (nvjet) GEMM has no custom epilogue; flash-attention owns the preprocess and its layout of dpsum/LSE; zeroing dQaccum is still needed unless the main kernel initializes it.
- **Tier:** T3 -- Needs the GEMM; the row dot product over head_dim=128 must use the preprocess's fp32 summation order.
- **Overlaps / chains with:** R-06, R-05
- **Notes:** Ideal is GEMM-compute bound (266 us), so the preprocess bytes hide under the GEMM in the fused form; measured GEMM is slightly faster than the random-data peak (low-entropy fake-PG data). [working id G1]

### R-08 MLA kv backward + wkv_b dY MXFP8 quantize  (T1, gap 6.50 ms)

- **What:** The attention backward's dK [4096,128,192] and dV [4096,128,128] are re-packed into the wkv_b output gradient [4096,128,256] plus the summed k_pe gradient [4096,64], and that gradient is then re-read to quantize it to MXFP8 (both orientations) with a separate scale-swizzle kernel.
- **Where:** fused_mla.py:903; linear.py:129; 61 instances x 4.0 kernels; 195.3 us measured vs 88.7 us ideal per instance (614.0 MB external). Shapes: dK bf16 [4096,128,192], dV bf16 [4096,128,128] -> dkv bf16 [4096,32768] -> fp8 [4096,32768] x2 + e8m0 [4096,1024].
- **Dataflow:** fused_mla_kv_backward writes the bf16 [4096,32768] tensor that only the quantizer reads; quantizing in the same kernel removes a 268 MB write+read per layer.
- **Blockers:** Two custom kernels from different owners (fused MLA override, torchao mxfp8_quantize); the quantizer needs 32-row blocks for the colwise orientation, so the fused kernel must tile 32 tokens x 32 features.
- **Tier:** T1 -- k_pe grad is a sum over 128 heads (order must match fused_mla_kv_backward); quantization amax is exact.
- **Notes:** [working id B6]

### R-09 Reduce-scatter output cast to fp32 + param.grad clone  (T0, gap 6.25 ms)

- **What:** After each layer's reduce-scatter, every gradient shard is cast from bf16 to fp32 by its own kernel (848 casts), and after the step's backward the runtime clones each fp32 gradient again to initialize param.grad (848 copies).
- **Where:** torchtitan/experiments/graph_trainer/simple_fsdp.py:366 (gradient cast to fp32 param dtype); graph_pp/runner.py:531 + common_utils.py:303-326 (param.grad clone); 63 instances x 26.9 kernels; 135.3 us measured vs 36.1 us ideal per instance (249.7 MB external). Shapes: per-param shards bf16 -> fp32, e.g. [6,7168], [96,1536], [28,16384], [1,2,2048,7168]; 848 tensors.
- **Dataflow:** Each reduce-scatter output shard is read once and written as fp32; the clone then reads and writes it again. A single multi-tensor kernel per bucket can cast directly into the param.grad storage.
- **Blockers:** The casts are graph outputs (one per parameter); the clone is done outside the graph by the gradient-accumulation helper (clone_grads_to_initialize_param_grad).
- **Tier:** T0 -- Casts and copies only.
- **Notes:** Per-layer split of the post-graph clones is not attributed (they run after the graph); their 4.76 ms is added to the region total. [working id C4]

### R-10 Colwise re-quantization of saved bf16 linear inputs in backward  (T1, gap 4.83 ms)

- **What:** For the linears whose inputs are saved in bf16 (wq_a, wkv_a, wq_b, wo, w13, lm_head), the backward re-reads the saved activation to quantize it along tokens for the weight-gradient GEMM; the attention_norm output is quantized twice (for wq_a and wkv_a).
- **Where:** linear.py:129; 62 instances x 9.9 kernels; 101.6 us measured vs 23.7 us ideal per instance (164.3 MB external). Shapes: bf16 [4096,7168] x2 (same tensor), [4096,1536], [4096,16384], [4096,7168] -> fp8 colwise + e8m0 [K,128].
- **Dataflow:** The forward quantize kernels (F2/F3/F4/F6) already read these tensors; emitting the colwise fp8 copy there costs only its write and removes the backward read (and the duplicate wq_a/wkv_a pass).
- **Blockers:** Changes what is saved for backward (fp8 colwise instead of bf16) -- the bf16 copy may still be needed by other consumers (router gate, aux loss); torchao's saved-format choice (306 bf16 vs 122 mxfp8 linears).
- **Tier:** T1 -- Exact amax; the forward quantizer can emit the colwise copy from the same 32x32 tile.
- **Overlaps / chains with:** R-21, R-23, R-31, R-30
- **Notes:** Ideal counts only the colwise fp8 write + scales (the marginal cost when folded into the forward quantize). [working id B10]

### R-11 Router aux-loss forward statistics  (T1, gap 4.02 ms)

- **What:** Per MoE layer, the load-balance auxiliary loss converts the selection mask to fp32, sums it over tokens, L2-normalizes the scores, sums them over tokens and combines them into a scalar loss term; about 23 kernels, most on [4096,256] or scalars.
- **Where:** local_compile.py:96; moe.py:598-607; 58 instances x 23.0 kernels; 70.1 us measured vs 0.8 us ideal per instance (5.3 MB external). Shapes: mask bool [4096,256] -> fp32; scores fp32 [4096,256]; column sums [256]; scalars.
- **Dataflow:** Reads the same sigmoid scores and selection mask that the router (F10) already holds; can be computed as a by-product of the routing kernel with a second-stage column reduction.
- **Blockers:** Separate function boundary from the router; cross-token column reductions need a two-pass or ordered reduction for determinism.
- **Tier:** T1 -- Column sums over 4096 tokens and norms must keep the reduction order (2-kernel split reductions today).
- **Overlaps / chains with:** R-03
- **Notes:** [working id F11]

### R-12 q RoPE as wq_b GEMM epilogue  (T3, gap 3.98 ms)

- **What:** The wq_b GEMM writes q [4096,128,192] in bf16 and a separate kernel reads it back to apply RoPE to the last 64 dims of every head and rewrite the whole tensor.
- **Where:** fused_mla.py:847; linear.py:174; 61 instances x 2.0 kernels; 150.8 us measured vs 85.5 us ideal per instance (247.8 MB external). Shapes: fp8 [4096,1536] x [1536,24576] -> bf16 [4096,24576] = [4096,128,192]; freqs fp32 [4096,32,2].
- **Dataflow:** RoPE pairs (2 adjacent features) lie inside one output tile row, so the epilogue can rotate before the store.
- **Blockers:** cuBLAS nvjet kernels have no custom epilogue; needs a CUTLASS/CuTe MXFP8 GEMM with an epilogue visitor.
- **Tier:** T3 -- Needs the GEMM; the rotation is pointwise in fp32 and can match bitwise.
- **Notes:** fused_mla_q_rope alone runs at roofline (58 us); the gap is the extra bf16 round trip. [working id F8]

### R-13 Fused cross-entropy forward + backward with lm_head dY quantize  (T1, gap 3.88 ms)

- **What:** The lm_head logits [4096,129280] bf16 are upcast to fp32, log-softmaxed and gathered for the loss, then in backward the fp32 softmax gradient is rebuilt, cast back to bf16 and quantized to MXFP8 for the lm_head backward GEMMs.
- **Where:** linear.py:129; local_compile.py:96; 1 instances x 13.0 kernels; 4192.6 us measured vs 310.8 us ideal per instance (2151.3 MB external). Shapes: logits bf16 [4096,129280] -> fp32 (2.1 GB) -> log_softmax fp32; grad fp32 -> bf16 -> fp8 [4096,129280] + scales.
- **Dataflow:** Forward needs per-row max/LSE and the target logit; backward needs softmax - onehot. Reading the bf16 logits once can produce the loss, and the fp8 gradient in the same pass.
- **Blockers:** Loss function boundary and the autograd split between forward and backward; the grad scale (1/num_tokens) must be known at forward time.
- **Tier:** T1 -- Max and sum-exp reductions over the vocabulary must follow the same order; one fused kernel can do fwd and bwd because the loss gradient is a known constant scale.
- **Notes:** Single instance; 4.2 ms measured vs 0.31 ms of bytes. [working id L1]

### R-14 attention_norm backward + residual-grad add (layer boundary)  (T1, gap 3.27 ms)

- **What:** Per layer, the wq_a and wkv_a input gradients are added, fed through attention_norm's RMSNorm backward (two kernels), and the result is added to the residual gradient to form the previous layer's output gradient.
- **Where:** nn_modules.py:155; 62 instances x 4.0 kernels; 94.9 us measured vs 42.1 us ideal per instance (291.8 MB external). Shapes: bf16 [4096,7168] x4 inputs (two dX, residual grad, saved input), rstd fp32 [4096,1], weight [7168].
- **Dataflow:** add(dX_wq_a, dX_wkv_a) -> rms_norm_backward -> add(residual grad) -> previous layer's output gradient; each intermediate has one consumer.
- **Blockers:** The weight-grad column reduction across 4096 tokens needs a second pass or ordered cross-CTA reduction.
- **Tier:** T1 -- RMSNorm row reduction and weight-grad column reduction order; separate bf16 rounding of each add.
- **Overlaps / chains with:** R-17, R-22
- **Notes:** Instance 61 is the final norm (no residual add); instance 0's add feeds the embedding backward. Concatenating wq_a/wkv_a dgrad (G2) would remove the first add. [working id B1]

### R-15 MoE top-k combine + shared-expert add + residual add  (T1, gap 3.10 ms)

- **What:** Per MoE layer, the routed-expert outputs [4096,8,7168] are weighted and summed over the 8 routes, then added to the shared-expert output and to the residual stream as three separate passes over [4096,7168].
- **Where:** _blockscaled.py:4377; model.py:64; moe.py:821; 58 instances x 3.0 kernels; 146.8 us measured vs 93.4 us ideal per instance (646.3 MB external). Shapes: route-wise bf16 [4096,8,7168] (470 MB), weights fp32 [4096,8], shared out bf16 [4096,7168], residual bf16 [4096,7168] -> layer output bf16 [4096,7168].
- **Dataflow:** scale_and_sum writes y [4096,7168] that only add_44 reads; add_44's output is read only by add_45; the layer output then feeds the next attention_norm (F2) and the residual.
- **Blockers:** The combine runs inside dist_moe.block_scaled_forward after an EP barrier; adding the shared output requires passing it into the library or exposing the combine buffer.
- **Tier:** T1 -- fp32 accumulation over K=8 in the library's order, then two bf16-rounded adds that must be reproduced separately.
- **Overlaps / chains with:** R-21
- **Notes:** Chaining with the next layer's attention_norm + quantize (F2) gives a cross-layer region. [working id F1]

### R-16 FSDP all-gather copy-in with fp32->bf16 cast and pad  (T0, gap 2.62 ms)

- **What:** Before each layer's parameter all-gather, the fp32 parameter shards are cast to bf16 and packed into the all-gather input buffer by a multi-tensor kernel, while unevenly sharded parameters (e.g. wkv_a's [3,7168] shard) take a separate cast + pad pair.
- **Where:** simple_fsdp.py:366; 62 instances x 4.9 kernels; 78.9 us measured vs 36.6 us ideal per instance (253.7 MB external). Shapes: fp32 shards e.g. [6,7168], [96,1536], [1,2,2048,7168] -> bf16 bucket shard [915520] / [44040192] / [2284608].
- **Dataflow:** The odd-shaped shard is cast and padded into a temporary and then copied again into the bucket; folding it into the multi-tensor copy-in removes two kernels per layer; small buckets are launch-bound.
- **Blockers:** Padding is decided by DTensor/simple_fsdp for uneven shards; the bucketing pass owns the copy-in.
- **Tier:** T0 -- Casts and copies only.
- **Overlaps / chains with:** R-02
- **Notes:** [working id C1]

### R-17 Layer-output gradient: routed-expert broadcast-and-scale + shared w2 dY quantize  (T1, gap 2.58 ms)

- **What:** Per layer, the layer-output gradient [4096,7168] is read by two consumers: dist_moe broadcasts it to the 8 routes scaled by the routing weights (also computing the routing-weight gradient against the saved route outputs), and the shared-expert w2 backward quantizes it to MXFP8.
- **Where:** _blockscaled.py:4377; linear.py:129; 61 instances x 4.0 kernels; 188.6 us measured vs 146.3 us ideal per instance (1012.8 MB external). Shapes: dy bf16 [4096,7168]; saved route outputs bf16 [4096,8,7168]; grad routes bf16 [4096,8,7168]; fp8 [4096,7168] x2 + scales.
- **Dataflow:** Both consumers read the same dy rows; one kernel can load dy once, quantize it and emit the scaled route gradients.
- **Blockers:** One consumer is inside dist_moe.block_scaled_backward (after recompute and barriers), the other is the torchao quantizer.
- **Tier:** T1 -- Routing-weight gradient is a dot product over D=7168 (order must match); amax exact.
- **Overlaps / chains with:** R-14, R-05
- **Notes:** The [4096,8,7168] read+write dominates the bytes; dense layers only have the quantize. [working id B2]

### R-18 dist_moe routing metadata and plan kernels  (T2, gap 2.38 ms)

- **What:** Inside each routed-expert forward and backward, small kernels split the 8 routes per token by destination rank, build prefix sums and per-expert counts, gather metadata and build the forward/backward GEMM plans.
- **Where:** _blockscaled.py:4377; 58 instances x 8.0 kernels; 41.2 us measured vs 0.1 us ideal per instance (0.7 MB external). Shapes: expert ids int64 [4096,8], weights fp32 [4096,8], per-rank/expert counts and offsets (KB-sized).
- **Dataflow:** All kernels consume only the [4096,8] routing ids/weights produced by the router (F10); the backward plan recomputes information known from the forward plan.
- **Blockers:** Library-internal (dist_moe); plan kernels are split by EP barrier points; backward plan could be cached from forward.
- **Tier:** T2 -- Prefix sums and slot assignment define the order of routed rows, which downstream data placement depends on.
- **Overlaps / chains with:** R-03
- **Notes:** Launch-bound: ~41 us per layer for <1 MB of data. [working id F13]

### R-19 k RoPE / k-v split as wkv_b GEMM epilogue  (T3, gap 2.38 ms)

- **What:** The wkv_b GEMM writes [4096,128,256] bf16 (k_nope|v per head), and a separate kernel reads it back with k_pe [4096,64] to build k [4096,128,192] (with RoPE on the shared k_pe broadcast to all heads) and v.
- **Where:** fused_mla.py:903; linear.py:174; 61 instances x 2.0 kernels; 110.0 us measured vs 70.9 us ideal per instance (490.8 MB external). Shapes: fp8 [4096,512] x [512,32768] -> bf16 [4096,32768]; k_pe bf16 [4096,64]; outputs k bf16 [4096,128,192], v [4096,128,128].
- **Dataflow:** Each output tile row is one token's (k_nope, v) for a head; the epilogue can write k_nope into k, append rotated k_pe and write v directly.
- **Blockers:** cuBLAS nvjet GEMM without custom epilogue; output split into two tensors with different strides.
- **Tier:** T3 -- Needs the GEMM; RoPE and copies are pointwise.
- **Notes:** [working id F9]

### R-20 Router gate backward GEMM operand packing and fp32 weight-grad accumulation  (T3, gap 2.01 ms)

- **What:** The 3-term (hi/mid/lo) bf16 router-gate backward replicates the [256,7168] gate weight three times for the input-gradient GEMM and runs three weight-gradient GEMMs whose fp32 outputs are accumulated by two adds and cast to bf16.
- **Where:** hi_mid_lo_linear.py:117; 58 instances x 7.0 kernels; 58.0 us measured vs 23.4 us ideal per instance (83.4 MB external). Shapes: W bf16 [256,7168] x3 -> [768,7168]; dW: 3x (bf16 [256,4096] x [4096,7168] -> fp32 [256,7168]), adds, fp32->bf16.
- **Dataflow:** cat_313 output only feeds one GEMM; mm outputs only feed the add_ chain; the final cast only feeds the gradient output.
- **Blockers:** aten.mm.dtype has no accumulate-into epilogue; replicated-weight cat is an artifact of expressing the 3-way split as one GEMM.
- **Tier:** T3 -- Accumulating the three products in the GEMM epilogue (beta=1) keeps the same fp32 add order; a single K-concatenated GEMM would not be bitwise identical.
- **Overlaps / chains with:** R-04
- **Notes:** [working id R8]

### R-21 attention_norm + duplicate MXFP8 quantize for wq_a / wkv_a  (T1, gap 1.88 ms)

- **What:** Per layer, attention_norm normalizes the residual stream [4096,7168] and the normalized tensor is then quantized to MXFP8 twice (once for wq_a and once for wkv_a, identical results), each followed by a scale-swizzle kernel; dense layers also add the previous layer's MLP residual first, and layer 0 starts from the embedding gather.
- **Where:** embedding.py:41; linear.py:162-168; model.py:64; nn_modules.py:155; 61 instances x 5.1 kernels; 61.8 us measured vs 30.9 us ideal per instance (214.2 MB external). Shapes: bf16 [4096,7168] -> rstd fp32 [4096,1], normed bf16 [4096,7168], fp8 [4096,7168] + e8m0 [4096,224].
- **Dataflow:** norm output -> two identical rowwise quantizations -> two swizzles; one kernel can normalize, quantize once, and write swizzled scales (plus the colwise copy, see B10).
- **Blockers:** Norm (aten _fused_rms_norm) and quantize (torchao) are separate custom ops inside different modules; the two linears do not know they share an input.
- **Tier:** T1 -- RMSNorm row reduction order; exact amax.
- **Overlaps / chains with:** R-15, R-10, R-22
- **Notes:** [working id F2]

### R-22 wq_a / wkv_a sibling GEMMs on the same normed input  (T3, gap 1.71 ms)

- **What:** wq_a [7168->1536] and wkv_a [7168->576] both consume the same quantized attention_norm output, and in backward both produce input gradients that are summed; the N=576 GEMM runs at half its roofline.
- **Where:** linear.py:129-174; 61 instances x 4.0 kernels; 96.6 us measured vs 68.6 us ideal per instance (220.3 MB external). Shapes: fp8 [4096,7168] x [7168,1536] and x [7168,576] (fwd); [4096,1536] x [1536,7168] + [4096,576] x [576,7168] (dgrad).
- **Dataflow:** Forward: one GEMM with N=2112 reads the activation once. Backward: [dq_a | dkv_a] x [Wq_a ; Wkv_a] gives dX directly and removes the add in B1.
- **Blockers:** Two separate MXFP8Linear modules with separate FSDP parameters; weight concatenation needs a fused parameter or a grouped GEMM.
- **Tier:** T3 -- Concatenating N in forward is bitwise identical per output column; concatenating K in dgrad changes the fp32 accumulation (sum of two GEMMs + bf16 add today).
- **Overlaps / chains with:** R-21, R-14
- **Notes:** [working id G2]

### R-23 Attention residual add + ffn_norm + w13 MXFP8 quantize  (T1, gap 1.64 ms)

- **What:** Per layer, the wo output is added to the residual stream, normalized by ffn_norm, and the normalized tensor is quantized to MXFP8 for the shared-expert (or dense) w13 GEMM with a separate scale swizzle.
- **Where:** linear.py:162-168; model.py:53; nn_modules.py:155; 61 instances x 4.0 kernels; 65.2 us measured vs 38.3 us ideal per instance (265.2 MB external). Shapes: bf16 [4096,7168] x2 -> h bf16 [4096,7168], normed bf16, rstd fp32, fp8 [4096,7168] + e8m0 [4096,224].
- **Dataflow:** add -> rms_norm -> quantize -> swizzle; h and the normed tensor are still needed (residual, router, routed experts, backward).
- **Blockers:** Three different op owners (aten add, aten fused rms norm, torchao quantize); the add could alternatively become the wo GEMM epilogue (T3).
- **Tier:** T1 -- RMSNorm reduction order; the add's bf16 rounding must precede the norm.
- **Overlaps / chains with:** R-10
- **Notes:** [working id F3]

### R-24 SwiGLU backward + w13 dY MXFP8 quantize  (T1, gap 1.54 ms)

- **What:** Per layer, the SwiGLU backward produces the w13 output gradient [4096,2,2048] (dense: [4096,2,18432]) in bf16, which is re-read to quantize it to MXFP8 for the w13 backward GEMMs.
- **Where:** feed_forward.py:75; linear.py:129; 61 instances x 4.0 kernels; 42.4 us measured vs 17.1 us ideal per instance (118.4 MB external). Shapes: bf16 [4096,2048] + [4096,2,2048] -> [4096,4096] -> fp8 x2 + e8m0 (dense: [4096,36864]).
- **Dataflow:** silu_and_mul_backward output has a single consumer (the quantizer).
- **Blockers:** Custom op (torchtitan.silu_and_mul_backward) and torchao quantizer are separate kernels.
- **Tier:** T1 -- Pointwise SwiGLU grad; exact amax.
- **Notes:** [working id B9]

### R-25 kv_norm backward + k_pe grad cat + wkv_a dY quantize  (T1, gap 1.39 ms)

- **What:** Per layer, kv_norm's RMSNorm backward (three kernels on [4096,512]) produces the compressed-kv gradient, which is concatenated with the k_pe gradient [4096,64] into [4096,576] and quantized to MXFP8 for the wkv_a backward GEMMs.
- **Where:** fused_mla.py:1069; linear.py:129; nn_modules.py:155; 61 instances x 7.0 kernels; 24.8 us measured vs 2.0 us ideal per instance (13.8 MB external). Shapes: bf16 [4096,512] grads + [4096,64] -> [4096,576] -> fp8 x2 + e8m0 [4096,18|20].
- **Dataflow:** rms bwd output and k_pe grad only feed the cat, which only feeds the quantizer; one kernel can write the concatenated fp8 tensor.
- **Blockers:** The k_pe gradient comes from fused_mla_kv_backward (B6); the strided [4096,576]->[4096,512] slice forces an extra copy kernel in the norm backward.
- **Tier:** T1 -- RMSNorm reductions; exact amax.
- **Overlaps / chains with:** R-08
- **Notes:** Small tensors: launch-bound (7 kernels, 25 us vs 2 us of bytes). [working id B7]

### R-26 dist_moe dispatch-buffer copies and activation save/restore  (T0, gap 1.35 ms)

- **What:** Per MoE layer, the routed-expert forward copies the normalized tokens [4096,7168] into the dispatch buffer and again into an activation buffer for backward recompute; the backward copies them back into the dispatch buffer before recomputing the expert forward.
- **Where:** _blockscaled.py:4377; 58 instances x 3.0 kernels; 57.3 us measured vs 33.9 us ideal per instance (234.9 MB external). Shapes: bf16 [4096,7168] (58.7 MB) per copy.
- **Dataflow:** The forward copies read the same input (ffn_norm output, which the graph already keeps alive for the shared experts and router backward); the backward restore re-creates data that is still resident.
- **Blockers:** Library-internal: dist_moe keeps its own activation buffer for recompute decisions made on device; the dispatch buffer must be in symmetric memory.
- **Tier:** T0 -- Copies only.
- **Notes:** Ideal keeps one forward dispatch copy and the backward restore. [working id F12]

### R-27 q_norm backward + wq_a dY quantize  (T1, gap 1.14 ms)

- **What:** Per layer, q_norm's RMSNorm backward (two kernels on [4096,1536]) produces the wq_a output gradient, which is re-read to quantize it to MXFP8 for the wq_a backward GEMMs.
- **Where:** linear.py:129; nn_modules.py:155; 61 instances x 5.0 kernels; 24.2 us measured vs 5.5 us ideal per instance (38.2 MB external). Shapes: bf16 [4096,1536] x2, rstd fp32 [4096,1] -> fp8 [4096,1536] x2 + e8m0 [4096,48].
- **Dataflow:** rms bwd output has a single consumer (quantize).
- **Blockers:** Separate op owners; weight-grad column reduction needs a second pass.
- **Tier:** T1 -- RMSNorm reductions; exact amax.
- **Notes:** Launch-bound. [working id B8]

### R-28 SwiGLU + w2 MXFP8 quantize  (T1, gap 1.04 ms)

- **What:** Per layer, silu_and_mul turns the w13 output [4096,2,2048] (dense: [4096,2,18432]) into [4096,2048] bf16, which is re-read to quantize it to MXFP8 (both orientations) for the w2 GEMM.
- **Where:** feed_forward.py:75; linear.py:162-170; 61 instances x 4.0 kernels; 27.2 us measured vs 10.2 us ideal per instance (70.9 MB external). Shapes: bf16 [4096,4096] -> [4096,2048] -> fp8 x2 + e8m0 (dense [4096,36864] -> [4096,18432]).
- **Dataflow:** silu_and_mul output only feeds the quantizer (and is not otherwise saved, since the fp8 colwise copy is saved for backward).
- **Blockers:** Custom op + torchao quantizer.
- **Tier:** T1 -- Pointwise; exact amax.
- **Notes:** [working id F7]

### R-29 kv_norm (strided slice) + wkv_b MXFP8 quantize  (T1, gap 0.77 ms)

- **What:** Per layer, kv_norm normalizes the first 512 columns of the wkv_a output [4096,576] (a strided slice that costs an extra copy kernel), and the result is quantized to MXFP8 in both orientations with two scale-swizzle kernels.
- **Where:** linear.py:162-170; nn_modules.py:155; 61 instances x 5.0 kernels; 13.9 us measured vs 1.2 us ideal per instance (8.5 MB external). Shapes: bf16 [4096,576] slice -> [4096,512] -> fp8 x2 + e8m0 [4096,16], [512,128].
- **Dataflow:** Slice copy -> norm -> quantize -> 2 swizzles; all on a 4 MB tensor.
- **Blockers:** aten fused RMSNorm requires a contiguous input; separate torchao quantizer.
- **Tier:** T1 -- RMSNorm reduction order; exact amax.
- **Notes:** Launch-bound. [working id F5]

### R-30 Attention output MXFP8 quantize + scale swizzle  (T1, gap 0.59 ms)

- **What:** Per layer, the attention output [4096,128,128] bf16 is re-read and quantized to MXFP8 for the wo GEMM, and its scales are swizzled by a separate kernel.
- **Where:** linear.py:162-168; 61 instances x 2.0 kernels; 39.0 us measured vs 29.4 us ideal per instance (203.4 MB external). Shapes: bf16 [4096,16384] -> fp8 [4096,16384] + e8m0 [4096,512].
- **Dataflow:** The swizzle reads only the quantizer's scale output; folding it removes one kernel. Further (T3): quantize in the attention forward epilogue.
- **Blockers:** torchao quantizer and swizzle are separate ops; attention epilogue belongs to flash-attention.
- **Tier:** T1 -- Exact amax; swizzle is a fixed permutation.
- **Overlaps / chains with:** R-10
- **Notes:** [working id F6]

### R-31 q_norm + wq_b MXFP8 quantize  (T1, gap 0.45 ms)

- **What:** Per layer, q_norm normalizes the wq_a output [4096,1536] and the result is quantized to MXFP8 for the wq_b GEMM with a separate scale swizzle.
- **Where:** linear.py:162-168; nn_modules.py:155; 61 instances x 3.0 kernels; 11.9 us measured vs 4.6 us ideal per instance (31.7 MB external). Shapes: bf16 [4096,1536] -> fp8 [4096,1536] + e8m0 [4096,48].
- **Dataflow:** norm output feeds the quantizer and is saved for backward (re-quantized in B10).
- **Blockers:** Separate op owners.
- **Tier:** T1 -- RMSNorm reduction; exact amax.
- **Overlaps / chains with:** R-10
- **Notes:** Launch-bound. [working id F4]

### R-32 Embedding dense backward  (T2, gap 0.28 ms)

- **What:** The token-embedding backward accumulates the [4096,7168] input gradient into a [129280,7168] bf16 table gradient with the deterministic sort-based path (21 kernels).
- **Where:** embedding.py:41; 1 instances x 21.0 kernels; 558.9 us measured vs 276.3 us ideal per instance (1912.1 MB external). Shapes: grad bf16 [4096,7168], ids int64 [4096] -> bf16 [129280,7168].
- **Dataflow:** Zero-fill of the table and the segment-sum can be one pass that also writes straight into the reduce-scatter bucket.
- **Blockers:** Deterministic mode forces sort + segment reduction.
- **Tier:** T2 -- Accumulation order for repeated token ids must match the deterministic implementation.
- **Overlaps / chains with:** R-01
- **Notes:** Single instance. [working id L2]

### R-33 Final norm + lm_head input MXFP8 quantize  (T1, gap 0.02 ms)

- **What:** The final RMSNorm output [4096,7168] is re-read to quantize it for the lm_head GEMM, with a separate scale swizzle.
- **Where:** linear.py:162-168; nn_modules.py:155; 1 instances x 3.0 kernels; 41.6 us measured vs 21.4 us ideal per instance (147.7 MB external). Shapes: bf16 [4096,7168] -> fp8 + e8m0 [4096,224].
- **Dataflow:** Same pattern as F2 for the head.
- **Blockers:** Separate op owners.
- **Tier:** T1 -- RMSNorm reduction; exact amax.
- **Overlaps / chains with:** R-15
- **Notes:** Single instance. [working id L3]

## 5. No opportunity

| what | ms | reason |
|---|---:|---|
| dist_moe zero-fill of the [4096,8,7168] bf16 peer-scatter buffers (3 x 470 MB memsets per MoE layer, forward combine buffer and backward dispatch/combine buffers) | 12.49 | Fake-process-group artifact: dist_moe zeroes the local buffer only when the symmetric-memory handle is fake (rows owned by other virtual ranks have no writer). A real run has one physical writer per row and launches no memset, so there is nothing to fuse. |
| Memsets inside the flash-attention forward/backward ops (dpsum/LSE/semaphore init) | 0.26 | Attention-internal fixed-size initialization (< 5 us per call); counted under attention, no external dataflow to fuse. |
| Communication emulation kernels under c10d / _c10d_functional ops (all-gather copies, reduce-scatter) | 90.28 | Stand-ins for collectives in the fake-PG run; excluded from compute and regions by the task rules (their surrounding pack/unpack glue is in the communication-glue regions). |
| Optimizer step (AdamW multi-tensor kernels, including 12.3 ms of the previous step's optimizer tail that executes at the start of the profiled window) | 31.77 | Out of scope: after backward. |
| Post-backward gradient-norm clipping and loss bookkeeping (foreach norms, scalars) | 5.35 | Out of scope: after backward. |

Every in-scope memory-bound kernel class (224 classes by phase x module x op x kernel) is assigned to a region or to the fake-PG memset row above; see `out/glue_kernel_class_coverage.tsv` (`out/unassigned.txt` is empty).

## 6. Observations about working from these artifacts

**What the artifacts made easy**
- Kernel -> op attribution is exact: every graph node runs inside a `"<node>: <op> (<file:line>)"` record-function, the step runs on one CPU
  thread and one CUDA stream, so mapping a kernel to its node is a correlation-id lookup plus interval containment. 99.1% of forward+backward
  kernel time landed on graph nodes with no heuristics.
- `custom.module_fqn` is present on forward *and* backward nodes, so the module of a backward op is a field read; `val` gives shapes, strides
  and dtypes of every input/output, so bytes per node and per region need no Kineto input-dim parsing.
- `args`/`users` give exact producer/consumer edges, so region external bytes were computed mechanically (inputs produced outside the region
  read once, outputs used outside written once). This also exposed duplicate work automatically (attention_norm output quantized twice in
  forward and twice again in backward: the shared producer is deduplicated).
- Single-stream execution means no overlap ambiguity when summing kernel time.

**What was hard**
- Custom ops are opaque nodes: `dist_moe.block_scaled_forward/backward` (15-18 kernels each, incl. GEMMs, routing metadata, combine,
  copies, memsets) and the flash-attention backward (preprocess / main / postprocess). I split them by kernel name and dist_moe's own
  record-function ranges (`moe_preprocess`, `moe_postprocess`, `moe_backward_preprocess`, ...), and read the installed dist_moe and
  flash_attn.cute sources to learn what each internal kernel reads and writes. Bytes for those kernels are estimates from shapes.
- Gradient-accumulation adds carry no `module_fqn`, and their stack trace is only "Gradient addition node due to multiple use of tensor
  around:" with no location. Their module was inferred from direct consumers (common module prefix), falling back to producers.
- FSDP bucket nodes (`_pre_bucket_*`, `split_with_sizes_copy`, reduce-scatter casts) are tagged with the first parameter's module (e.g.
  `layers.N.attention.wq_a` for the whole attention+norm+shared-expert bucket) or with none, and the post-RS casts have no stack trace.
- `nn_module_stack` is empty for every node; only `custom.module_fqn` identifies modules.
- The profiled step took 932 s of wall time (CPU-bound fake PG + Python-stack profiling), so GPU idle is not informative; the 100 MB trace
  holds 3.9 M Python events, which I dropped after a first pass.

**Evidence that was missing or needed workarounds**
- No per-kernel byte counts in the trace: bytes come from graph `val`s; for library-internal kernels from shapes + library source.
- Kineto has no output shapes; graph `val`s filled that gap.
- `_pre_bucket_all_gather`'s `val` is the full gathered buffer although the kernel writes only the local shard; using `val` made the copy-in
  ideal exceed the measured time until I counted shard bytes instead.
- 848 device-to-device copies inside the forward/backward annotation had no graph label; the Python stack showed they come from
  `accumulate_param_grads_` (`param.grad = grad.clone()`) right after the graph. They cannot be assigned to a layer, only aggregated.
- Fake-PG effects had to be recognized from library source: dist_moe zero-fills its 470 MB peer buffers only for fake symmetric memory, and
  routed grouped GEMMs process only rank-local routes (about 5 us per GEMM), so routed-expert GEMM time is not representative of a real run.
  Data is low-entropy: some MXFP8 GEMMs run at or slightly above the random-data peak (e.g. wo at 0.97x of ideal).

**How phase and module were identified for backward ops**
- Phase: the node's `autograd_backward` flag; non-autograd nodes placed after the first backward node (reduce-scatter packing, gradient casts)
  are backward by position. All-gathers occur only before the backward (parameters stay unsharded with `memory_policy=none`).
- Module: `custom.module_fqn` copied from the forward op. Cross-check through `seq_nr` (backward node -> forward nodes with the same `seq_nr`):
  8,251 agree, 58 disagree (the `ones_like` seeds of the aux loss, tagged `router.aux_loss` vs `routed_experts`), 2,833 have no forward
  partner (bucketing / gradient accumulation nodes). Gradient-accumulation adds: consumer/producer rule above.
- Library-internal backward kernels (dist_moe, flash-attention) take the module of their custom-op node.

**False leads discarded**
- "7% of kernel time is unlabeled" (chunk_cat, cat, add): some labels have no `(file:line)` suffix and my first label regex missed them.
- `_pre_bucket_all_gather/_reduce_scatter` looked like communication by name, but they do not run under a collective CPU op; they are glue and
  are in the communication-glue regions.
- The 3 x 72 us memsets per MoE layer (12.5 ms) looked like the biggest single fusion target; they exist only under the fake PG.
- The tiny routed-expert GEMMs (5 us) looked like a GEMM efficiency win; they are a fake-PG artifact (rank-local routes only).
- The backward `scale_and_sum` kernel first looked like a recompute of the forward combine; it is `reduce_from_topk` (sum of route gradients).
- Walking chains of gradient-accumulation adds to find a consumer walked down the whole residual stream and assigned 57 adds to layer 0;
  fixed by looking only at direct consumers.

**Kinds of opportunity I may have missed**
- Fusions inside opaque library kernels (dist_moe CuTe grouped-GEMM prologue/epilogue, e.g. whether the routed-expert dispatch re-quantizes the
  same ffn_norm output that the shared expert quantizes; flash-attention internals).
- Real-EP behaviour: dispatch/combine packing and barriers around the symmetric-memory all-to-all are hidden by the fake PG, and routed GEMMs
  would be 64x larger, which changes which epilogue fusions matter.
- GEMM-shape inefficiencies that are not fusions (wkv_a N=576 at 2x its ideal; wkv_b dgrad 53 us vs 38 us) and alternative GEMM-epilogue
  placements of residual adds (noted in region notes, not separate regions).
- Optimizer-side and CPU-launch-side opportunities (out of scope).

**Effort**
- Wall time: started 17:05 PST; attribution done by 17:12, region inventory by 17:22, metrics and deliverables by 17:31 (about 27 minutes in total, well inside the 80/90-minute budget).
- Scripts (all under `out/`): `load_trace.py`, `peek.py`, `attribute.py`, `node_table.py`, `budget.py`, `classes.py`, `layer_dump.py`,
  `regions_build.py`, `region_meta.py`, `write_outputs.py`, `kernel_csv.py`, `write_report.py`, `observations_text.py`.

## 7. Reproduction

All scripts are CPU-only and read only `artifacts/` and `hw_profile.json`. Run from any directory, in order:

```
cd /home/bahuang/local/step_analysis/dsv3_graph
python -I out/load_trace.py        # trace -> out/trace_events.pkl (reduced event list)
python -I out/peek.py              # drops Python events -> out/trace_nopy.pkl; prints event inventory
python -I out/attribute.py         # kernel -> launch -> CPU-op stack -> graph node -> out/kernels.pkl
python -I out/node_table.py        # per-node phase/module/shapes/bytes/flops/ideal -> out/node_table.{pkl,csv}
python -I out/budget.py            # categories, coverage, idle -> out/budget.json
python -I out/classes.py memory_bound_glue 80   # glue node classes ranked by gap -> out/classes.csv
python -I out/regions_build.py     # region assignment + measured/ideal -> out/region_metrics.json, out/glue_kernel_class_coverage.tsv, out/unassigned.txt
python -I out/write_outputs.py     # -> out/regions.json, out/summary.json
python -I out/kernel_csv.py        # per-kernel attribution -> out/kernel_attribution.csv.gz
python -I out/write_report.py      # -> out/report.md
python -I out/layer_dump.py 10 backward   # optional: execution-ordered node dump for one layer/phase
```
`out/regions_log.jsonl` is the append-only discovery log.
