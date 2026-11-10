# Eager fusion-region finder: DeepSeek-V3 671B, one training step

Substrate: eager `Trainer`, its Python model code, and its runtime traces (Kineto with stacks + execution trace from the same run). Companion data: `regions.json` (schema from `EAGER_REGION_FINDER_PROMPT.md`), `analysis/op_table.parquet` (one row per aten op instance that owns GPU work).

## 1. Setup

- Commit: `c5c112d82e8d369a76368a62282b44718f1a42db` (clean tracked tree; nothing tracked was modified).
- Command: `CUDA_VISIBLE_DEVICES=0 outputs/fusion_finder/eager/run_capture.sh  (= NGPU=256 LOCAL_RANK=0 LOG_RANK=0 PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=outputs/fusion_finder/eager/recipe:. python -m torchtitan.train --module eager_fusion_recipe --config profile_config --comm-backend fake)`
- Recipe: `outputs/fusion_finder/eager/recipe/eager_fusion_recipe.py::profile_config`, derived from `torchtitan_recipes.models.deepseek_v3.deepseek_v3_671b(seq_len=4096)`.
- Config deltas from the recipe:

| field | recipe | this run |
| --- | --- | --- |
| `parallelism.data_parallel_shard_degree` | -1 (auto) | 256 |
| `parallelism.expert_parallel_degree` | 2 | 64 |
| `model` | build_model_config('671B', parallelism=recipe) | rebuilt with the updated parallelism (same flavor; only enable_ep/enable_sp are read) |
| `model.local_compile.regions` | 6 regions (loss, fused_binary_activation, fp32_to_bf16_split, moe_aux_loss, rope, mla_qk) | [] |
| `training.num_tokens_per_microbatch_per_dp_rank` | 16384 (4 x 4096) | 98304 (24 x 4096); no gradient accumulation |
| `training.steps` | 10000 | 10 (profile step 10 after 3 warmup steps) |
| `debug.seed / debug.deterministic` | None / False | 42 / True |
| `activation_remat` | DefaultActivationRematConfig() | FullActivationRematConfig() |
| `profiler` | Profiler.Config() | CaptureProfiler: Kineto CPU+CUDA, record_shapes, with_stack, ExecutionTraceObserver attached |
| `comm.backend` | default (NCCL launcher) | fake (`--comm-backend fake`), NGPU=256 |
| `dump_folder` | ./outputs | outputs/fusion_finder/eager/capture |

- Unchanged and verified at runtime: `training.disable_accelerator_graphs = True` (asserted in the recipe and recorded in `capture/config_deltas.json`), gradient accumulation steps = 1, dataset = recipe default (C4 streaming through the HF proxy).
- Instrumentation added by the recipe (no tracked file touched): forward pre/post hooks on all 1,506 modules open `record_function('## mod <fwd|rc> <fqn> [<class>] ##')`; a patch of `torch_remat._region._CheckpointPhaseContext` opens `## remat_recompute <block> ##` around each RECOMPUTE phase and pauses the execution trace there (see 6).
- GPU: NVIDIA GB300 (GPU 0, 152 SMs). `nvidia-smi` before the capture and before calibration: GPU 0 at 3 MiB, 0% utilization; no other process on GPU 0 during the capture or the calibration (the expected ~17.6 GB tenant was not present; GPU 2 held 14.5 GB from the sibling run). `gpu_busy_during_timing = false`.
- Training numerics of the profiled run (from `capture/train.log`): step 1 loss 12.24438, step 10 loss 12.05490, grad_norm 5.9943, peak memory 257 GiB.

Roofline calibration on GPU 0 (`calib/calib_gpu0.json`, 30 iterations, medians):

| measurement | result |
| --- | --- |
| device-to-device copy, 4 GiB (read + write bytes) | 6,876 GB/s |
| read-only reduction `sum` over 4 GiB bf16 -> 1 fp32 | 6,150 GB/s |
| read-only row reduction (8192-wide rows) | 3,596 GB/s |
| elementwise add, 4 GiB (2 reads + 1 write) | 7,216 GB/s |
| bf16 GEMM dense_ffn_w1 (T,7168)x(7168,18432) | 1,672 TFLOP/s |
| bf16 GEMM dense_ffn_w2 (T,18432)x(18432,7168) | 1,722 TFLOP/s |
| bf16 GEMM attn_wo (T,16384)x(16384,7168) | 1,666 TFLOP/s |
| bf16 GEMM attn_wq_b (T,1536)x(1536,24576) | 1,617 TFLOP/s |
| bf16 GEMM attn_wq_a (T,7168)x(7168,1536) | 1,681 TFLOP/s |
| bf16 GEMM attn_wkv_a (T,7168)x(7168,576) | 1,276 TFLOP/s |
| bf16 GEMM shared_expert_w1 (T,7168)x(7168,2048) | 1,786 TFLOP/s |

Roofline used: **HBM 7,216 GB/s** (best measured streaming bandwidth; the copy alone gives 6,876 GB/s, and several trace kernels exceed it), **bf16 GEMM peak 1,786 TFLOP/s** (best median over the DSv3 shapes with random-normal data). Ideal time = max(bytes / HBM, flops / peak). Caveat: GEMM throughput is data dependent on this part (random data 1.76 PFLOP/s, low-entropy data 2.0, zeros 2.37 for the `wo` shape); in-trace GEMMs reach up to 2.31 PFLOP/s, so about half of the GEMM rows have measured/ideal < 1. GEMMs are out of scope for regions, so this only affects the GEMM column.

## 2. Step budget

Profiled step: `ProfilerStep#9` (the 10th step), CPU wall 42,098.0 ms. GPU window 42,097.2 ms, **GPU busy 41,276.9 ms** (union over 4 streams), **GPU idle 820.3 ms** inside the window.

| category | GPU kernel time (ms) | % of busy |
| --- | ---: | ---: |
| GEMM (mm, addmm, grouped_mm) | 17,084.7 | 41.4% |
| attention (flash fwd/bwd) | 3,218.1 | 7.8% |
| fake collectives (copies under c10d ops) | 1,762.7 | 4.3% |
| memory-bound glue (forward, recompute, backward) | 19,178.8 | 46.5% |
| optimizer (fused AdamW, grad norm, clip) | 19.5 | 0.0% |
| other (input copies + previous-step tail) | 595.7 | 1.4% |
| sum (streams overlap, so the sum exceeds busy) | 41,859.4 | |

By phase: forward 11,261.9 ms, recompute 10,522.5 ms, backward 19,459.9 ms, optimizer 19.5 ms, input staging 0.1 ms.

**Attribution coverage: 98.56%** of GPU-busy time is attributed to an aten op of the profiled step (op, phase, module FQN, source). The remaining 595.6 ms (1179 kernels: GEMMs, flash-attention backward, copies) all run in the first 0.60 s of the window, before the first kernel launched in the window: they belong to step 9, whose launches the CPU issued ahead of the GPU before the active window opened. They are reported under `other`. Every kernel launched inside the window has an owner op (Kineto External id).
Backward provenance: 98.4% of backward GPU time is linked to its forward op; the unlinked rest is FSDP hook work (all-gather/reduce-scatter copies), the embedding backward and the final-norm backward.

Memory-bound glue = 19,178.8 ms: 18,962.8 ms in the 30 regions below, 215.9 ms in `no_opportunity`; every memory-bound row is claimed exactly once (checked by `scripts/regions.py`).

## 3. Ranked regions

Gap = (measured - ideal) summed over instances; % of step = gap / 41,276.9 ms. Regions do not overlap (no op row is counted twice), so gaps add: total gap 16,341.8 ms (39.6% of the step).

| id | title | phases | scope | kind | inst | measured ms | ideal ms | gap ms | % step | tier |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| E-01 | MoE combine: post-all-to-all concat + score multiply + bf16 cast + deterministic scatter-add | F+R | intra-module | gather/segmented-reduction (scatter-add) with pointwise prologue; copy elimination | 116 | 5,693.3 | 204.0 | 5,489.2 | 13.30% | T2 |
| E-02 | MoE combine backward: scatter-add grad gather + score-multiply backward (dgrad, score grad reduction) | B | intra-module | gather + pointwise + row reduction | 58 | 2,854.9 | 192.7 | 2,662.2 | 6.45% | T1 |
| E-03 | MoE dispatch all-to-all backward concat + local-reorder backward (zeros + deterministic scatter-add of K=8 rows per token) | B | intra-module | copy elimination + segmented reduction | 58 | 1,365.1 | 102.0 | 1,263.0 | 3.06% | T2 |
| E-04 | MoE permute backward: zeros + deterministic index_put accumulate over a permutation | B | intra-module | scatter-accumulate -> gather rewrite | 58 | 1,347.4 | 181.3 | 1,166.1 | 2.83% | T0 |
| E-05 | Routed-expert SwiGLU recompute + backward in one pass (h for the w2 weight grad and d_gate_up from one read of gate_up) | R+B | recompute-backward | pointwise | 58 | 1,309.4 | 155.3 | 1,154.1 | 2.80% | T0 |
| E-06 | MoE unpermute (expert-major -> rank-major) as deterministic index_put scatter | F+R | intra-module | permutation (scatter -> gather rewrite) | 116 | 1,402.9 | 362.6 | 1,040.3 | 2.52% | T0 |
| E-07 | MoE dispatch all-to-all concat + rank-major -> expert-major permute gather + permute index math | F+R | intra-module | copy elimination + gather; integer scan for indices | 116 | 913.7 | 362.6 | 551.1 | 1.33% | T0 |
| E-08 | Query RoPE (bf16->fp32 strided copy, complex multiply, fp32->bf16) + concat with q_nope | F+R | cross-module | pointwise + concat | 122 | 670.2 | 163.8 | 506.4 | 1.23% | T0 |
| E-09 | Routed-expert SwiGLU: silu(gate) * up on strided halves of w13 output | F | intra-module | pointwise | 58 | 512.0 | 77.7 | 434.3 | 1.05% | T0 |
| E-10 | MoE local reorder: token counts + argsort + score gather + token gather (all-to-all send buffer) | F+R | intra-module | sort + gather | 116 | 551.7 | 204.6 | 347.1 | 0.84% | T2 |
| E-11 | MLA K/V layout glue: K materialization concat (fwd) and dK/dV pack + dk_pe head-sum (bwd) | F+R+B | intra-module | absorb layout change into attention kernels | 183 | 279.5 | 0.0 | 279.5 | 0.68% | T3 |
| E-12 | MoE combine all-to-all backward concat + unpermute backward gather | B | intra-module | copy elimination + permutation gather | 58 | 448.1 | 181.3 | 266.8 | 0.65% | T0 |
| E-13 | Query RoPE backward + q split backward concat | B | cross-module | pointwise + concat | 61 | 326.8 | 81.9 | 244.9 | 0.59% | T0 |
| E-14 | Router group-limited top-k selection (bias add, group top-2 sum, group top-4, mask, top-8) | F | intra-module | per-token selection (top-k) in registers | 58 | 240.9 | 0.9 | 240.1 | 0.58% | T2 |
| E-15 | Shared-expert SwiGLU recompute + backward in one pass | R+B | recompute-backward | pointwise | 58 | 137.8 | 19.4 | 118.4 | 0.29% | T0 |
| E-16 | Chunked cross-entropy: fp32 upcast + log_softmax + nll (fwd) and nll/log_softmax backward + bf16 cast (bwd) | F+B | intra-module | row reduction (logsumexp) + pointwise | 1 | 92.1 | 7.0 | 85.1 | 0.21% | T1 |
| E-17 | ffn_norm input-grad accumulation (routed + shared + router-gate) + RMSNorm backward + residual grad add | B | cross-module | multi-input add + RMSNorm backward | 61 | 145.2 | 70.3 | 74.9 | 0.18% | T1 |
| E-18 | Dense-FFN SwiGLU recompute + backward in one pass (layers 0-2) | R+B | recompute-backward | pointwise | 3 | 75.7 | 9.0 | 66.6 | 0.16% | T0 |
| E-19 | Router + aux-loss backward (route-norm backward, gather backward, sigmoid backward, aux grads) | B | intra-module | pointwise + scatter + reductions | 58 | 60.7 | 1.9 | 58.8 | 0.14% | T1 |
| E-20 | MoE load-balance statistics (aux loss forward) | F+R | intra-module | column reductions over [T,E] | 116 | 55.5 | 2.0 | 53.5 | 0.13% | T1 |
| E-21 | attention_norm input-grad accumulation (wq_a + wkv_a) + RMSNorm backward + residual grad add | B | cross-module | add + RMSNorm backward | 61 | 110.5 | 59.6 | 51.0 | 0.12% | T1 |
| E-22 | Shared-expert SwiGLU forward | F | intra-module | pointwise | 58 | 55.2 | 9.7 | 45.5 | 0.11% | T0 |
| E-23 | MoE output add (routed + shared) + block residual add + next layer's attention_norm (or final norm) | F | cross-module, cross-layer | multi-input add + RMSNorm | 61 | 97.8 | 59.0 | 38.9 | 0.09% | T1 |
| E-24 | Dense-FFN SwiGLU forward (layers 0-2) | F | intra-module | pointwise | 3 | 30.5 | 4.5 | 26.0 | 0.06% | T0 |
| E-25 | Attention residual add + ffn_norm RMSNorm | F+R | cross-module | add + RMSNorm | 122 | 119.9 | 95.3 | 24.6 | 0.06% | T1 |
| E-26 | Router scoring: sigmoid + gather top-k scores + route norm + scale + routing map + token counts | F+R | intra-module | pointwise + per-token reduction + scatter | 116 | 23.4 | 4.2 | 19.2 | 0.05% | T1 |
| E-27 | HiMidLo router-gate backward: split fp32 grad into bf16 hi/mid pieces + concat | B | intra-module | pointwise bit manipulation + concat | 58 | 20.5 | 1.7 | 18.8 | 0.05% | T0 |
| E-28 | wkv_a output epilogue: kv contiguous copy + kv_norm RMSNorm + k_pe RoPE | F+R | cross-module | norm + pointwise | 122 | 11.4 | 3.8 | 7.6 | 0.02% | T1 |
| E-29 | kv_norm backward + k_pe RoPE backward + kv split backward concat | B | cross-module | norm backward + pointwise + concat | 61 | 9.9 | 2.8 | 7.2 | 0.02% | T1 |
| E-30 | MoE dispatch split bookkeeping (count sums before the host copy) | F+R | intra-module | tiny reductions (launch bound) | 116 | 0.8 | 0.0 | 0.8 | 0.00% | T1 |

Regions per scope tag: cross-layer 1, cross-module 8, intra-module 19, recompute-backward 3 (one region carries two tags).

| area | measured ms | ideal ms | gap ms | % step |
| --- | ---: | ---: | ---: | ---: |
| MoE token dispatch / combine (E: combine, permute, unpermute, local reorder) | 14,577.7 | 1,791.1 | 12,786.6 | 31.0% |
| SwiGLU (routed, shared, dense) | 2,120.7 | 275.7 | 1,844.9 | 4.5% |
| Attention glue (RoPE, MLA layout, kv epilogue) | 1,297.8 | 252.3 | 1,045.4 | 2.5% |
| Router and aux loss | 401.1 | 10.7 | 390.4 | 0.9% |
| Residual adds and RMSNorm | 473.4 | 284.2 | 189.3 | 0.5% |
| Loss | 92.1 | 7.0 | 85.1 | 0.2% |

## 4. Region details

### E-01: MoE combine: post-all-to-all concat + score multiply + bf16 cast + deterministic scatter-add

- **Where:** `layers.{3..60}.feed_forward.routed_experts.token_dispatcher`; torchtitan/models/common/moe/token_dispatcher.py:144-172 (_score_and_scatter_add), token_dispatcher.py:346 (_combine_token_exchange -> spmd_types _AllToAllUneven cat), torchtitan/ops/scatter_add.py:18.
- **What:** ops `aten::_assert_async`, `aten::_index_put_impl_`, `aten::aminmax`, `aten::arange`, `aten::cat`, `aten::copy_`, `aten::div_`, `aten::ge`, `aten::lt`, `aten::mul`, `aten::remainder`, `aten::zero_`; 57.0 kernels per instance, 116 instances per step (forward 2,847.7 ms, recompute 2,845.6 ms). Shapes: routed_output [786432,7168] bf16 (a2a output chunks) * scores [786432,1] fp32 -> fp32 [786432,7168] -> bf16; scatter-add by token index [786432] int64 into zeros [98304,7168] bf16.
- **Numbers:** 49,080 us measured vs 1,759 us ideal per instance (external bytes 12.693 GB); gap 5,489.2 ms (13.30% of step). Evidence: measured.
- **Why fusable (dataflow):** the all-to-all output chunks are concatenated, multiplied row-wise by the score (fp32 product, 22.5 GB), cast to bf16 (11.3 GB) and scatter-added into a zero [98304,7168] buffer; every intermediate has a single consumer inside the span, so one kernel can read each routed row once and write each token row once.
- **Blockers:** The all-to-all boundary sits before the region (real EP); reading per-rank chunks needs spmd_types _AllToAllUneven to expose them or an all_to_all_single into a caller buffer. Deterministic mode forces sort-based index_put. Under full remat this whole span is replayed in recompute only to re-capture saved inputs of the multiply and the scatter-add index; its outputs are dead there.
- **Tier T2:** Product is computed in fp32 and rounded to bf16 per row (keep); each output row is the sum of top_k=8 bf16 rows accumulated by the deterministic sort-based index_put (stable sort by token, fp32 accumulation, one bf16 rounding). A gather-based segmented reduction per token must use the same contribution order (expert-sorted position order) and the same accumulation dtype. The concat is pure data movement.
- **Notes:** Recompute instances (58 of 116, ~2846 ms) produce dead outputs: torch_remat replays the span only because the multiply and the scatter-add save tensors; a combine op whose saved inputs are already persisted would skip it entirely (remat-policy change, not fusion; not counted in the gap). A non-bitwise variant could scatter into the shared-expert output instead of zeros (removes zeros + shared_add) but changes the summation order. Fusing the score multiply into the w2 grouped-GEMM epilogue before the all-to-all is T3 and also changes rounding order.
- **How found:** op table: largest memory-bound class (aten::mul at token_dispatcher.py:155, 21.4 ms/layer); ET dataflow component mul -> copy_ -> index_put; source review of _score_and_scatter_add.

### E-02: MoE combine backward: scatter-add grad gather + score-multiply backward (dgrad, score grad reduction)

- **Where:** `layers.{3..60}.feed_forward.routed_experts.token_dispatcher`; token_dispatcher.py:155-169 (backward of _score_and_scatter_add), ops/scatter_add.py backward.
- **What:** ops `aten::copy_`, `aten::gather`, `aten::mul`, `aten::sum`; 88.0 kernels per instance, 58 instances per step (backward 2,854.9 ms). Shapes: grad_out [98304,7168] bf16 gathered to [786432,7168]; bf16->fp32 grad; *scores fp32; *routed_output fp32 then sum over 7168 -> score grad [786432] fp32; fp32 -> bf16 routed grad.
- **Numbers:** 49,222 us measured vs 3,322 us ideal per instance (external bytes 23.970 GB); gap 2,662.2 ms (6.45% of step). Evidence: measured.
- **Why fusable (dataflow):** grad_out [98304,7168] is gathered to routed order, upcast, multiplied by the score (routed grad, cast to bf16) and by the saved routed output (row-summed into the score grad); all intermediates are single-consumer; external inputs are grad_out, routed_output, scores, indices.
- **Blockers:** Autograd splits this into GatherBackward / ToCopyBackward / MulBackward nodes; a fused version needs a custom autograd op for the whole combine. The all-to-all backward follows immediately.
- **Tier T1:** Gather and casts are exact; the routed-output grad is fp32 grad*score rounded to bf16 (pointwise). The score grad is a 7168-wide fp32 row sum: the fused kernel must reproduce aten's sum reduction tree.
- **Notes:** Pair of E-01 (combine_fwd).
- **How found:** op table backward rows linked by autograd sequence number to the combine forward ops; source review.

### E-03: MoE dispatch all-to-all backward concat + local-reorder backward (zeros + deterministic scatter-add of K=8 rows per token)

- **Where:** `layers.{3..60}.feed_forward.routed_experts.token_dispatcher`; spmd_types a2a backward cat (token_dispatcher.py:321); token_dispatcher.py:89-91 backward (x_TD[token_indices] and topk_scores.view(-1)[idx] backward).
- **What:** ops `aten::_index_put_impl_`, `aten::arange`, `aten::cat`, `aten::div_`, `aten::mul`, `aten::remainder`, `aten::zero_`; 34.0 kernels per instance, 58 instances per step (backward 1,365.1 ms). Shapes: grad [786432,7168] bf16 -> zeros [98304,7168] + index_put_(accumulate) by token index (8 rows per token).
- **Numbers:** 23,535 us measured vs 1,759 us ideal per instance (external bytes 12.696 GB); gap 1,263.0 ms (3.06% of step). Evidence: measured.
- **Why fusable (dataflow):** the all-to-all-backward chunks are concatenated and scatter-added (8 rows per token) into a zero [98304,7168] buffer; the zeros and the concat are consumed only by the index_put.
- **Blockers:** Collective boundary before; IndexBackward materializes zeros; deterministic mode.
- **Tier T2:** Each token row sums 8 bf16 rows in the deterministic sort order with fp32 accumulation; must match exactly.
- **Notes:** The InputBuffer add that follows (routed grad + shared-expert grad) is in E-17 (ffn_norm_bwd_accum).
- **How found:** op table backward rows linked by seq_id to the local reorder gathers.

### E-04: MoE permute backward: zeros + deterministic index_put accumulate over a permutation

- **Where:** `layers.{3..60}.feed_forward.routed_experts.token_dispatcher`; token_dispatcher.py:563 (routed_input_RD[permuted_indices, :] backward).
- **What:** ops `aten::_index_put_impl_`, `aten::arange`, `aten::div_`, `aten::mul`, `aten::remainder`, `aten::zero_`; 16.0 kernels per instance, 58 instances per step (backward 1,347.4 ms). Shapes: zeros [786432,7168] bf16; index_put_(accumulate=True) of grad [786432,7168] bf16 by permuted_indices.
- **Numbers:** 23,231 us measured vs 3,126 us ideal per instance (external bytes 22.555 GB); gap 1,166.1 ms (2.83% of step). Evidence: measured.
- **Why fusable (dataflow):** IndexBackward zero-fills [786432,7168] and accumulates the w13-input grad through permuted_indices; the zero buffer has no other consumer and the indices are a permutation.
- **Blockers:** IndexBackward always materializes zeros and accumulates; replacing it needs a custom autograd op that knows the indices are a permutation.
- **Tier T0:** Indices form a permutation, so each element is 0 + g computed in fp32 and rounded: equal to g except that -0.0 becomes +0.0. A gather by the inverse permutation must add +0.0 to match.
- **How found:** op table backward rows (largest backward index_put); seq_id link to _permute.

### E-05: Routed-expert SwiGLU recompute + backward in one pass (h for the w2 weight grad and d_gate_up from one read of gate_up)

- **Where:** `layers.{3..60}.feed_forward.routed_experts.activation_fn`; activation.py:61, 174-176 (torch_remat replay + backward).
- **What:** ops `aten::cat`, `aten::mul`, `aten::silu`, `aten::silu_backward`; 26.0 kernels per instance, 58 instances per step (recompute 525.6 ms, backward 783.8 ms). Shapes: recompute: gate_up [786432,2,2048] -> h [786432,2048]; backward: d_h + gate_up -> d_gate_up [786432,2,2048] bf16.
- **Numbers:** 22,576 us measured vs 2,678 us ideal per instance (external bytes 19.327 GB); gap 1,154.1 ms (2.80% of step). Evidence: measured.
- **Why fusable (dataflow):** the recompute reads gate_up and writes h (consumed only by the w2 weight-grad GEMM); the backward reads d_h, gate_up and the saved silu output and writes d_gate_up through two muls, silu_backward and a stack; gate_up is read by both phases and every intermediate is single-consumer.
- **Blockers:** Autograd node boundaries (MulBackward0, SiluBackward0, UnbindBackward0). The one-pass form also needs torch_remat to stop replaying the activation (it replays the whole block before the block's backward) and the w2 grouped-GEMM backward to run dgrad before the activation kernel and wgrad after it (one node today).
- **Tier T0:** Pointwise; the fused kernel must round silu(gate) to bf16 where eager saved it, and evaluate silu_backward with aten's fp32 formula (FMA contraction must match).
- **Notes:** Fallback without reordering: one fused recompute kernel + one fused backward kernel, ideal 1339 + 2232 = 3571 us per layer (vs 2678 us one-pass).
- **How found:** op table backward rows linked to activation.py by seq_id; recompute-to-backward edges (rc_to_bwd) from the silu/mul replay to the mul backward.

### E-06: MoE unpermute (expert-major -> rank-major) as deterministic index_put scatter

- **Where:** `layers.{3..60}.feed_forward.routed_experts.token_dispatcher`; token_dispatcher.py:567-572 (_unpermute).
- **What:** ops `aten::_assert_async`, `aten::_index_put_impl_`, `aten::aminmax`, `aten::arange`, `aten::div_`, `aten::ge`, `aten::lt`, `aten::mul`, `aten::remainder`; 21.0 kernels per instance, 116 instances per step (forward 696.5 ms, recompute 706.4 ms). Shapes: out[permuted_indices,:] = routed_output, [786432,7168] bf16, permuted_indices [786432] int64.
- **Numbers:** 12,094 us measured vs 3,126 us ideal per instance (external bytes 22.555 GB); gap 1,040.3 ms (2.52% of step). Evidence: measured.
- **Why fusable (dataflow):** the w2 output [786432,7168] is scattered through permuted_indices into a fresh buffer that is the all-to-all send buffer; no other consumer.
- **Blockers:** Deterministic mode routes index_put through the sort-based kernel even without accumulation. The output feeds the combine all-to-all, so fusion with the combine is blocked by the collective; fusion with the w2 grouped GEMM epilogue would be T3.
- **Tier T0:** permuted_indices is a permutation (no duplicates), so a gather by the inverse permutation (or a plain scatter) writes identical bits; only the bounds checks of index_put are extra.
- **How found:** op table: aten::_index_put_impl_ at token_dispatcher.py:569 at 3.4x its roofline; source review.

### E-07: MoE dispatch all-to-all concat + rank-major -> expert-major permute gather + permute index math

- **Where:** `layers.{3..60}.feed_forward.routed_experts.token_dispatcher`; token_dispatcher.py:321 (a2a concat), 517-565 (_permute), moe.py:136 (offsets cumsum).
- **What:** ops `aten::add`, `aten::arange`, `aten::cat`, `aten::copy_`, `aten::cumsum`, `aten::gather`, `aten::index`, `aten::repeat_interleave`, `aten::sub`, `aten::sum`; 32.0 kernels per instance, 116 instances per step (forward 455.1 ms, recompute 458.6 ms). Shapes: a2a output chunks [786432,7168] bf16 -> concat -> routed[permuted_indices]; index math on [256] counts -> [786432] int64.
- **Numbers:** 7,876 us measured vs 3,126 us ideal per instance (external bytes 22.555 GB); gap 551.1 ms (1.33% of step). Evidence: measured.
- **Why fusable (dataflow):** the all-to-all output chunks are concatenated and gathered by permuted_indices, which ~15 tiny integer kernels derive from the per-expert counts; the concat has one consumer.
- **Blockers:** Collective boundary before; permuted_indices must exist before the gather (two-pass or a scan kernel).
- **Tier T0:** Data movement plus integer cumsum/arange/repeat_interleave (exact in any order).
- **Notes:** The permute gather could be folded into the w13 grouped-GEMM A-operand load (T3).
- **How found:** op table + ET dataflow component (cat -> index); source review of _permute.

### E-08: Query RoPE (bf16->fp32 strided copy, complex multiply, fp32->bf16) + concat with q_nope

- **Where:** `layers.{0..60}.attention (+ .rope)`; torchtitan/models/deepseek_v3/attention.py:196-203 (_apply_rope_to_q_and_k_pe), torchtitan/models/common/rope.py:280-328 (ComplexRoPE.apply_rotary_emb); local_compile mla_qk (disabled here).
- **What:** ops `aten::_assert_async`, `aten::all`, `aten::cat`, `aten::copy_`, `aten::index`, `aten::le`, `aten::mul`; 20.0 kernels per instance, 122 instances per step (forward 336.5 ms, recompute 333.7 ms). Shapes: q [98304,128,192] bf16 (q_pe slice [.., 64] strided) -> q' [98304,128,192] bf16; k_pe [98304,1,64].
- **Numbers:** 5,493 us measured vs 1,343 us ideal per instance (external bytes 9.691 GB); gap 506.4 ms (1.23% of step). Evidence: measured.
- **Why fusable (dataflow):** q_pe, a strided slice of the wq_b output, is upcast to fp32, multiplied by the complex cache, cast back and concatenated with q_nope into a new q; every intermediate has one consumer.
- **Blockers:** Spans RoPE module and the attention method; the existing mla_qk local_compile region covers it but is disabled in this setup.
- **Tier T0:** Complex multiply in fp32 (re = ac - bd, im = ad + bc) then round to bf16; FMA contraction must match aten's complex mul kernel.
- **Notes:** An in-place variant (rotate only the 64-wide tail of q) would need 3.2 GB instead of 9.7 GB.
- **How found:** op table: strided copy_ at rope.py:321 at 3x roofline; ET dataflow copy -> mul -> copy -> cat.

### E-09: Routed-expert SwiGLU: silu(gate) * up on strided halves of w13 output

- **Where:** `layers.{3..60}.feed_forward.routed_experts.activation_fn`; torchtitan/models/common/activation.py:61 (unbind), 174-176 (SwiGLU._activation_fn); local_compile region fused_binary_activation (disabled here).
- **What:** ops `aten::mul`, `aten::silu`; 8.0 kernels per instance, 58 instances per step (forward 512.0 ms). Shapes: gate_up [786432,2,2048] bf16 -> h [786432,2048] bf16.
- **Numbers:** 8,828 us measured vs 1,339 us ideal per instance (external bytes 9.664 GB); gap 434.3 ms (1.05% of step). Evidence: measured.
- **Why fusable (dataflow):** gate and up are strided halves of the w13 output; silu(gate) is written to HBM and read once by the multiply.
- **Blockers:** None structural (the existing local_compile region targets it); eager runs two strided kernels at ~30% of roofline.
- **Tier T0:** silu runs in fp32 and rounds to bf16, then the bf16*bf16 product rounds again: keep both roundings.
- **Notes:** Also the recompute instance; see E-05 (swiglu_routed_rc_bwd) for the recompute-backward fusion.
- **How found:** op table: aten::silu / aten::mul at activation.py:175-176 at 3.6x roofline.

### E-10: MoE local reorder: token counts + argsort + score gather + token gather (all-to-all send buffer)

- **Where:** `layers.{3..60}.feed_forward.routed_experts.token_dispatcher`; token_dispatcher.py:59-95 (_local_reorder), :394 (routing_map.sum).
- **What:** ops `aten::copy_`, `aten::floor_divide`, `aten::index`, `aten::sort`, `aten::sum`; 35.0 kernels per instance, 116 instances per step (forward 275.6 ms, recompute 276.0 ms). Shapes: x [98304,7168] bf16 gathered by token index -> [786432,7168]; argsort of [786432] int64; routing_map [98304,256] bool summed over tokens.
- **Numbers:** 4,756 us measured vs 1,764 us ideal per instance (external bytes 12.728 GB); gap 347.1 ms (0.84% of step). Evidence: measured.
- **Why fusable (dataflow):** routing-map counts, the argsort of expert ids, the score gather and the token gather all derive from router outputs; the sorted indices feed only the gathers.
- **Blockers:** The gather output is the all-to-all send buffer (fine); the sort is a separate pass; deterministic mode.
- **Tier T2:** Stable argsort order must be reproduced (ties broken by position); gathers and integer counts are exact.
- **Notes:** The bool->int64 copy before routing_map.sum writes 201 MB per call; the count could come from the sort.
- **How found:** op table + ET component; the token gather re-reads each x row 8 times (4.4 ms vs 1.8 ms ideal).

### E-11: MLA K/V layout glue: K materialization concat (fwd) and dK/dV pack + dk_pe head-sum (bwd)

- **Where:** `layers.{0..60}.attention.inner_attention`; torchtitan/models/common/attention/mla.py:38-48 (materialize_mla_kv).
- **What:** ops `aten::cat`, `aten::sum`; 9.33 kernels per instance, 183 instances per step (forward 72.6 ms, recompute 72.7 ms, backward 134.2 ms). Shapes: k = cat(k_nope [98304,128,128] (strided view of kv), k_pe expand) -> [98304,128,192]; bwd: cat(dk_nope, dv) -> [98304,128,256], sum dk_pe over heads.
- **Numbers:** 1,527 us measured vs 0 us ideal per instance (external bytes 0.000 GB); gap 279.5 ms (0.68% of step). Evidence: estimated.
- **Why fusable (dataflow):** k = cat(k_nope view of the wkv_b output, broadcast k_pe) is consumed only by flash attention; in backward dk and dv from flash attention are only repacked (cat) and head-summed for the wkv_b dgrad.
- **Blockers:** Requires attention-kernel changes (FA4 cute kernels).
- **Tier T3:** Needs the flash-attention kernels to read k_nope/k_pe and write dk_nope/dv/dk_pe in the packed layout.
- **Notes:** Ideal assumes the attention kernels absorb the layout at no extra cost (estimated).
- **How found:** op table: standalone the copies are at roofline, so the only gain is elimination.

### E-12: MoE combine all-to-all backward concat + unpermute backward gather

- **Where:** `layers.{3..60}.feed_forward.routed_experts.token_dispatcher`; spmd_types _AllToAllUneven backward cat (called at token_dispatcher.py:346); token_dispatcher.py:569 (_unpermute backward = index gather).
- **What:** ops `aten::cat`, `aten::index`; 10.0 kernels per instance, 58 instances per step (backward 448.1 ms). Shapes: [786432,7168] bf16 chunks -> concat -> gather by permuted_indices.
- **Numbers:** 7,726 us measured vs 3,126 us ideal per instance (external bytes 22.555 GB); gap 266.8 ms (0.65% of step). Evidence: measured.
- **Why fusable (dataflow):** the combine all-to-all backward output chunks are concatenated and gathered by permuted_indices; the concat has one consumer.
- **Blockers:** Collective boundary before the concat; needs per-chunk access (spmd_types) to skip the concat.
- **Tier T0:** Pure data movement.
- **How found:** op table backward rows; ET shows cat output consumed only by the index gather.

### E-13: Query RoPE backward + q split backward concat

- **Where:** `layers.{0..60}.attention (+ .rope)`; attention.py:196-203 and rope.py:316-328 backward.
- **What:** ops `aten::cat`, `aten::copy_`, `aten::mul`; 16.0 kernels per instance, 61 instances per step (backward 326.8 ms). Shapes: dq [98304,128,192] bf16 -> rotate dq_pe in fp32 -> concat [dq_nope, dq_pe] -> [98304,128,192] bf16.
- **Numbers:** 5,357 us measured vs 1,343 us ideal per instance (external bytes 9.689 GB); gap 244.9 ms (0.59% of step). Evidence: measured.
- **Why fusable (dataflow):** dq from flash attention: the rope part is upcast, rotated by the conjugate cache, cast back and concatenated with dq_nope for the wq_b dgrad; a single-consumer chain.
- **Blockers:** Autograd nodes CatBackward/ToCopyBackward/MulBackward/SplitWithSizesBackward.
- **Tier T0:** As E-08 (rope_q_fwd) with the conjugate cache.
- **How found:** op table backward rows.

### E-14: Router group-limited top-k selection (bias add, group top-2 sum, group top-4, mask, top-8)

- **Where:** `layers.{3..60}.feed_forward.router`; torchtitan/models/deepseek_v3/moe.py:30-83 (DeepSeekV3Router._select_experts), remat region routing_decision (recompute=False).
- **What:** ops `aten::add`, `aten::copy_`, `aten::fill_`, `aten::masked_fill_`, `aten::scatter_`, `aten::sum`, `aten::topk`; 10.0 kernels per instance, 58 instances per step (forward 240.9 ms). Shapes: scores [98304,256] fp32 -> [98304,8,32] top-2 -> [98304,8] top-4 -> masked [98304,256] -> top-8 ids int64.
- **Numbers:** 4,154 us measured vs 15 us ideal per instance (external bytes 0.107 GB); gap 240.1 ms (0.58% of step). Evidence: measured.
- **Why fusable (dataflow):** scores [98304,256] -> +bias -> group top-2 -> sum -> group top-4 -> mask -> top-8; every intermediate is per token with one consumer.
- **Blockers:** Forward only (saved region). aten::topk on short rows is slow (2.1 ms for k=2 over 32, 1.6 ms for k=8 over 256).
- **Tier T2:** topk(sorted=False) index order and tie-breaking must match aten's implementation; values are exact copies.
- **How found:** op table: aten::topk at 120x-2000x its per-op roofline.

### E-15: Shared-expert SwiGLU recompute + backward in one pass

- **Where:** `layers.{3..60}.feed_forward.shared_experts`; activation.py:61, 174-176 (replay + backward).
- **What:** ops `aten::cat`, `aten::mul`, `aten::silu`, `aten::silu_backward`; 6.0 kernels per instance, 58 instances per step (recompute 55.2 ms, backward 82.6 ms). Shapes: [98304,2,2048] -> h [98304,2048]; d_h + gate_up -> [98304,2,2048] bf16.
- **Numbers:** 2,376 us measured vs 335 us ideal per instance (external bytes 2.416 GB); gap 118.4 ms (0.29% of step). Evidence: measured.
- **Why fusable (dataflow):** as the routed variant, at [98304,2,2048].
- **Blockers:** As E-05 (swiglu_routed_rc_bwd) (the w2 _LinearFunctionBackward computes dgrad and wgrad in one node).
- **Tier T0:** As E-05 (swiglu_routed_rc_bwd).
- **How found:** op table.

### E-16: Chunked cross-entropy: fp32 upcast + log_softmax + nll (fwd) and nll/log_softmax backward + bf16 cast (bwd)

- **Where:** `loss (ChunkedLossWrapper, 8 chunks of 12288 tokens)`; torchtitan/components/loss.py:80-86 (cross_entropy), 600-830 (ChunkedLossWrapper); local_compile region loss (disabled here).
- **What:** ops `aten::_log_softmax`, `aten::_log_softmax_backward_data`, `aten::copy_`, `aten::nll_loss_backward`, `aten::nll_loss_forward`, `aten::zero_`; 104.0 kernels per instance, 1 instances per step (forward 42.2 ms, backward 49.9 ms). Shapes: logits [12288,129280] bf16 per chunk x 8 -> loss + grad logits bf16.
- **Numbers:** 92,124 us measured vs 7,045 us ideal per instance (external bytes 50.836 GB); gap 85.1 ms (0.21% of step). Evidence: measured.
- **Why fusable (dataflow):** per chunk, logits bf16 -> fp32 copy -> log_softmax -> nll; the backward zero-fills a [12288,129280] fp32 grad, scatters the targets, runs log_softmax backward and casts to bf16, right after the forward of the same chunk; all [12288,129280] intermediates are single-consumer.
- **Blockers:** Chunk forward and backward run inside the loss call; the existing loss local_compile region covers it.
- **Tier T1:** logsumexp reduction order and fp32 math must match aten's log_softmax / nll kernels.
- **Notes:** One instance = the whole step's loss (8 chunks).
- **How found:** op table rows under components/loss.py.

### E-17: ffn_norm input-grad accumulation (routed + shared + router-gate) + RMSNorm backward + residual grad add

- **Where:** `layers.{0..60}.ffn_norm (+ InputBuffer adds of its output grad)`; nn_modules.py:155 backward; autograd InputBuffer for ffn_norm output (token_dispatcher.py:89, hi_mid_lo_linear.py:66, shared_experts).
- **What:** ops `_native::_fused_rms_norm_backward_cutedsl_5`, `aten::add`, `aten::add_`; 4.9 kernels per instance, 61 instances per step (backward 145.2 ms). Shapes: up to 3 grads [98304,7168] bf16 + h + residual grad -> grad_h [98304,7168] bf16.
- **Numbers:** 2,380 us measured vs 1,153 us ideal per instance (external bytes 8.317 GB); gap 74.9 ms (0.18% of step). Evidence: measured.
- **Why fusable (dataflow):** autograd's InputBuffer sums the three branch grads of ffn_norm's output (routed, shared, router gate) with two add kernels; the sum is consumed only by the RMSNorm backward, whose output is consumed only by the residual-grad add_.
- **Blockers:** Autograd engine owns the InputBuffer accumulation; fusing needs a custom autograd op owning the norm and its consumers.
- **Tier T1:** InputBuffer sums 3 grads in arrival order (3 terms do not commute bitwise): keep the order. RMSNorm backward reductions must match.
- **Notes:** Dense layers (0-2) have no accumulation adds (single consumer).
- **How found:** op table: [T,D] add/add_ kernels between rmsnorm backward and its producers; ET grad edges.

### E-18: Dense-FFN SwiGLU recompute + backward in one pass (layers 0-2)

- **Where:** `layers.{0,1,2}.feed_forward`; activation.py:61, 174-176 (replay + backward).
- **What:** ops `aten::cat`, `aten::mul`, `aten::silu`, `aten::silu_backward`; 26.0 kernels per instance, 3 instances per step (recompute 30.8 ms, backward 44.9 ms). Shapes: [98304,2,18432] -> h [98304,18432]; d_h + gate_up -> [98304,2,18432] bf16.
- **Numbers:** 25,225 us measured vs 3,013 us ideal per instance (external bytes 21.743 GB); gap 66.6 ms (0.16% of step). Evidence: measured.
- **Why fusable (dataflow):** as the routed variant, at [98304,2,18432].
- **Blockers:** As E-15 (swiglu_shared_rc_bwd).
- **Tier T0:** As E-05 (swiglu_routed_rc_bwd).
- **How found:** op table.

### E-19: Router + aux-loss backward (route-norm backward, gather backward, sigmoid backward, aux grads)

- **Where:** `layers.{3..60}.feed_forward.router`; moe.py:232-330 and 490-610 backward.
- **What:** ops `aten::_assert_async`, `aten::_index_put_impl_`, `aten::add_`, `aten::aminmax`, `aten::arange`, `aten::copy_`, `aten::div`, `aten::div_`, `aten::fill_`, `aten::ge`, `aten::gt`, `aten::lt`, `aten::mul`, `aten::neg`, `aten::remainder`, `aten::sgn`, `aten::sigmoid_backward`, `aten::sum`, `aten::where`, `aten::zero_`; 61.0 kernels per instance, 58 instances per step (backward 60.7 ms). Shapes: grad topk scores [98304,8] fp32 -> grad gate [98304,256] fp32.
- **Numbers:** 1,047 us measured vs 33 us ideal per instance (external bytes 0.236 GB); gap 58.8 ms (0.14% of step). Evidence: measured.
- **Why fusable (dataflow):** the grad of the top-k scores flows through route-scale and route-norm backward, the gather backward (scatter into [98304,256]), aux-loss backward and sigmoid backward into the gate grad; a per-token chain of small tensors.
- **Blockers:** Many small autograd nodes; deterministic index_put for the gather backward.
- **Tier T1:** Reductions (route-norm sum backward, aux sums) must keep aten's order.
- **How found:** op table backward rows linked to router forward ops.

### E-20: MoE load-balance statistics (aux loss forward)

- **Where:** `layers.{3..60}.feed_forward.router.aux_loss`; torchtitan/models/common/moe/moe.py:490-610, torchtitan/models/common/aux_loss.py:205-236.
- **What:** ops `aten::add_`, `aten::bitwise_not`, `aten::clamp_min`, `aten::copy_`, `aten::div`, `aten::linalg_vector_norm`, `aten::mul`, `aten::reciprocal`, `aten::sum`; 19.5 kernels per instance, 116 instances per step (forward 28.4 ms, recompute 27.1 ms). Shapes: scores [98304,256] fp32, masked routing map bool -> per-expert sums [256].
- **Numbers:** 479 us measured vs 17 us ideal per instance (external bytes 0.126 GB); gap 53.5 ms (0.13% of step). Evidence: measured.
- **Why fusable (dataflow):** scores and the masked routing map are reduced over tokens into per-expert statistics; all intermediates are single-consumer.
- **Blockers:** Interleaved with the routing decision; norm/clamp/div chain.
- **Tier T1:** fp32 column sums over 98304 tokens: reduction order must match aten.
- **How found:** op table.

### E-21: attention_norm input-grad accumulation (wq_a + wkv_a) + RMSNorm backward + residual grad add

- **Where:** `layers.{0..60}.attention_norm`; nn_modules.py:155 backward; InputBuffer for attention_norm output (wq_a, wkv_a).
- **What:** ops `_native::_fused_rms_norm_backward_cutedsl_5`, `aten::add`, `aten::add_`; 4.0 kernels per instance, 61 instances per step (backward 110.5 ms). Shapes: 2 grads [98304,7168] + x + residual grad -> grad_x [98304,7168] bf16.
- **Numbers:** 1,812 us measured vs 977 us ideal per instance (external bytes 7.047 GB); gap 51.0 ms (0.12% of step). Evidence: measured.
- **Why fusable (dataflow):** the wq_a and wkv_a dgrads are summed by an InputBuffer add, consumed only by the RMSNorm backward, whose output is consumed only by the residual-grad add_.
- **Blockers:** InputBuffer accumulation owned by the autograd engine.
- **Tier T1:** Two-term add commutes exactly; RMSNorm backward reductions must match.
- **How found:** op table + ET grad edges.

### E-22: Shared-expert SwiGLU forward

- **Where:** `layers.{3..60}.feed_forward.shared_experts`; activation.py:61, 174-176.
- **What:** ops `aten::mul`, `aten::silu`; 2.0 kernels per instance, 58 instances per step (forward 55.2 ms). Shapes: [98304,2,2048] bf16 -> [98304,2048] bf16.
- **Numbers:** 952 us measured vs 167 us ideal per instance (external bytes 1.208 GB); gap 45.5 ms (0.11% of step). Evidence: measured.
- **Why fusable (dataflow):** as swiglu_routed_fwd at [98304,2,2048].
- **Blockers:** As E-09 (swiglu_routed_fwd).
- **Tier T0:** As E-09 (swiglu_routed_fwd).
- **Notes:** Dataflow: as E-09 (swiglu_routed_fwd) at [98304,2,2048].
- **How found:** op table.

### E-23: MoE output add (routed + shared) + block residual add + next layer's attention_norm (or final norm)

- **Where:** `layers.L.feed_forward (shared_add) + layers.L (ffn_residual) + layers.L+1.attention_norm`; moe.py:822-826; model.py:66-70; nn_modules.py:155 (next block).
- **What:** ops `_native::_fused_rms_norm_cutedsl_4`, `aten::add`; 2.95 kernels per instance, 61 instances per step (forward 97.8 ms). Shapes: routed, shared, h [98304,7168] bf16 -> x_next + attention_norm(x_next) [98304,7168] bf16.
- **Numbers:** 1,604 us measured vs 967 us ideal per instance (external bytes 6.978 GB); gap 38.9 ms (0.09% of step). Evidence: measured.
- **Why fusable (dataflow):** routed + shared (shared_add) -> + h (ffn_residual) -> x_next, which the next block's attention_norm reads first (x_next is also the next checkpoint's saved input).
- **Blockers:** Crosses the transformer-block boundary and the torch_remat checkpoint boundary (the residual adds are saved regions; the next block's norm is inside the next checkpoint and replays alone in recompute).
- **Tier T1:** Two-term adds in the same order ((routed+shared) then h + ...); RMSNorm reduction must match.
- **Notes:** Forward only: in recompute the adds are skipped (saved regions) and attention_norm reads the saved block input.
- **How found:** ET dataflow (cross-layer edge add -> rms_norm); op table.

### E-24: Dense-FFN SwiGLU forward (layers 0-2)

- **Where:** `layers.{0,1,2}.feed_forward`; activation.py:61, 174-176.
- **What:** ops `aten::mul`, `aten::silu`; 8.0 kernels per instance, 3 instances per step (forward 30.5 ms). Shapes: [98304,2,18432] bf16 -> [98304,18432] bf16.
- **Numbers:** 10,174 us measured vs 1,507 us ideal per instance (external bytes 10.872 GB); gap 26.0 ms (0.06% of step). Evidence: measured.
- **Why fusable (dataflow):** as swiglu_routed_fwd at [98304,2,18432].
- **Blockers:** As E-09 (swiglu_routed_fwd).
- **Tier T0:** As E-09 (swiglu_routed_fwd).
- **Notes:** Dataflow: as E-09 (swiglu_routed_fwd) at [98304,2,18432].
- **How found:** op table.

### E-25: Attention residual add + ffn_norm RMSNorm

- **Where:** `layers.{0..60} (block) + layers.{0..60}.ffn_norm`; torchtitan/models/deepseek_v3/model.py:56-59; nn_modules.py:155.
- **What:** ops `_native::_fused_rms_norm_cutedsl_4`, `aten::add`; 2.0 kernels per instance, 122 instances per step (forward 60.0 ms, recompute 59.9 ms). Shapes: x, attn_out [98304,7168] bf16 -> h, ffn_norm(h) [98304,7168] bf16 + rstd.
- **Numbers:** 983 us measured vs 781 us ideal per instance (external bytes 5.638 GB); gap 24.6 ms (0.06% of step). Evidence: measured.
- **Why fusable (dataflow):** x + attn_out -> h, read by ffn_norm (and later by the trailing residual add).
- **Blockers:** Spans block forward and the RMSNorm module.
- **Tier T1:** The add is exact-order (two terms); RMSNorm reduction must match the CuTeDSL kernel.
- **How found:** ET dataflow: add output consumed by rms_norm; op table.

### E-26: Router scoring: sigmoid + gather top-k scores + route norm + scale + routing map + token counts

- **Where:** `layers.{3..60}.feed_forward.router`; torchtitan/models/common/moe/moe.py:232-330 (TokenChoiceTopKRouter.forward).
- **What:** ops `aten::add`, `aten::add_`, `aten::bitwise_and`, `aten::bitwise_not`, `aten::copy_`, `aten::div`, `aten::gather`, `aten::mul`, `aten::scatter_`, `aten::sigmoid`, `aten::sum`, `aten::zero_`; 12.0 kernels per instance, 116 instances per step (forward 17.1 ms, recompute 6.3 ms). Shapes: gate [98304,256] fp32 -> sigmoid; gather [98304,8]; sum/div/mul; routing map [98304,256] bool; masked map; counts [256].
- **Numbers:** 202 us measured vs 36 us ideal per instance (external bytes 0.261 GB); gap 19.2 ms (0.05% of step). Evidence: measured.
- **Why fusable (dataflow):** sigmoid(gate) -> gather of the top-k scores -> sum, divide, scale; routing-map scatter -> masked map -> counts; per-token chains.
- **Blockers:** The routing decision (topk) sits in a saved remat region between sigmoid and gather.
- **Tier T1:** The route-norm sum over 8 values in fp32 must keep aten's order; counts are integer sums (exact).
- **Notes:** tokens_per_expert accumulation (moe.py:313) runs only in forward.
- **How found:** op table small-kernel sweep of the router module.

### E-27: HiMidLo router-gate backward: split fp32 grad into bf16 hi/mid pieces + concat

- **Where:** `layers.{3..60}.feed_forward.router.gate`; torchtitan/models/common/linear/hi_mid_lo_linear.py:140-220 (_HiMidLoLinearFunction.backward).
- **What:** ops `aten::__lshift__`, `aten::add_`, `aten::cat`, `aten::copy_`, `aten::sub`; 13.0 kernels per instance, 58 instances per step (backward 20.5 ms). Shapes: grad [98304,256] fp32 -> hi/mid bf16 pieces concatenated for the backward GEMMs.
- **Numbers:** 353 us measured vs 29 us ideal per instance (external bytes 0.209 GB); gap 18.8 ms (0.05% of step). Evidence: measured.
- **Why fusable (dataflow):** the fp32 gate grad is split into bf16 hi/mid pieces by casts, shifts and subtractions, then concatenated for the backward GEMMs; a single-consumer chain.
- **Blockers:** Lives inside a custom autograd Function between GEMMs.
- **Tier T0:** Bit-exact casts/shifts/subtractions.
- **How found:** op table.

### E-28: wkv_a output epilogue: kv contiguous copy + kv_norm RMSNorm + k_pe RoPE

- **Where:** `layers.{0..60}.attention.{kv_norm,rope}`; attention.py:128-145; nn_modules.py:155 (RMSNorm); rope.py:324-328.
- **What:** ops `_native::_fused_rms_norm_cutedsl_4`, `aten::copy_`, `aten::mul`; 5.0 kernels per instance, 122 instances per step (forward 5.7 ms, recompute 5.7 ms). Shapes: wkv_a out [98304,576] bf16 -> kv_norm [98304,512] + k_pe rope [98304,1,64].
- **Numbers:** 94 us measured vs 31 us ideal per instance (external bytes 0.227 GB); gap 7.6 ms (0.02% of step). Evidence: measured.
- **Why fusable (dataflow):** the wkv_a output [98304,576] is split; the kv part is made contiguous and RMS-normalized, the k_pe part is rotated; each piece has one consumer.
- **Blockers:** Small; spans RMSNorm and RoPE modules.
- **Tier T1:** RMSNorm reduction order must match the CuTeDSL kernel; RoPE as E-08 (rope_q_fwd).
- **How found:** op table.

### E-29: kv_norm backward + k_pe RoPE backward + kv split backward concat

- **Where:** `layers.{0..60}.attention.{kv_norm,rope}`; attention.py:128-145 backward.
- **What:** ops `_native::_fused_rms_norm_backward_cutedsl_5`, `aten::cat`, `aten::copy_`, `aten::mul`; 9.0 kernels per instance, 61 instances per step (backward 9.9 ms). Shapes: -> d(wkv_a out) [98304,576] bf16.
- **Numbers:** 163 us measured vs 45 us ideal per instance (external bytes 0.328 GB); gap 7.2 ms (0.02% of step). Evidence: measured.
- **Why fusable (dataflow):** the reverse: kv_norm backward, k_pe rope backward, and a concat into the wkv_a output grad.
- **Blockers:** Small.
- **Tier T1:** RMSNorm backward reductions must match.
- **How found:** op table.

### E-30: MoE dispatch split bookkeeping (count sums before the host copy)

- **Where:** `layers.{3..60}.feed_forward.routed_experts.token_dispatcher`; token_dispatcher.py:264-301 (_token_count_exchange / _sync_token_count_exchange).
- **What:** ops `aten::cat`, `aten::sum`; 3.0 kernels per instance, 116 instances per step (forward 0.4 ms, recompute 0.4 ms). Shapes: [256] int64 counts -> [64] sums.
- **Numbers:** 7 us measured vs 0 us ideal per instance (external bytes 0.000 GB); gap 0.8 ms (0.00% of step). Evidence: measured.
- **Why fusable (dataflow):** per-expert counts are reduced to per-rank splits right before the host copy.
- **Blockers:** Interleaved with the count all-to-all and a device-to-host copy (host sync).
- **Tier T1:** Integer sums: exact in any order.
- **Notes:** Launch-bound; listed for completeness.
- **How found:** op table small-kernel sweep.

## 5. No opportunity

| what | GPU ms | reason |
| --- | ---: | --- |
| FSDP2 all-gather copy-in / copy-out and reduce-scatter packing (_foreach_copy_, split_with_sizes_copy, chunk_cat, chunk_cat_mixed_dtype, cat) | 150.0 | Owned by FSDP2's flat-buffer protocol around real collectives; the copies remain with NCCL. Fusing them means changing FSDP (all-gather into parameter storage, reduce-scatter from strided grads), not model code. With the fake process group the neighbouring collective is itself a copy, so no fusion can be judged. |
| MoE split host copies (device-to-host sync for all-to-all splits) and int32 offsets copy | 0.5 | Device-to-host copies forced by the variable-size all-to-all; a host sync, not fusable glue. |
| Standalone RMSNorm kernels at roofline: q_norm (fwd/rc/bwd), attention_norm in recompute and layer 0 forward, final norm backward | 56.4 | CuTeDSL RMSNorm kernels run at 1.0-1.3x roofline; their neighbours are GEMMs (q_norm sits between wq_a and wq_b; recompute attention_norm reads the saved block input), so only GEMM prologue/epilogue fusion (T3) remains. |
| Embedding gather (fwd) and embedding_dense_backward | 5.4 | Single ops with no memory-bound neighbour; the backward's 10x gap is the deterministic sort-based algorithm, not missing fusion. |
| Flash-attention auxiliary zero-fills and tiny metadata kernels (cu_seqlens/aux buffers) | 3.0 | Microsecond buffer initializations inside the attention op; would disappear inside the attention kernel (T3). |
| Loss metadata and chunking copies outside the cross-entropy core (chunk_cat of lm_head grad, label splits, misc) | 0.7 | Chunk bookkeeping and FSDP-adjacent gradient packing for the lm_head; a few milliseconds total. |
| Optimizer (fused AdamW, grad-norm, clipping, expert-bias update) | 19.5 | Already fused (_fused_adamw_, _foreach_*); outside the training-step glue under study. |

GEMMs, attention, and fake-collective copies are excluded from regions by definition (section 2). Two GEMM-adjacent ideas are recorded as notes rather than regions because they need a GEMM inside (T3): folding the permute gather into the w13 grouped-GEMM A-operand load, and folding the unpermute scatter and score multiply into the w2 grouped-GEMM epilogue.

## 6. Substrate observations

### What the eager substrate made easy

- **Real per-instance kernel time at real shapes.** Every number in the inventory is a measured kernel duration for one aten op instance (66,303 owner ops, 93,253 GPU events). The trace shows which aten kernel actually ran, and that explains most gaps directly: the combine multiply (`bf16 [786432,7168] * fp32 [786432,1]`) runs the generic non-vectorized `gpu_kernel_impl<BinaryFunctor<float,...>>` at 1.6 TB/s; `deterministic=True` routes every `index_put_` (even a pure permutation) through the sort-based `indexing_backward_kernel` (10-19 ms per call); `topk` on 32- and 256-wide rows uses bitonic sort plus `gatherTopK` (1.6-2.1 ms for 100 MB of input); SwiGLU on the strided halves of `[R,2,F]` runs the strided `elementwise_kernel<128,4>` at about 30% of roofline.
- **Source attribution.** `with_stack=True` python frames give an exact `file:line` for every op dispatched through a Python-visible call (`torch.cat`, `F.silu`, `x[idx]`), and the module hooks give FQN and layer index. 99.8% of forward/recompute GPU time has a module FQN.
- **Backward provenance.** Each backward op sits under an `autograd::engine::evaluate_function: <Node>` event whose `Sequence number` matches the forward op that created the node; 98.4% of backward GPU time is linked to a forward op, and the backward row inherits the forward op's FQN and source line.
- **Remat behaviour is visible as it happens.** The recompute phase is a separate, timed stream of kernels (10.5 s here), so one can see exactly what torch_remat replays: everything in the block except the saved regions (routing decision, aux-loss accumulation, `shared_add`, `ffn_residual`). That exposed a non-fusion finding: the MoE combine (49 ms/layer, 2.85 s per step) is replayed in recompute only because its multiply and scatter-add save their inputs; its outputs are dead there.
- **Fake collectives are easy to separate.** Their copies sit under `c10d::alltoall_` / `c10d::allgather_` CPU ops (1.76 s). The same view showed that the concat after each uneven all-to-all is real code (`spmd_types._AllToAllUneven` builds a list of chunks and `torch.cat`s them, 3.3 ms per call, 4 calls per MoE layer), not a fake-PG artifact.
- **Code spans map to override points.** Each region has an owning `Configurable` and method (token dispatcher `_score_and_scatter_add`, `_permute`, `SwiGLU._activation_fn`, `DeepSeekV3Router._select_experts`, ...), which is what an eager kernel would replace.

### What was hard, and the workarounds

- **The execution trace crashes under torch_remat.** The observer reads every tensor's storage `data_ptr`; torch_remat placeholders (outputs of saved regions skipped during recompute, here `shared_add` and `ffn_residual`) poison that pointer. The observer logs an exception per op, its internal op stack desynchronizes, and backward later dies with `RuntimeError: std::get: wrong index for variant` (reproduced on the debug model; Kineto alone is fine). Workaround: the recipe patches `torch_remat._region._CheckpointPhaseContext` to pause the observer during each RECOMPUTE phase. Consequence: recompute ops have Kineto timing, shapes and source but **no tensor IDs**; their bytes and dataflow come from the matching forward op (same layer, module FQN, op, input shapes, ordinal: 10,823 of 10,823 matched).
- **Tensor and storage IDs are pointer-keyed, with no free events.** A freed address that is reused gets the same ID, and recompute writes are invisible, so "last writer of this storage" can name a stale producer (seen as spurious cross-layer edges such as "layer L concat -> layer L+1 index_put" from `new_empty` buffers reusing freed addresses, and multi-terabyte "saved tensor" edges before strides were used for expanded index tensors). Mitigation in `dataflow.py`: backward reads are classified as intra-node, incoming gradient (storage listed in the inputs of the backward node record), or saved tensor of the forward op (matched by shape/dtype to that op's execution-trace inputs/outputs and mapped to its recompute replay). 31.5k backward edges still could not be verified. Region external bytes were therefore derived from the traced shapes and the source, not from the automatic dataflow.
- **Sequence numbers live on autograd-level ops only.** Inner kernel ops carry `seq_id = -1`, many non-differentiable ops share the peeked number, and custom `autograd.Function`s record theirs on the Function's record. The creator of node S is taken as the last forward op with sequence number S outside recompute windows.
- **Backward runs on the forward thread** in this build (same `tid`, `fw_tid` of the node record = 1), so thread IDs cannot separate phases; `evaluate_function` ancestry and the recompute markers do.
- **Python frames stop at functions.** An op dispatched by an operator inside a function body (`F.silu(gate) * up`, `a + b`) has no frame of its own, so its source is the enclosing function (`activation.py:175+ (_activation_fn)`), not a line.
- **The chunked loss runs each chunk's backward inside the forward call**, so "first autograd node" is not the start of the model backward; the phase window had to exclude nodes launched from `components/loss.py`.
- **The CPU runs ahead of the GPU**, so the active window opens while step 9's last 0.6 s of kernels still run; those have no owner op in the trace (the 1.44% unattributed).
- **Volume.** The Kineto trace has 3.15M events (2.1M python-function events, 72 MB gzip); the execution trace has 505k nodes (579 MB). Joining both takes about a minute and 9.5 GB of RAM.

### How recompute and backward provenance were identified

1. **Recompute:** (a) `## remat_recompute layers.N ##` record_function opened by the patched torch_remat RECOMPUTE phase context (exact window per block, 61 windows); (b) the module pre-hook labels every module call made while `torch_remat.is_recomputing()` as `rc`. Ops inside (a) are phase `recompute`; ops of the block body outside any submodule get FQN `layers.N` from (a).
2. **Backward:** any op under an `evaluate_function` event and not under (a). Its forward op is the one with the node's `Sequence number` (see above); it inherits FQN, layer and source from it.
3. **Recompute-to-backward edges:** a backward read that is neither intra-node nor an incoming gradient is a saved tensor of the forward op; its forward producer is mapped to that producer's recompute replay (`rc_to_bwd` edges), which is how the three SwiGLU recompute-backward regions were found.

### False leads discarded

- Automatic connected components of memory-bound ops (`analysis/components.json`, `patterns.json`) merged router, dispatcher and combine of a MoE layer into 50-100-op components, and data-dependent shapes split the same code into many patterns. Useful for discovery only; regions were cut by code span instead.
- Spurious cross-layer and multi-terabyte edges from address reuse and stride-0 index tensors (fixed with strides; the rest left as unverified).
- The post-all-to-all `aten::cat` first looked like fake-PG overhead; it is real `spmd_types` code.
- Loss chunks 2-8 first landed in an `other` phase (chunked loss runs backward inside forward).
- GEMM measured/ideal < 1 at first suggested a broken flop model; it is data-dependent GEMM power on GB300 (calibration data matters).
- The first probe on the debug model failed with a device-side assert: the 671B recipe's tokenizer does not fit the debug model's vocabulary (probe-only issue).

### Opportunity types this substrate could miss

- **Recompute-internal dataflow.** Without tensor IDs in recompute, fusions that only exist inside the replay (or buffer reuse between replay and backward beyond the SwiGLU case) are inferred, not observed.
- **Fusions across autograd nodes that eager never co-schedules**, for example reordering a GEMM's dgrad and wgrad around an activation kernel; eager shows the nodes but not alternative orders.
- **GEMM prologue/epilogue and attention-kernel fusions (T3)** are only noted; the eager trace gives no cost model for them.
- **Communication-adjacent fusions under real EP/FSDP**: fake collectives are copies, so overlap and the true cost of the all-to-all neighbours are unknown, and R = 786,432 routed rows per layer is the fake-PG value (balanced by construction).
- **Inside custom kernels** (CuTeDSL RMSNorm, flash attention): treated as opaque.
- **The production configuration**: `local_compile` regions are off here, so eager glue that Inductor already fuses in the default recipe is counted (by design, to share aten semantics with the graph substrate).

### Effort

- Wall time: about 51 min (start 10:56, end 11:47). The 671B capture took 8.2 min (init 18 s, step 1 61 s, then ~40 s per step; the profiled step 42 s); the probe and calibration runs about 5 min.
- Scripts written (all under `outputs/fusion_finder/eager/`): `recipe/eager_fusion_recipe.py` (derived recipe, capture profiler, FQN hooks, remat markers), `run_capture.sh`, `scripts/calibrate_roofline.py`, `scripts/build_op_table.py` (Kineto + ET join, per-op roofline), `scripts/dataflow.py` (producer/consumer edges, components), `scripts/dump_layer.py` (per-layer op listing), `scripts/regions.py` (region definitions, coverage check, `regions.json`), `scripts/validate_regions.py`, `scripts/make_report.py`.

## 7. Reproduction

All paths are relative to the repo root `/home/bahuang/local/torchtitan`; everything lives under `outputs/fusion_finder/eager/` (gitignored).

| step | command | output |
| --- | --- | --- |
| capture (GPU 0, ~9 min) | `CUDA_VISIBLE_DEVICES=0 outputs/fusion_finder/eager/run_capture.sh` | `capture/traces/iteration_10/rank0_trace.json.gz` (Kineto), `capture/traces/rank0_execution_trace.json` (ET), `capture/train.log`, `capture/config_deltas.json` |
| probe (optional, GPU 1, ~1 min) | `CUDA_VISIBLE_DEVICES=1 NGPU=8 FUSION_FINDER_FLAVOR=debugmodel FUSION_FINDER_DP_SHARD=8 FUSION_FINDER_EP=4 FUSION_FINDER_OUT=$PWD/outputs/fusion_finder/eager/probe outputs/fusion_finder/eager/run_capture.sh` | `probe/` (debug model, same capture path) |
| roofline (GPU 0, ~1 min) | `CUDA_VISIBLE_DEVICES=0 python outputs/fusion_finder/eager/scripts/calibrate_roofline.py --out outputs/fusion_finder/eager/calib/calib_gpu0.json --iters 30` | `calib/calib_gpu0.json` |
| op table (CPU, ~70 s, ~10 GB RAM) | `cd outputs/fusion_finder/eager && python scripts/build_op_table.py --kineto capture/traces/iteration_10/rank0_trace.json.gz --et capture/traces/rank0_execution_trace.json --calib calib/calib_gpu0.json --out-dir analysis` | `analysis/op_table.parquet` (+ `.csv.gz`), `op_table_summary.json`, `et_index.pkl`, `annotations.json`, `unowned_gpu_events.json` |
| dataflow (CPU, ~15 s) | `python scripts/dataflow.py --analysis analysis --calib calib/calib_gpu0.json` | `analysis/edges.parquet`, `components.json`, `patterns.json`, `cross_edges.json` |
| regions | `python scripts/regions.py --analysis analysis --calib calib/calib_gpu0.json --setup analysis/setup.json --out regions.json && python scripts/validate_regions.py regions.json` | `regions.json`, `analysis/regions_side.json`; fails if any memory-bound row is unclaimed |
| report | `python scripts/make_report.py` | `report.md` |

Helpers: `python scripts/dump_layer.py analysis/op_table.parquet <layer> <forward|recompute|backward>` lists one layer's ops (e.g. `analysis/layer10_forward_ops.txt`, `analysis/layer10_backward_ops.txt`); `analysis/mb_classes.txt` is the memory-bound kernel-class inventory used for the coverage check. `analysis/setup.json` holds the setup block of `regions.json`.

Notes: the recipe imports only public torchtitan APIs plus a monkeypatch of the private `torch_remat._region._CheckpointPhaseContext` (instrumentation only; the probe's step-10 loss is identical with and without it). The capture recipe reads optional `FUSION_FINDER_*` environment variables only for the probe; the 671B run uses none of them.
