# Graph fusion-region finder: DeepSeek-V3 671B, one GraphTrainer training step

Substrate: **graph** (GraphTrainer traced joint fwd+bwd FX graph after all passes, plus a Kineto trace of the same graph executed with per-node `_RecordFunctionFast` labels). All numbers are from one profiled step (`ProfilerStep#9`, the 10th step) on GPU 2.

## 1. Setup

- Commit: `c5c112d82e8d369a76368a62282b44718f1a42db` (working tree has only untracked files; no tracked file was modified).
- Command: `CUDA_VISIBLE_DEVICES=2 outputs/fusion_finder/graph/scripts/profile_dsv3_fake_graph.sh  (derived copy of scripts/profile_dsv3_fake.sh, TRAINER=graph; runs: python -m torchtitan.train --module profile_dsv3_fake_recipe --config profile_config --comm-backend fake)`
- Config deltas from the recipe:
  - `base`: torchtitan_recipes.models.deepseek_v3.deepseek_v3_671b(seq_len=4096) -> to_graph_trainer_config(..., GraphTrainerDeepSeekV3Model.Config)
  - `compile`: GraphTrainerCompileConfig(memory_policy='full', pass_pipeline='fusion_finder')
  - `compile.pass_pipeline`: fusion_finder = default GraphTrainer passes + recompute_fresh_buffer_index_put_pass (after tag_with_memory_policy_pass) + graph dumps + custom_codegen_pass (last)
  - `parallelism`: data_parallel_shard_degree=256, expert_parallel_degree=64, NGPU=256 fake PG
  - `training.num_tokens_per_microbatch_per_dp_rank`: 98304
  - `training.steps`: 10
  - `debug`: seed=42, deterministic=True
  - `profiler`: enable_profiling=True, profile_freq=10, warmup=3, active=1, enable_memory_snapshot=False
  - `verified`: training.disable_accelerator_graphs=True, model.local_compile.regions=[], no gradient accumulation (num_tokens_per_train_step=-1)
- Temporary `graph_trainer` edits: **none**. Instrumentation is a pass pipeline registered from `scripts/fusion_finder_passes.py` (imported by the generated recipe) through the existing `PASS_PIPELINE_REGISTRY` / `compile.pass_pipeline` hook.
- Deviation needed to run the fixed setup (reported, not hidden): with HEAD, `memory_policy='full'` OOMs on one GB300 in step 1 (274.6 GiB allocated at layer 22). `tag_sac_policy` Pass 1b forces every in-place mutation and its direct consumers to MUST_SAVE; in the MoE combine path that is `new_empty -> index_put_ -> all_to_all_single` and the saved combine all_to_all output is [786432,7168] bf16 = 10.5 GiB per MoE layer (~610 GiB total). `recompute_fresh_buffer_index_put_pass` (in my pipeline, after `tag_with_memory_policy_pass`) retags these 58 chains (`capture/retagged_index_put_nodes.json`) to MUST_RECOMPUTE, which matches eager full AC (the whole block is recomputed). Effect on the graph: the recompute region gains one w2 grouped GEMM + `index_put_` + combine all_to_all per MoE layer. Peak memory 235 GiB. First attempt (dump bug) and second attempt (this OOM) are kept under `failed_attempts/`.
- GPU: NVIDIA GB300 (GPU 2 via CUDA_VISIBLE_DEVICES=2). During calibration another process held 17.7 GB at 0% utilization; during the capture (11:13-11:26) GPU 2 had only this process (nvidia-smi checked at step 1 and mid-run), GPUs 0/1 were used intermittently by the sibling. `gpu_busy_during_timing=false`.
- Roofline (measured on GPU 2, `scripts/calibrate_roofline.py`, before and after the capture):
  - HBM device-to-device copy (8 GiB): 6876 GB/s before, 6908 GB/s after. **Used: 6876.3 GB/s.**
  - Read-only reduction: fp32 amax 4 GiB 6364 GB/s, bf16 sum 8 GiB 6112 GB/s. Pointwise read+write (bf16 mul) 6995 GB/s.
  - bf16 GEMM, random operands: dense_ffn_w1 1727; dense_ffn_w2 1781; attn_wq_b 1847; attn_wo 1801; attn_wq_a 1662; shared_expert_w1 1874; square 1665 TFLOP/s.
  - bf16 GEMM at the attention.wo shape [98304,16384]x[16384,7168] by operand values: randn_x_randn 1811; randn_x_0.02randn 1796; 0.1randn_x_0.006randn 1837; uniform01_x_uniform01 1927; half_zeros_x_randn 1767; zeros_x_zeros 2360 TFLOP/s. The profiled step runs this GEMM at 2.34 PFLOP/s, i.e. in the near-zero-operand power regime (the fake process group gives weights/activations of very low entropy). **Used: 2360.2 TFLOP/s** so the ideal stays a lower bound; with the random-operand 1874 TFLOP/s every large GEMM would look faster than peak.
- Ideal time = max(bytes / HBM, flops / peak). Region bytes = external inputs read once + external outputs written once (views resolved to the bytes actually consumed; stride-0 broadcasts counted once).

## 2. Step budget

Profiled step GPU window 38043.4 ms; GPU busy 37142.3 ms; GPU idle 901.1 ms.

| category | ms | % of busy |
|---|---:|---:|
| gemm | 16463.5 | 44.33 |
| attention | 3118.5 | 8.40 |
| fake_comm | 2050.1 | 5.52 |
| memory_bound_glue | 15487.1 | 41.70 |
| optimizer | 16.8 | 0.05 |
| other | 11.0 | 0.03 |

- Attribution: 99.93% of kernel time maps to an FX node through its `_RecordFunctionFast` label; with the eager optimizer/clip kernels attributed by their enclosing CPU op the coverage is 99.984% of kernel time (99.996% of GPU-busy time). Only 6.1 ms (11 activities) had no CPU launch event.
- `fake_comm` = kernels launched by `_c10d_functional` nodes under the fake backend (DtoD memcpy + memset for all_to_all_single, ~134 small memcpys per all_gather_into_tensor_out, memcpy for reduce_scatter). Reported, excluded from compute.
- `memory_bound_glue` = every other FX node with kernels (includes memsets/memcpys launched by glue ops, e.g. the deterministic index_put path). `attention` = `torch_attn._varlen_attn(_backward)`.
- GPU idle breakdown (gaps between consecutive kernels, labelled by the previous and next kernel's FX target, or the enclosing eager CPU op outside the graph):
  - 278.7 ms over 1 gaps: `?` -> `aten::to`
  - 275.1 ms over 58 gaps: `aten._to_copy.default` -> `_c10d_functional.all_to_all_single.default`
  - 104.5 ms over 31982 gaps: `_c10d_functional.all_gather_into_tensor_out.default` -> `_c10d_functional.all_gather_into_tensor_out.default`
  - 39.8 ms over 602 gaps: `Optimizer.step#OptimizersContainer.step` -> `Optimizer.step#OptimizersContainer.step`
  - 32.7 ms over 1 gaps: `aten::item` -> `bucketing._pre_bucket_all_gather.default`
  - 25.5 ms over 237 gaps: `_c10d_functional.all_gather_into_tensor_out.default` -> `aten.split_with_sizes_copy.out`
  - 9.7 ms over 1 gaps: `_c10d_functional::all_reduce` -> `aten::to`
  - The 278.7 ms gap before the first `aten::to` is the start of the step (CPU-side data loading and input preprocessing before the input H2D copy); the all_gather -> all_gather gaps are between the fake collective's ~134 small memcpys.
  - The dominant idle is the host sync before each MoE token-count exchange: `_to_copy` (D2H of the 64x4 split sizes) then 256 CPU-side `_local_scalar_dense` reads per MoE layer (14848 `.item()` nodes in the graph) before the dispatch all_to_all can be launched. Not a fusion opportunity, but visible in the graph as a structural sync.

## 3. Ranked regions

146 regions, ranked by `gap_ms_total`. Scope tags: cross-module: 62, intra-module: 76, cross-layer: 8, recompute-backward: 14. Tiers: T0: 33, T1: 30, T2: 22, T3: 61.
Regions overlap by construction: L1 (intra-module) components partition the glue; cross-module (XM-max / XM), cross-layer (XL), recompute-backward (RB) and GEMM-adjacent (T3) regions are unions of L1 parts (plus a GEMM) and list the overlapped regions. `incr` = incremental gap over the parts = bytes of intermediates that stop being materialized when the boundary is fused (at HBM roofline). For T3 regions the gap includes the GEMM's own distance from the flops roofline; the fusion-attributable part is in the notes.

Non-overlapping totals (greedy by gap): all regions 17058 ms over 69 regions (includes GEMM inefficiency inside T3 regions); without T3 regions 11119 ms over 45 regions = 29.2% of the step.
Largest non-overlapping hot spots: G-001 2902 ms; G-004 2512 ms; G-006 2337 ms; G-009 1026 ms; G-010 987 ms; G-011 972 ms; G-012 861 ms; G-013 848 ms.

| id | title | phases | scope | kind | inst | measured ms | ideal ms | gap ms | % step | incr ms | tier |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| G-001 | MoE combine backward + router/aux-loss score backward (one stage, per layer) | backward | cross-module | gather/scatter + pointwise | 58 | 3107.6 | 205.3 | 2902.4 | 7.63 | 3.4 | T2 |
| G-002 | MoE combine backward: gather grad + fp32 upcast + score mul/sum + cast (routed_experts) | backward | intra-module | gather/scatter + pointwise | 58 | 3026.7 | 202.2 | 2824.4 | 7.42 |  | T2 |
| G-003 | Cross-layer: next layer's attention_norm backward + residual-grad add fused into combine-backward gather | backward | cross-layer | gather/scatter + pointwise | 57 | 3046.9 | 245.5 | 2801.4 | 7.36 | 11.7 | T2 |
| G-004 | MoE combine forward + routed/shared expert sum (score mul, cast, deterministic scatter-add, add) | forward | cross-module | gather/scatter + pointwise | 58 | 2643.2 | 130.8 | 2512.4 | 6.60 | 23.8 | T2 |
| G-005 | MoE combine forward: fp32 score mul + cast + zeros + deterministic_scatter_add (routed_experts) | forward | intra-module | gather/scatter + pointwise | 58 | 2609.2 | 119.0 | 2490.2 | 6.55 |  | T2 |
| G-006 | Backward un-permute: zeros + index_put(accumulate=True) of routed-token grads (dispatch backward) | backward | intra-module | gather/scatter + pointwise | 116 | 2634.1 | 297.3 | 2336.8 | 6.14 |  | T2 |
| G-007 | T3: w13 dgrad grouped GEMM epilogue writes un-permuted rows (replace zeros + index_put) | backward | cross-module | GEMM epilogue | 58 | 3298.4 | 1134.8 | 2163.6 | 5.69 | 190.2 | T3 |
| G-008 | Token-grad accumulation index_put + add of shared-expert dgrad (backward) | backward | cross-module | gather/scatter + pointwise | 58 | 1159.4 | 118.9 | 1040.5 | 2.73 | 23.8 | T2 |
| G-009 | T3: w2 dgrad grouped GEMM + SwiGLU backward epilogue (routed experts) | backward | cross-module | GEMM epilogue | 58 | 1593.6 | 567.4 | 1026.2 | 2.70 | 163.0 | T3 |
| G-010 | T3: w2 grouped GEMM epilogue stores rows un-permuted (replace new_empty + index_put_), forward | forward | cross-module | GEMM epilogue | 58 | 1554.8 | 567.4 | 987.4 | 2.60 | 190.2 | T3 |
| G-011 | T3: w2 grouped GEMM + un-permute index_put_ (recompute copy) | recompute | cross-module | GEMM epilogue | 58 | 1539.8 | 567.4 | 972.4 | 2.56 | 190.2 | T3 |
| G-012 | T3: post-all_to_all token permutation gather fused into w13 grouped GEMM prologue (recompute) | recompute | cross-module | GEMM prologue | 58 | 1995.7 | 1134.8 | 860.9 | 2.26 | 190.2 | T3 |
| G-013 | T3: post-all_to_all token permutation gather fused into w13 grouped GEMM prologue (forward) | forward | cross-module | GEMM prologue | 58 | 1983.2 | 1134.8 | 848.4 | 2.23 | 190.2 | T3 |
| G-014 | T3: SwiGLU (silu*mul) fused into w2 grouped GEMM prologue (forward) | forward | cross-module | GEMM prologue | 58 | 1365.8 | 567.4 | 798.4 | 2.10 | 81.5 | T3 |
| G-015 | Routed-expert SwiGLU backward: mul, mul, silu_backward, stack -> one kernel | backward | intra-module | pointwise chain | 58 | 729.3 | 163.0 | 566.3 | 1.49 |  | T0 |
| G-016 | T3: backward permutation gather of incoming expert grads fused into the w2 dgrad/wgrad grouped GEMM prologue | backward | cross-module | GEMM prologue | 58 | 1120.0 | 567.4 | 552.6 | 1.45 | 190.2 | T3 |
| G-017 | T3: w13 wgrad grouped GEMM epilogue emits fp32 directly into the reduce-scatter bucket | backward | intra-module | GEMM epilogue | 58 | 1615.9 | 1134.8 | 481.1 | 1.26 | 9.9 | T3 |
| G-018 | MoE router forward: sigmoid/bias/group-limited top-k/normalize + aux-loss stats + dispatch sort | forward | cross-module | routing selection (top-k/sort) + pointwise | 57 | 550.3 | 106.3 | 444.1 | 1.17 | 2.5 | T2 |
| G-019 | Routed-expert SwiGLU forward: silu + mul on strided w13 halves (1 kernel instead of 2) | forward | intra-module | pointwise chain | 58 | 479.1 | 81.5 | 397.5 | 1.04 |  | T0 |
| G-020 | Routed-expert SwiGLU forward, recompute copy: silu + mul | recompute | intra-module | pointwise chain | 58 | 495.5 | 108.7 | 386.9 | 1.02 |  | T0 |
| G-021 | MoE router + dispatch prologue, recompute copy (sigmoid/topk/normalize/sort/gather) | recompute | cross-module | routing selection (top-k/sort) + pointwise | 58 | 413.2 | 109.2 | 304.0 | 0.80 | 1.0 | T2 |
| G-022 | T3: router gate GEMM (mm.dtype fp32 out) + router top-k epilogue (forward) | forward | cross-module | GEMM epilogue | 58 | 272.0 | 13.3 | 258.7 | 0.68 | 0.0 | T3 |
| G-023 | MoE router selection: sigmoid + bias + 3x topk + masks + gather + normalize (forward) | forward | intra-module | routing selection (top-k/sort) + pointwise | 58 | 257.3 | 2.2 | 255.0 | 0.67 |  | T2 |
| G-024 | T3: w2 wgrad grouped GEMM epilogue emits fp32 (drop _to_copy) | backward | intra-module | GEMM epilogue | 58 | 820.4 | 567.4 | 253.0 | 0.67 | 3.0 | T3 |
| G-025 | Q RoPE backward + grad concat: fp32 casts, conj-mul, cast, cat (backward) | backward | cross-module | RoPE + cast/concat | 61 | 328.7 | 86.0 | 242.7 | 0.64 | 28.6 | T0 |
| G-026 | Q RoPE + Q concat, recompute copy (fp32 cast, complex mul, cast, cat) | recompute | cross-module | RoPE + cast/concat | 61 | 327.3 | 86.0 | 241.4 | 0.63 | 28.6 | T0 |
| G-027 | Q RoPE + Q concat (forward): fp32 cast, complex mul, bf16 cast, cat(q_nope, q_pe) | forward | cross-module | RoPE + cast/concat | 61 | 326.5 | 86.0 | 240.6 | 0.63 | 28.6 | T0 |
| G-028 | RoPE frequency gather (recompute) + RoPE backward (q and k paths) | backward/recompute | intra-module, recompute-backward | RoPE + cast/concat | 122 | 245.3 | 29.3 | 216.1 | 0.57 | 0.4 | T0 |
| G-029 | RoPE backward (q_pe and k_pe): fp32 cast + conj-mul + cast (backward) | backward | intra-module | RoPE + cast/concat | 122 | 243.2 | 29.2 | 213.9 | 0.56 |  | T0 |
| G-030 | RoPE (q_pe and k_pe), recompute copy: fp32 cast + complex mul + cast | recompute | intra-module | RoPE + cast/concat | 122 | 242.0 | 29.2 | 212.7 | 0.56 |  | T0 |
| G-031 | RoPE (q_pe and k_pe) forward: fp32 cast + complex mul + cast | forward | intra-module | RoPE + cast/concat | 122 | 241.3 | 29.2 | 212.0 | 0.56 |  | T0 |
| G-032 | MoE dispatch, recompute copy: stable sort + floor_divide + routed-score and token gathers | recompute | intra-module | routing selection (top-k/sort) + pointwise | 58 | 271.8 | 107.2 | 164.6 | 0.43 |  | T2 |
| G-033 | MoE dispatch: stable sort of expert ids + floor_divide + routed-score gather + token gather [786432,7168] | forward | intra-module | routing selection (top-k/sort) + pointwise | 58 | 264.7 | 107.1 | 157.6 | 0.41 |  | T2 |
| G-034 | Router + aux-loss recompute fused into their backward (score grads) | backward/recompute | cross-module, recompute-backward | routing selection (top-k/sort) + pointwise | 58 | 146.1 | 2.7 | 143.4 | 0.38 | 0.8 | T2 |
| G-035 | Router recompute (sigmoid/topk/normalize) fused into router backward | backward/recompute | intra-module, recompute-backward | gather/scatter + pointwise | 58 | 140.7 | 3.5 | 137.1 | 0.36 | 1.0 | T2 |
| G-036 | T3: wq_b GEMM epilogue writes q directly (RoPE + cat into the GEMM output layout) | forward | cross-module | GEMM epilogue | 61 | 313.8 | 191.8 | 122.0 | 0.32 | 85.7 | T3 |
| G-037 | T3: wq_b GEMM epilogue writes q directly, recompute copy | recompute | cross-module | GEMM epilogue | 61 | 312.6 | 191.8 | 120.8 | 0.32 | 85.7 | T3 |
| G-038 | T3: q-grad concat (dq_nope, dq_pe) folded into the wq_b dgrad GEMM prologue (backward) | backward | cross-module | GEMM prologue | 61 | 312.1 | 191.8 | 120.3 | 0.32 | 85.7 | T3 |
| G-039 | T3: attention.wo wgrad GEMM epilogue emits fp32 (drop _to_copy) | backward | intra-module | GEMM epilogue | 61 | 714.8 | 596.8 | 118.0 | 0.31 | 6.3 | T3 |
| G-040 | T3: router gate GEMM + router top-k epilogue, recompute copy | recompute | cross-module | GEMM epilogue | 58 | 127.4 | 12.9 | 114.5 | 0.30 | 0.0 | T3 |
| G-041 | MoE router selection, recompute copy (sigmoid, bias, topk, gather, normalize) | recompute | intra-module | routing selection (top-k/sort) + pointwise | 58 | 112.7 | 1.8 | 110.8 | 0.29 |  | T2 |
| G-042 | T3: kv-grad concat folded into the wkv_b dgrad GEMM prologue (backward) | backward | cross-module | GEMM prologue | 61 | 223.1 | 115.5 | 107.6 | 0.28 | 84.1 | T3 |
| G-043 | T3: shared-expert w2 dgrad GEMM + SwiGLU backward epilogue | backward | cross-module | GEMM epilogue | 58 | 169.9 | 70.9 | 99.0 | 0.26 | 20.4 | T3 |
| G-044 | T3: wkv_b GEMM epilogue writes K = cat(k_nope, broadcast k_pe) directly | forward | cross-module | GEMM epilogue | 61 | 187.5 | 101.3 | 86.2 | 0.23 | 55.5 | T3 |
| G-045 | T3: wkv_b GEMM epilogue writes K directly, recompute copy | recompute | cross-module | GEMM epilogue | 61 | 187.2 | 101.3 | 85.9 | 0.23 | 55.5 | T3 |
| G-046 | T3: attention.wo GEMM epilogue adds the residual, recompute copy | recompute | cross-module | GEMM epilogue | 61 | 676.3 | 596.8 | 79.6 | 0.21 | 37.5 | T3 |
| G-047 | T3: attention.wo GEMM epilogue adds the residual (forward) | forward | cross-module | GEMM epilogue | 61 | 670.9 | 596.8 | 74.1 | 0.20 | 37.5 | T3 |
| G-048 | T3: shared-expert SwiGLU fused into w2 GEMM prologue | forward | cross-module | GEMM prologue | 58 | 137.7 | 70.9 | 66.7 | 0.17 | 10.2 | T3 |
| G-049 | T3: shared-expert w13 dgrad GEMM accumulates into the token grad (epilogue add) | backward | intra-module | GEMM epilogue | 58 | 206.1 | 141.9 | 64.3 | 0.17 | 35.7 | T3 |
| G-050 | Shared-expert SwiGLU backward: mul, mul, silu_backward, stack -> one kernel | backward | intra-module | pointwise chain | 58 | 84.0 | 20.4 | 63.7 | 0.17 |  | T0 |
| G-051 | T3: router gate hi/mid/lo bf16 split (casts, shifts, subs) fused into the gate GEMM prologue (backward) | backward | intra-module | GEMM prologue | 174 | 102.4 | 42.2 | 60.2 | 0.16 | 0.0 | T3 |
| G-052 | Aux-loss statistics recompute fused into aux-loss backward | backward/recompute | intra-module, recompute-backward | reduction + pointwise | 58 | 62.1 | 1.9 | 60.2 | 0.16 | 0.9 | T1 |
| G-053 | T3: chunked loss backward (nll/log_softmax backward + cast) fused into lm_head dgrad GEMM prologue | backward | cross-module | GEMM prologue | 8 | 135.9 | 77.2 | 58.7 | 0.15 | 18.5 | T3 |
| G-054 | Backward residual-grad adds around ffn_norm: add(dgrads) + RMSNorm backward + add (cross-module) | backward | cross-module | RMSNorm + pointwise | 58 | 139.7 | 82.8 | 56.9 | 0.15 | 47.5 | T1 |
| G-055 | T3: dense FFN (layers 0-2) w2 dgrad GEMM + SwiGLU backward epilogue | backward | cross-module | GEMM epilogue | 3 | 83.9 | 33.0 | 50.8 | 0.13 | 9.5 | T3 |
| G-056 | T3: wq_a/wkv_a dgrad GEMM accumulates into the token grad (epilogue add) | backward | cross-module | GEMM epilogue | 61 | 103.1 | 55.9 | 47.1 | 0.12 | 37.5 | T3 |
| G-057 | Cross-layer: final norm backward fused into last layer's combine-backward gather | backward | cross-layer | gather/scatter + pointwise | 1 | 50.8 | 3.9 | 46.9 | 0.12 | 0.2 | T2 |
| G-058 | T3: shared-expert w2 GEMM epilogue adds the routed-expert output | forward | cross-module | GEMM epilogue | 58 | 116.6 | 70.9 | 45.6 | 0.12 | 35.7 | T3 |
| G-059 | Shared-expert SwiGLU forward: silu + mul | forward | intra-module | pointwise chain | 58 | 55.1 | 10.2 | 45.0 | 0.12 |  | T0 |
| G-060 | T3: wq_b dgrad GEMM + q_norm RMSNorm backward epilogue | backward | cross-module | GEMM epilogue | 61 | 235.3 | 191.8 | 43.5 | 0.11 | 8.0 | T3 |
| G-061 | Shared-expert SwiGLU forward, recompute copy | recompute | intra-module | pointwise chain | 58 | 56.3 | 13.6 | 42.8 | 0.11 |  | T0 |
| G-062 | T3: wq_a wgrad GEMM epilogue emits fp32 into the reduce-scatter bucket | backward | intra-module | GEMM epilogue | 61 | 98.0 | 56.0 | 42.0 | 0.11 | 17.6 | T3 |
| G-063 | T3: chunked loss: lm_head GEMM + fp32 upcast + log_softmax + nll (fused linear cross-entropy) | forward | cross-module | GEMM epilogue | 1 | 52.8 | 11.1 | 41.7 | 0.11 | 9.8 | T3 |
| G-064 | T3: q_norm RMSNorm fused into the wq_b GEMM prologue (forward) | forward | cross-module | GEMM prologue | 61 | 231.9 | 191.8 | 40.0 | 0.10 | 5.4 | T3 |
| G-065 | T3: router-gate dgrad GEMM accumulates into the token grad (epilogue add) | backward | intra-module | GEMM epilogue | 58 | 64.6 | 26.6 | 38.0 | 0.10 | 35.7 | T3 |
| G-066 | T3: wkv_a dgrad GEMM accumulates into the token grad (epilogue add) | backward | intra-module | GEMM epilogue | 61 | 63.7 | 26.1 | 37.6 | 0.10 | 32.4 | T3 |
| G-067 | Attention/shared-expert weight-grad casts, shard re-cat and bucket pack (backward) | backward | cross-module | dtype cast + FSDP bucket pack | 58 | 49.4 | 12.1 | 37.3 | 0.10 | 14.9 | T0 |
| G-068 | Backward: dgrad add + attention_norm RMSNorm backward + residual-grad add | backward | cross-module | RMSNorm + pointwise | 61 | 111.2 | 75.0 | 36.2 | 0.10 | 25.0 | T1 |
| G-069 | T3: attention_norm RMSNorm fused into the wq_a GEMM prologue (forward) | forward | cross-module | GEMM prologue | 61 | 91.6 | 55.9 | 35.6 | 0.09 | 25.0 | T3 |
| G-070 | T3 GEMM epilogue (backward): feed_forward.shared_experts.w13: mm + _to_copy + cat | backward | intra-module | GEMM epilogue | 58 | 177.2 | 141.9 | 35.3 | 0.09 | 1.5 | T3 |
| G-071 | T3 GEMM prologue (forward): feed_forward, feed_forward.w2: silu + mul + mm | forward | cross-module | GEMM prologue | 3 | 68.0 | 33.0 | 34.9 | 0.09 | 4.7 | T3 |
| G-072 | T3 GEMM prologue (forward): attention.inner_attention.wkv_b, attention.kv_norm: _fused_rms_norm + mm | forward | cross-module | GEMM prologue | 61 | 118.6 | 85.3 | 33.4 | 0.09 | 1.8 | T3 |
| G-073 | Intra-module (backward): feed_forward: mul + silu_backward + stack | backward | intra-module | pointwise chain | 3 | 42.4 | 9.5 | 32.9 | 0.09 |  | T0 |
| G-074 | T3: attention_norm RMSNorm fused into the wkv_a GEMM prologue (forward) | forward | cross-module | GEMM prologue | 61 | 57.9 | 26.1 | 31.8 | 0.08 | 19.9 | T3 |
| G-075 | Aux-loss backward: ~16 tiny kernels on [256] / [98304,256] | backward | intra-module | reduction + pointwise | 58 | 33.4 | 1.7 | 31.7 | 0.08 |  | T1 |
| G-076 | T3 GEMM prologue (backward/recompute): feed_forward.shared_experts.w13: cat + mm | backward/recompute | intra-module, recompute-backward | GEMM prologue | 58 | 173.4 | 141.9 | 31.5 | 0.08 | 1.0 | T3 |
| G-077 | Chunked loss backward: nll_loss_backward + log_softmax_backward + bf16 cast | backward | intra-module | cross-entropy (log_softmax + nll) | 8 | 49.5 | 18.5 | 31.0 | 0.08 |  | T1 |
| G-078 | T3 GEMM epilogue (backward): attention.inner_attention.wkv_b, attention.kv_norm: mm + _fused_rms_norm_backward | backward | cross-module | GEMM epilogue | 61 | 116.0 | 85.3 | 30.7 | 0.08 | 2.7 | T3 |
| G-079 | Chunked loss forward: fp32 upcast + log_softmax + nll + scaling (8 chunks, one stage) | forward | intra-module | cross-entropy (log_softmax + nll) | 1 | 41.8 | 11.3 | 30.5 | 0.08 |  | T1 |
| G-080 | Routed-expert weight-grad fp32 casts + reduce-scatter bucket pack | backward | cross-module | dtype cast + FSDP bucket pack | 58 | 38.6 | 8.9 | 29.7 | 0.08 | 4.0 | T0 |
| G-081 | T3 GEMM epilogue (backward): attention.wq_b: mm + _to_copy | backward | intra-module | GEMM epilogue | 61 | 221.3 | 191.8 | 29.5 | 0.08 | 2.0 | T3 |
| G-082 | Aux-loss statistics, recompute copy | recompute | intra-module | reduction + pointwise | 58 | 28.7 | 1.1 | 27.6 | 0.07 |  | T1 |
| G-083 | T3 GEMM epilogue (backward): attention.inner_attention.wkv_b: mm + _to_copy | backward | intra-module | GEMM epilogue | 61 | 112.5 | 85.3 | 27.2 | 0.07 | 0.9 | T3 |
| G-084 | Cross-layer: aux-loss scalar accumulation chained across layers | forward | cross-layer | reduction + pointwise | 57 | 27.8 | 1.0 | 26.8 | 0.07 | 0.0 | T1 |
| G-085 | Aux-loss forward statistics: casts, sums, norms, clamps (~19 tiny kernels) | forward | intra-module | reduction + pointwise | 57 | 27.4 | 1.0 | 26.4 | 0.07 |  | T1 |
| G-086 | w13 weight-grad fp32 cast + reduce-scatter bucket pack | backward | intra-module | dtype cast + FSDP bucket pack | 58 | 35.8 | 9.9 | 25.9 | 0.07 |  | T0 |
| G-087 | Router backward: score-normalization grads + scatter_add into [98304,256] + sigmoid_backward | backward | intra-module | gather/scatter + pointwise | 58 | 28.0 | 2.7 | 25.4 | 0.07 |  | T2 |
| G-088 | Recomputed ffn_norm fused with its RMSNorm backward + residual-grad add | backward/recompute | intra-module, recompute-backward | RMSNorm + pointwise | 60 | 98.8 | 73.8 | 25.0 | 0.07 | 12.3 | T1 |
| G-089 | Recomputed attention_norm fused with its RMSNorm backward + residual-grad add | backward/recompute | intra-module, recompute-backward | RMSNorm + pointwise | 61 | 99.8 | 75.0 | 24.8 | 0.07 | 12.5 | T1 |
| G-090 | T3 GEMM epilogue (backward): feed_forward.w13: mm + _to_copy + cat | backward | intra-module | GEMM epilogue | 3 | 89.3 | 66.0 | 23.3 | 0.06 | 0.7 | T3 |
| G-091 | Intra-module (recompute): feed_forward: silu + mul | recompute | intra-module | pointwise chain | 3 | 28.9 | 6.3 | 22.6 | 0.06 |  | T0 |
| G-092 | T3 GEMM epilogue (backward): feed_forward.w13, ffn_norm: mm + _fused_rms_norm_backward + add | backward | cross-module | GEMM epilogue | 3 | 88.5 | 66.0 | 22.5 | 0.06 | 3.1 | T3 |
| G-093 | Intra-module (forward): feed_forward: silu + mul | forward | intra-module | pointwise chain | 3 | 27.1 | 4.7 | 22.4 | 0.06 |  | T0 |
| G-094 | T3 GEMM prologue (forward): feed_forward.shared_experts.w13: cat + mm | forward | intra-module | GEMM prologue | 58 | 163.4 | 141.9 | 21.5 | 0.06 | 1.0 | T3 |
| G-095 | T3 GEMM epilogue (backward/recompute): attention.q_norm, attention.wq_a: mm + _fused_rms_norm_backward | backward/recompute | cross-module, recompute-backward | GEMM epilogue | 61 | 77.2 | 55.9 | 21.2 | 0.06 | 8.0 | T3 |
| G-096 | Routed + shared expert sum + residual add (two adds, cross-module) | forward | cross-module | pointwise chain | 58 | 68.1 | 47.5 | 20.5 | 0.05 | 23.8 | T0 |
| G-097 | T3 GEMM prologue (backward/recompute): feed_forward.w13: cat + mm | backward/recompute | intra-module, recompute-backward | GEMM prologue | 3 | 84.9 | 66.0 | 18.9 | 0.05 | 0.5 | T3 |
| G-098 | T3 GEMM epilogue (backward): lm_head: mm + _to_copy | backward | intra-module | GEMM epilogue | 8 | 95.2 | 77.2 | 18.0 | 0.05 | 6.5 | T3 |
| G-099 | Router gate hi/mid/lo split chain (casts, shifts, subs), backward | backward | intra-module | pointwise chain | 58 | 19.6 | 2.1 | 17.5 | 0.05 |  | T0 |
| G-100 | T3 GEMM epilogue (forward): attention.q_norm, attention.wq_a: mm + _fused_rms_norm | forward | cross-module | GEMM epilogue | 61 | 72.4 | 55.9 | 16.4 | 0.04 | 5.4 | T3 |
| G-101 | T3 GEMM epilogue (recompute): attention.q_norm, attention.wq_a: mm + _fused_rms_norm | recompute | cross-module | GEMM epilogue | 61 | 72.1 | 55.9 | 16.1 | 0.04 | 5.4 | T3 |
| G-102 | T3 GEMM epilogue (backward): feed_forward.shared_experts.w2: mm + _to_copy | backward | intra-module | GEMM epilogue | 58 | 85.1 | 70.9 | 14.1 | 0.04 | 0.7 | T3 |
| G-103 | T3 GEMM prologue (forward): feed_forward.w13, ffn_norm: _fused_rms_norm + mm | forward | cross-module | GEMM prologue | 3 | 79.9 | 66.0 | 13.9 | 0.04 | 1.2 | T3 |
| G-104 | T3 GEMM epilogue (backward): feed_forward.w2: mm + _to_copy | backward | intra-module | GEMM epilogue | 3 | 46.5 | 33.0 | 13.5 | 0.04 | 0.3 | T3 |
| G-105 | Intra-module (backward): ffn_norm: _fused_rms_norm_backward + add | backward | intra-module | RMSNorm + pointwise | 60 | 74.8 | 61.5 | 13.3 | 0.04 |  | T1 |
| G-106 | T3 GEMM prologue (forward): feed_forward.w13: cat + mm | forward | intra-module | GEMM prologue | 3 | 79.2 | 66.0 | 13.1 | 0.04 | 0.5 | T3 |
| G-107 | Intra-module (backward): attention_norm: _fused_rms_norm_backward + add | backward | intra-module | RMSNorm + pointwise | 61 | 75.3 | 62.5 | 12.8 | 0.03 |  | T1 |
| G-108 | Attention weight-grad fp32 cast + reduce-scatter bucket pack | backward | intra-module | dtype cast + FSDP bucket pack | 61 | 30.1 | 17.6 | 12.5 | 0.03 |  | T0 |
| G-109 | Cross-module (backward): attention, attention.inner_attention, attention.kv_norm, attention.rope: sum + cat + _fused_rms_norm_backward + _to_copy + clone + mul | backward | cross-module | RMSNorm + pointwise | 61 | 143.9 | 132.5 | 11.4 | 0.03 | 1.3 | T1 |
| G-110 | Intra-module (backward): feed_forward.shared_experts.w13: _to_copy + cat | backward | intra-module | copy/concat + cast | 58 | 12.7 | 1.5 | 11.2 | 0.03 |  | T0 |
| G-111 | MoE token counts: per-expert sums + cast before the count all_to_all | forward | intra-module | reduction + pointwise | 58 | 10.8 | 0.2 | 10.5 | 0.03 |  | T1 |
| G-112 | Residual add after attention + ffn_norm RMSNorm (recompute copy) | recompute | cross-module | RMSNorm + pointwise | 61 | 60.2 | 50.0 | 10.2 | 0.03 | 12.5 | T1 |
| G-113 | Cross-layer: residual add (end of layer l) + attention_norm RMSNorm (layer l+1) | forward | cross-layer | RMSNorm + pointwise | 60 | 59.3 | 49.2 | 10.1 | 0.03 | 12.3 | T1 |
| G-114 | Residual add after attention + ffn_norm RMSNorm (forward) | forward | cross-module | RMSNorm + pointwise | 61 | 59.9 | 50.0 | 9.9 | 0.03 | 12.5 | T1 |
| G-115 | T3 GEMM epilogue (forward): layers.*, feed_forward.w2: mm + add | forward | cross-module | GEMM epilogue | 3 | 42.6 | 33.0 | 9.6 | 0.03 | 1.8 | T3 |
| G-116 | T3 GEMM epilogue (backward): attention.wkv_a: mm + _to_copy | backward | intra-module | GEMM epilogue | 61 | 30.1 | 21.0 | 9.1 | 0.02 | 0.2 | T3 |
| G-117 | T3 GEMM prologue (backward): attention, attention.wkv_a: cat + mm | backward | cross-module | GEMM prologue | 61 | 30.0 | 21.0 | 9.0 | 0.02 | 2.0 | T3 |
| G-118 | Cross-module (forward): feed_forward.routed_experts, feed_forward.router, feed_forward.router.aux_loss: sigmoid + add + topk + sum + ones_like + scatter_.value + masked_fill.Scalar + gather + di | forward | cross-module | routing selection (top-k/sort) + pointwise | 1 | 10.4 | 1.9 | 8.5 | 0.02 | 0.0 | T2 |
| G-119 | Cross-layer (backward): attention_norm, tok_embeddings: _fused_rms_norm_backward + add + embedding_dense_backward + _to_copy + bkt._pre_bucket_red | backward | cross-layer | dtype cast + FSDP bucket pack | 1 | 8.7 | 1.4 | 7.3 | 0.02 | 0.4 | T2 |
| G-120 | Cross-module (backward): attention.inner_attention.wkv_b, attention.kv_norm, attention.q_norm, attention.: _to_copy + cat + bkt._pre_bucket_reduce_scatter | backward | cross-module | dtype cast + FSDP bucket pack | 3 | 8.6 | 1.5 | 7.1 | 0.02 | 2.0 | T0 |
| G-121 | Embedding backward (dense scatter-add) + fp32 cast + bucket pack | backward | intra-module | dtype cast + FSDP bucket pack | 1 | 7.4 | 0.7 | 6.7 | 0.02 |  | T2 |
| G-122 | Recompute->backward (backward/recompute): attention.kv_norm: _fused_rms_norm + _fused_rms_norm_backward | backward/recompute | intra-module, recompute-backward | RMSNorm + pointwise | 61 | 9.7 | 3.6 | 6.1 | 0.02 | 0.9 | T1 |
| G-123 | T3 GEMM epilogue (backward): feed_forward.router.gate: mm.dtype + add_ | backward | intra-module | GEMM epilogue | 116 | 30.2 | 25.1 | 5.1 | 0.01 | 0.0 | T3 |
| G-124 | Recompute->backward (backward/recompute): attention.q_norm: _fused_rms_norm + _fused_rms_norm_backward | backward/recompute | intra-module, recompute-backward | RMSNorm + pointwise | 61 | 15.5 | 10.7 | 4.8 | 0.01 | 2.7 | T1 |
| G-125 | Cross-module (recompute): attention.inner_attention, attention.kv_norm, attention.rope: _to_copy + mul + _fused_rms_norm + cat | recompute | cross-module | RMSNorm + pointwise | 61 | 78.3 | 73.6 | 4.8 | 0.01 | 0.2 | T1 |
| G-126 | Cross-module (forward): attention.inner_attention, attention.kv_norm, attention.rope: _to_copy + mul + _fused_rms_norm + cat | forward | cross-module | RMSNorm + pointwise | 61 | 78.3 | 73.6 | 4.7 | 0.01 | 0.2 | T1 |
| G-127 | T3 GEMM prologue (backward): feed_forward.router.gate: cat + mm | backward | intra-module | GEMM prologue | 58 | 30.6 | 26.6 | 4.0 | 0.01 | 0.1 | T3 |
| G-128 | Intra-module (recompute): feed_forward.routed_experts: arange + repeat_interleave + index_select + index + add + sub | recompute | intra-module | pointwise chain | 58 | 3.9 | 0.1 | 3.8 | 0.01 |  | T2 |
| G-129 | Intra-module (backward): feed_forward.w13: _to_copy + cat | backward | intra-module | copy/concat + cast | 3 | 4.5 | 0.7 | 3.8 | 0.01 |  | T0 |
| G-130 | Intra-module (forward): feed_forward.routed_experts: arange + repeat_interleave + index_select + index + add + sub | forward | intra-module | pointwise chain | 58 | 3.7 | 0.1 | 3.6 | 0.01 |  | T2 |
| G-131 | Intra-module (forward): feed_forward.routed_experts: sum + _to_copy + cumsum + sub + clone | forward | intra-module | reduction + pointwise | 58 | 1.8 | 0.0 | 1.8 | 0.01 |  | T1 |
| G-132 | Recompute->backward (backward/recompute): attention.wkv_a, attention.wq_a: _to_copy + constant_pad_nd + bkt._pre_bucket_all_gather | backward/recompute | cross-module, recompute-backward | dtype cast + FSDP bucket pack | 61 | 1.9 | 0.1 | 1.8 | 0.01 | 0.0 | T0 |
| G-133 | Cross-module (forward): attention.wkv_a, attention.wq_a: _to_copy + constant_pad_nd + bkt._pre_bucket_all_gather | forward | cross-module | dtype cast + FSDP bucket pack | 61 | 1.8 | 0.1 | 1.7 | 0.01 | 0.0 | T0 |
| G-134 | Intra-module (recompute): feed_forward.routed_experts: cumsum + sub + clone + sum | recompute | intra-module | reduction + pointwise | 58 | 1.6 | 0.0 | 1.6 | 0.00 |  | T1 |
| G-135 | Intra-module (forward): feed_forward.router.aux_loss: _to_copy + sum + linalg_vector_norm + clamp_min + div + bitwise_not + mul + add_ | forward | intra-module | reduction + pointwise | 1 | 0.5 | 0.0 | 0.5 | 0.00 |  | T1 |
| G-136 | Recompute->backward (backward/recompute): norm: _fused_rms_norm + _fused_rms_norm_backward | backward/recompute | intra-module, recompute-backward | RMSNorm + pointwise | 1 | 1.1 | 0.6 | 0.5 | 0.00 | 0.4 | T1 |
| G-137 | Intra-module (recompute): attention.wkv_a: _to_copy + constant_pad_nd | recompute | intra-module | copy/concat + cast | 61 | 0.2 | 0.0 | 0.2 | 0.00 |  | T0 |
| G-138 | Recompute->backward (backward/recompute): ffn_norm: _fused_rms_norm + _fused_rms_norm_backward | backward/recompute | intra-module, recompute-backward | RMSNorm + pointwise | 1 | 1.0 | 0.8 | 0.2 | 0.00 | 0.2 | T1 |
| G-139 | Intra-module (forward): attention.wkv_a: _to_copy + constant_pad_nd | forward | intra-module | copy/concat + cast | 61 | 0.2 | 0.0 | 0.2 | 0.00 |  | T0 |
| G-140 | Cross-layer (backward): ffn_norm, norm: _fused_rms_norm_backward + add | backward | cross-layer | RMSNorm + pointwise | 1 | 1.2 | 1.0 | 0.2 | 0.00 | 0.2 | T1 |
| G-141 | Cross-layer (forward): layers.*, norm: add + _fused_rms_norm | forward | cross-layer | RMSNorm + pointwise | 1 | 1.0 | 0.8 | 0.2 | 0.00 | 0.2 | T1 |
| G-142 | Cross-layer (forward): attention_norm, tok_embeddings: embedding + _fused_rms_norm | forward | cross-layer | RMSNorm + pointwise | 1 | 0.8 | 0.7 | 0.1 | 0.00 | 0.2 | T1 |
| G-143 | Intra-module (backward): feed_forward.router.gate: add_ | backward | intra-module | pointwise chain | 58 | 0.4 | 0.3 | 0.1 | 0.00 |  | T0 |
| G-144 | Intra-module (backward): loss: add_ | backward | intra-module | pointwise chain | 1 | 0.0 | 0.0 | 0.0 | 0.00 |  | T0 |
| G-145 | Intra-module (forward): feed_forward.router.aux_loss: clamp_min + reciprocal + mul | forward | intra-module | pointwise chain | 1 | 0.0 | 0.0 | 0.0 | 0.00 |  | T0 |
| G-146 | Intra-module (backward): norm: _to_copy + bkt._pre_bucket_reduce_scatter | backward | intra-module | dtype cast + FSDP bucket pack | 1 | 0.0 | 0.0 | 0.0 | 0.00 |  | T0 |

## 4. Region details

Per region: what/where (example node names from the first instance, source lines from `stack_trace`), why it is fusable (dataflow), blockers, tier. Per-instance numbers are averages over instances.

### G-001 MoE combine backward + router/aux-loss score backward (one stage, per layer)
- XM-max; phases backward; scope cross-module; 58 instances; 183.0 kernels/instance; 53580 us measured vs 3539 us ideal per instance (24.337 GB external); gap 2902.4 ms (7.63% of step).
- Where: `layers.*.feed_forward.routed_experts, layers.*.feed_forward.router, layers.*.feed_forward.router.aux_loss, layers.*.feed_forward.router.gate`; nodes `gather_58, _to_copy_1474, mul_997, mul_998, _to_copy_1475, sum_581, new_zeros_3, index_put_2`; source torchtitan/torchtitan/models/common/moe/token_dispatcher.py:403-601; torchtitan/torchtitan/models/common/moe/moe.py:256-606; torchtitan/torchtitan/distributed/local_compile.py:96; torchtitan/torchtitan/models/common/linear/hi_mid_lo_linear.py:117.
- Ops: __lshift__.Scalar, _to_copy.default, add.Tensor, cat.default, div.Tensor, gather.default, gt.Scalar, index_put.default, mul.Tensor, neg.default, new_zeros.default, ones_like.default, scalar_tensor.default, scatter_add.default, sgn.default, sigmoid_backward.default, sub.Tensor, sum.dim_IntList, where.self. Shapes: float32[98304, 256], float32[98304, 8], int32[98304, 256], float32[786432, 7168].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #0, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 3.4 ms | Backward of the MoE combine (token_dispatcher.py:601). Per routed row r with token t(r) and score s(r): g = grad_out[t(r)] (gather_58), dy = bf16(g * s) (mul_998 + _to_copy_1475, feeds the combine all_to_all backward) and ds = sum(g * y) (mul_997 + sum_581, y = recomputed combine output wait_tensor_*_recomputed), ds then un-permuted by index_put_2 (sort permutation). All members are at one fusion stage (no GEMM/collective between them) so one row-wise kernel can read g (gathered), y and s once and write dy and ds. Eager materializes three fp32 [786432,7168] tensors (22.5 GB each) and splits every pass into 16 launches. This XM-max adds the router score-grad chain (div/mul/sum on [98304,8], aux-loss backward, gate hi/mid/lo split casts) that consumes ds at the same stage.
- Blockers: aten.index_put(_) under deterministic mode runs a sort-based path (~21-22 kernels incl. memset/memcpy of the full destination); accumulate=True needs a deterministic segmented reduction. scatter_add accumulation order (atomics) must be fixed for bitwise results. [786432,7168] tensors exceed 2^31 elements: eager TensorIterator splits each op into 16 launches (32-bit indexing). fp32 [N,7168] intermediates from bf16 x fp32-score type promotion double the bytes of every pass.
- Tier T2: order-sensitive data movement (index_put, scatter_add): accumulation / selection order must match eager exactly.
- Overlaps with: G-002, G-003, G-034, G-035, G-051, G-052, G-057, G-075, G-087, G-099

### G-002 MoE combine backward: gather grad + fp32 upcast + score mul/sum + cast (routed_experts)
- L1; phases backward; scope intra-module; 58 instances; 111.0 kernels/instance; 52184 us measured vs 3487 us ideal per instance (23.977 GB external); gap 2824.4 ms (7.42% of step).
- Where: `layers.*.feed_forward.routed_experts`; nodes `gather_58, _to_copy_1474, mul_997, mul_998, _to_copy_1475, sum_581, new_zeros_3, index_put_2`; source torchtitan/torchtitan/models/common/moe/token_dispatcher.py:403-601.
- Ops: _to_copy.default, gather.default, index_put.default, mul.Tensor, new_zeros.default, sum.dim_IntList. Shapes: float32[786432, 7168], bfloat16[786432, 7168], float32[786432], float32[786432, 1].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #2, L1). Backward of the MoE combine (token_dispatcher.py:601). Per routed row r with token t(r) and score s(r): g = grad_out[t(r)] (gather_58), dy = bf16(g * s) (mul_998 + _to_copy_1475, feeds the combine all_to_all backward) and ds = sum(g * y) (mul_997 + sum_581, y = recomputed combine output wait_tensor_*_recomputed), ds then un-permuted by index_put_2 (sort permutation). All members are at one fusion stage (no GEMM/collective between them) so one row-wise kernel can read g (gathered), y and s once and write dy and ds. Eager materializes three fp32 [786432,7168] tensors (22.5 GB each) and splits every pass into 16 launches.
- Blockers: aten.index_put(_) under deterministic mode runs a sort-based path (~21-22 kernels incl. memset/memcpy of the full destination); accumulate=True needs a deterministic segmented reduction. [786432,7168] tensors exceed 2^31 elements: eager TensorIterator splits each op into 16 launches (32-bit indexing). fp32 [N,7168] intermediates from bf16 x fp32-score type promotion double the bytes of every pass.
- Tier T2: order-sensitive data movement (index_put): accumulation / selection order must match eager exactly.
- Overlaps with: G-001, G-003, G-057

### G-003 Cross-layer: next layer's attention_norm backward + residual-grad add fused into combine-backward gather
- XL; phases backward; scope cross-layer; 57 instances; 114.0 kernels/instance; 53455 us measured vs 4307 us ideal per instance (29.614 GB external); gap 2801.4 ms (7.36% of step).
- Where: `layers.*.attention_norm, layers.*.feed_forward.routed_experts`; nodes `_fused_rms_norm_backward_4, add_4081, gather_59, _to_copy_1498, mul_1018, mul_1019, _to_copy_1499, sum_585`; source torchtitan/torchtitan/models/common/nn_modules.py:155; torchtitan/torchtitan/models/common/moe/token_dispatcher.py:403-601.
- Ops: _fused_rms_norm_backward.default, _to_copy.default, add.Tensor, gather.default, index_put.default, mul.Tensor, new_zeros.default, sum.dim_IntList. Shapes: float32[786432, 7168], bfloat16[98304, 7168], bfloat16[786432, 7168], float32[786432].
- Dataflow / notes: XL pair: two L1 components of one phase and stage joined by an edge across a layer/root-module boundary (candidate #3, XL). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 11.7 ms | Layer l+1's RMSNorm backward dx (_fused_rms_norm_backward) + residual gradient add (add_*) produce grad_out of layer l, which the combine backward of layer l gathers immediately (same stage). Backward of the MoE combine (token_dispatcher.py:601). Per routed row r with token t(r) and score s(r): g = grad_out[t(r)] (gather_58), dy = bf16(g * s) (mul_998 + _to_copy_1475, feeds the combine all_to_all backward) and ds = sum(g * y) (mul_997 + sum_581, y = recomputed combine output wait_tensor_*_recomputed), ds then un-permuted by index_put_2 (sort permutation). All members are at one fusion stage (no GEMM/collective between them) so one row-wise kernel can read g (gathered), y and s once and write dy and ds. Eager materializes three fp32 [786432,7168] tensors (22.5 GB each) and splits every pass into 16 launches.
- Blockers: aten.index_put(_) under deterministic mode runs a sort-based path (~21-22 kernels incl. memset/memcpy of the full destination); accumulate=True needs a deterministic segmented reduction. [786432,7168] tensors exceed 2^31 elements: eager TensorIterator splits each op into 16 launches (32-bit indexing). fp32 [N,7168] intermediates from bf16 x fp32-score type promotion double the bytes of every pass. crosses a decoder-layer boundary in backward: the residual gradient handed from layer l+1 to layer l is also read by layer l's other dgrad branch, so the fused kernel must still write it; graph passes treat layers as units (per-layer FSDP reduce-scatter buckets), no compute dependency blocks it. RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T2: order-sensitive data movement (index_put): accumulation / selection order must match eager exactly.
- Overlaps with: G-001, G-002, G-068, G-089, G-107

### G-004 MoE combine forward + routed/shared expert sum (score mul, cast, deterministic scatter-add, add)
- XM; phases forward; scope cross-module; 58 instances; 56.0 kernels/instance; 45573 us measured vs 2256 us ideal per instance (15.512 GB external); gap 2512.4 ms (6.60% of step).
- Where: `layers.*.feed_forward, layers.*.feed_forward.routed_experts`; nodes `mul_24, _to_copy_67, zeros_like_1, deterministic_scatter_add, add_74`; source torchtitan/torchtitan/models/common/moe/token_dispatcher.py:601; torchtitan/torchtitan/models/common/moe/moe.py:825.
- Ops: _to_copy.default, add.Tensor, mul.Tensor, zeros_like.default, torchtitan.deterministic_scatter_add.default. Shapes: bfloat16[98304, 7168], float32[N_routed, 7168], bfloat16[N_routed, 7168].
- Dataflow / notes: XM pair: two L1 components of one phase/layer/stage joined by an edge through a barrier op (candidate #4, XM). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 23.8 ms | Forward MoE combine (token_dispatcher.py:601): mul_24 = fp32(score) * y (fp32 [N,7168], 22.5 GB written), _to_copy_67 back to bf16, zeros_like_1 + torchtitan.deterministic_scatter_add into [98304,7168] (17 ms, 22 kernels incl. a full-size memcpy). One kernel could read y and the scores once and accumulate the 8 routed rows per token in a fixed order (the order deterministic_scatter_add uses). The XM part adds add_74 (routed + shared_experts output) which consumes the scatter result at the same stage.
- Blockers: torchtitan.deterministic_scatter_add is an opaque custom op (deterministic, sort-based scatter-add, ~22 kernels incl. a full-size memcpy); a fused kernel must reproduce its accumulation order. routed-token dimension is data-dependent (unbacked SymInts from .item() on all_to_all split sizes): kernels need dynamic shapes. fp32 [N,7168] intermediates from bf16 x fp32-score type promotion double the bytes of every pass.
- Tier T2: order-sensitive data movement (tt.deterministic_scatter_add): accumulation / selection order must match eager exactly.
- Overlaps with: G-005, G-058, G-096

### G-005 MoE combine forward: fp32 score mul + cast + zeros + deterministic_scatter_add (routed_experts)
- L1; phases forward; scope intra-module; 58 instances; 55.0 kernels/instance; 44986 us measured vs 2051 us ideal per instance (14.102 GB external); gap 2490.2 ms (6.55% of step).
- Where: `layers.*.feed_forward.routed_experts`; nodes `mul_24, _to_copy_67, zeros_like_1, deterministic_scatter_add`; source torchtitan/torchtitan/models/common/moe/token_dispatcher.py:601.
- Ops: _to_copy.default, mul.Tensor, zeros_like.default, torchtitan.deterministic_scatter_add.default. Shapes: bfloat16[98304, 7168], float32[N_routed, 7168], bfloat16[N_routed, 7168].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #5, L1). Forward MoE combine (token_dispatcher.py:601): mul_24 = fp32(score) * y (fp32 [N,7168], 22.5 GB written), _to_copy_67 back to bf16, zeros_like_1 + torchtitan.deterministic_scatter_add into [98304,7168] (17 ms, 22 kernels incl. a full-size memcpy). One kernel could read y and the scores once and accumulate the 8 routed rows per token in a fixed order (the order deterministic_scatter_add uses).
- Blockers: torchtitan.deterministic_scatter_add is an opaque custom op (deterministic, sort-based scatter-add, ~22 kernels incl. a full-size memcpy); a fused kernel must reproduce its accumulation order. routed-token dimension is data-dependent (unbacked SymInts from .item() on all_to_all split sizes): kernels need dynamic shapes. fp32 [N,7168] intermediates from bf16 x fp32-score type promotion double the bytes of every pass.
- Tier T2: order-sensitive data movement (tt.deterministic_scatter_add): accumulation / selection order must match eager exactly.
- Overlaps with: G-004

### G-006 Backward un-permute: zeros + index_put(accumulate=True) of routed-token grads (dispatch backward)
- L1; phases backward; scope intra-module; 116 instances; 23.0 kernels/instance; 22708 us measured vs 2563 us ideal per instance (17.622 GB external); gap 2336.8 ms (6.14% of step).
- Where: `layers.*.feed_forward.routed_experts`; nodes `new_zeros_1, index_put`; source torchtitan/torchtitan/models/common/moe/token_dispatcher.py:403.
- Ops: index_put.default, new_zeros.default. Shapes: bfloat16[N_routed, 7168].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #7, L1). merged signature variants (first/last-layer differences): candidates [10] | Two index_put per MoE layer in backward (merged, x116): (a) index_put(new_zeros_1, [sub_*], dX of the w13 grouped GEMM, accumulate=True) is the backward of the post-all_to_all permutation gather index_8; its indices come from sort/cumsum and are a permutation, so accumulation is unnecessary and the op equals a gather with the inverse permutation. (b) index_put(new_zeros_2 [98304,7168], [floor_divide], a2a-backward output, accumulate=True) is the backward of the token gather index_5 and sums topk=8 rows per token. Under debug.deterministic both run torch's sort-based deterministic index_put (~22 kernels, full-size memset + memcpy): ~22 ms and ~19 ms per layer vs ~2 ms of bytes.
- Blockers: Deterministic mode forces the sort-based index_put path; autograd emits accumulate=True for every index backward. Case (a) needs knowledge that the indices are a permutation (visible in the graph: they derive from aten.sort.stable) to become a gather; case (b) needs a deterministic segmented reduction (rows are already grouped by token after the sort). Routed-token dimension is data-dependent (unbacked SymInts).
- Tier T2: order-sensitive data movement (index_put): accumulation / selection order must match eager exactly.
- Overlaps with: G-007, G-008

### G-007 T3: w13 dgrad grouped GEMM epilogue writes un-permuted rows (replace zeros + index_put)
- T3-epi; phases backward; scope cross-module; 58 instances; 25.0 kernels/instance; 56869 us measured vs 19566 us ideal per instance (17.958 GB external); gap 2163.6 ms (5.69% of step).
- Where: `layers.*.feed_forward.routed_experts, layers.*.feed_forward.routed_experts.w13`; nodes `_grouped_mm_119, new_zeros_1, index_put`; source torchtitan/torchtitan/models/common/linear/linear.py:438; torchtitan/torchtitan/models/common/moe/token_dispatcher.py:403.
- Ops: _grouped_mm.default, index_put.default, new_zeros.default. Shapes: bfloat16[N_routed, 7168].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #6, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 190.2 ms | GEMM _grouped_mm_119: 30856.8 us measured vs 19565.9 us flops-ideal (GEMM-own gap 654.9 ms of this region's gap); glue part 26012.5 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 1508.7 ms. | grouped_mm_119 (dX = dH13 @ W13) output is immediately scattered by index_put(new_zeros_1, [perm], dX, accumulate=True); since perm is a permutation the GEMM epilogue could store each row at perm[r]. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: aten.index_put(_) under deterministic mode runs a sort-based path (~21-22 kernels incl. memset/memcpy of the full destination); accumulate=True needs a deterministic segmented reduction. routed-token dimension is data-dependent (unbacked SymInts from .item() on all_to_all split sizes): kernels need dynamic shapes. needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-006

### G-008 Token-grad accumulation index_put + add of shared-expert dgrad (backward)
- XM; phases backward; scope cross-module; 58 instances; 24.0 kernels/instance; 19990 us measured vs 2050 us ideal per instance (14.099 GB external); gap 1040.5 ms (2.73% of step).
- Where: `layers.*.feed_forward.routed_experts, layers.*.feed_forward.shared_experts.w13`; nodes `new_zeros_2, index_put_1, add_4074`; source torchtitan/torchtitan/models/common/moe/token_dispatcher.py:403.
- Ops: add.Tensor, index_put.default, new_zeros.default. Shapes: bfloat16[98304, 7168].
- Dataflow / notes: XM pair: two L1 components of one phase/layer/stage joined by an edge through a barrier op (candidate #8, XM). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 23.8 ms | index_put_1 (sum of the 8 routed-row grads per token, sort-based deterministic) followed by add_4074 (+ shared_experts.w13 dgrad) at the same stage; a segmented-sum kernel could add the shared-expert grad while writing.
- Blockers: aten.index_put(_) under deterministic mode runs a sort-based path (~21-22 kernels incl. memset/memcpy of the full destination); accumulate=True needs a deterministic segmented reduction.
- Tier T2: order-sensitive data movement (index_put): accumulation / selection order must match eager exactly.
- Overlaps with: G-006, G-049, G-054

### G-009 T3: w2 dgrad grouped GEMM + SwiGLU backward epilogue (routed experts)
- T3-epi; phases backward; scope cross-module; 58 instances; 20.0 kernels/instance; 27475 us measured vs 9783 us ideal per instance (27.498 GB external); gap 1026.2 ms (2.70% of step).
- Where: `layers.*.feed_forward.routed_experts, layers.*.feed_forward.routed_experts.w2`; nodes `_grouped_mm_117, mul_1001, mul_1002, silu_backward_1, stack_1`; source torchtitan/torchtitan/models/common/linear/linear.py:438; torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: _grouped_mm.default, mul.Tensor, silu_backward.default, stack.default. Shapes: bfloat16[N_routed, 2048], bfloat16[N_routed, 2, 2048].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #9, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 163.0 ms | GEMM _grouped_mm_117: 14901.2 us measured vs 9783.0 us flops-ideal (GEMM-own gap 296.9 ms of this region's gap); glue part 12574.1 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 729.3 ms. | grouped_mm_117 (dH = dY @ W2^T) feeds the SwiGLU backward (mul_1001, mul_1002, silu_backward_1, stack_1) that needs the recomputed w13 output; fusing the SwiGLU backward into the dgrad GEMM epilogue avoids writing/reading dH [N,2048] and the stacked [N,2,2048] grad. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: routed-token dimension is data-dependent (unbacked SymInts from .item() on all_to_all split sizes): kernels need dynamic shapes. needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-015, G-016

### G-010 T3: w2 grouped GEMM epilogue stores rows un-permuted (replace new_empty + index_put_), forward
- T3-epi; phases forward; scope cross-module; 58 instances; 23.0 kernels/instance; 26807 us measured vs 9783 us ideal per instance (14.619 GB external); gap 987.4 ms (2.60% of step).
- Where: `layers.*.feed_forward.routed_experts, layers.*.feed_forward.routed_experts.w2`; nodes `_grouped_mm_1, index_put_`; source torchtitan/torchtitan/models/common/linear/linear.py:438; torchtitan/torchtitan/models/common/moe/token_dispatcher.py:601.
- Ops: _grouped_mm.default, index_put_.default. Shapes: bfloat16[N_routed, 7168].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #11, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 190.2 ms | GEMM _grouped_mm_1: 15288.4 us measured vs 9783.0 us flops-ideal (GEMM-own gap 319.3 ms of this region's gap); glue part 11519.0 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 668.1 ms. | grouped_mm_1 output is written into a new_empty buffer by index_put_(accumulate=False) with a permutation index (sub_* from cumsum) before the combine all_to_all; deterministic mode makes this a sort-based 21-kernel op (11 ms/layer). A GEMM epilogue with row remap (or a gather) is one pass. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: aten.index_put(_) under deterministic mode runs a sort-based path (~21-22 kernels incl. memset/memcpy of the full destination); accumulate=True needs a deterministic segmented reduction. routed-token dimension is data-dependent (unbacked SymInts from .item() on all_to_all split sizes): kernels need dynamic shapes. needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-014

### G-011 T3: w2 grouped GEMM + un-permute index_put_ (recompute copy)
- T3-epi; phases recompute; scope cross-module; 58 instances; 23.0 kernels/instance; 26548 us measured vs 9783 us ideal per instance (14.619 GB external); gap 972.4 ms (2.56% of step).
- Where: `layers.*.feed_forward.routed_experts, layers.*.feed_forward.routed_experts.w2`; nodes `_grouped_mm_115_recomputed, index_put__57_recomputed`; source torchtitan/torchtitan/models/common/linear/linear.py:438; torchtitan/torchtitan/models/common/moe/token_dispatcher.py:601.
- Ops: _grouped_mm.default, index_put_.default. Shapes: bfloat16[N_routed, 7168].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #12, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 190.2 ms | GEMM _grouped_mm_115_recomputed: 15164.0 us measured vs 9783.0 us flops-ideal (GEMM-own gap 312.1 ms of this region's gap); glue part 11384.3 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 660.3 ms. | Recompute duplicate of the forward w2 grouped GEMM + index_put_ un-permute, added by recompute_fresh_buffer_index_put_pass so the combine all_to_all input is recomputed instead of saved (full AC semantics). | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: aten.index_put(_) under deterministic mode runs a sort-based path (~21-22 kernels incl. memset/memcpy of the full destination); accumulate=True needs a deterministic segmented reduction. routed-token dimension is data-dependent (unbacked SymInts from .item() on all_to_all split sizes): kernels need dynamic shapes. needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.

### G-012 T3: post-all_to_all token permutation gather fused into w13 grouped GEMM prologue (recompute)
- T3-pro; phases recompute; scope cross-module; 58 instances; 10.0 kernels/instance; 34409 us measured vs 19566 us ideal per instance (29.232 GB external); gap 860.9 ms (2.26% of step).
- Where: `layers.*.feed_forward.routed_experts, layers.*.feed_forward.routed_experts.w13`; nodes `index_350_recomputed, _grouped_mm_114_recomputed`; source torchtitan/torchtitan/models/common/moe/token_dispatcher.py:403; torchtitan/torchtitan/models/common/linear/linear.py:438.
- Ops: _grouped_mm.default, index.Tensor. Shapes: bfloat16[N_routed, 7168], bfloat16[N_routed, 4096].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #13, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 190.2 ms | GEMM _grouped_mm_114_recomputed: 29892.1 us measured vs 19565.9 us flops-ideal (GEMM-own gap 598.9 ms of this region's gap); glue part 4517.3 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 262.0 ms. | identical kernel set as candidate(s) [16] | index_350_recomputed (rows of the received tokens permuted by expert, [N,7168] 11 GB copy) feeds only the w13 grouped GEMM; a gather-GEMM (A rows loaded through the index) removes the copy. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: routed-token dimension is data-dependent (unbacked SymInts from .item() on all_to_all split sizes): kernels need dynamic shapes. needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.

### G-013 T3: post-all_to_all token permutation gather fused into w13 grouped GEMM prologue (forward)
- T3-pro; phases forward; scope cross-module; 58 instances; 10.0 kernels/instance; 34193 us measured vs 19566 us ideal per instance (17.958 GB external); gap 848.4 ms (2.23% of step).
- Where: `layers.*.feed_forward.routed_experts, layers.*.feed_forward.routed_experts.w13`; nodes `index_8, _grouped_mm`; source torchtitan/torchtitan/models/common/moe/token_dispatcher.py:403; torchtitan/torchtitan/models/common/linear/linear.py:438.
- Ops: _grouped_mm.default, index.Tensor. Shapes: bfloat16[N_routed, 7168], bfloat16[N_routed, 4096].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #14, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 190.2 ms | GEMM _grouped_mm: 29764.1 us measured vs 19565.9 us flops-ideal (GEMM-own gap 591.5 ms of this region's gap); glue part 4428.7 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 256.9 ms. | index_8 gathers the received tokens into expert order and feeds only _grouped_mm (w13); a gather-GEMM prologue removes the [N,7168] copy. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: routed-token dimension is data-dependent (unbacked SymInts from .item() on all_to_all split sizes): kernels need dynamic shapes. needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.

### G-014 T3: SwiGLU (silu*mul) fused into w2 grouped GEMM prologue (forward)
- T3-pro; phases forward; scope cross-module; 58 instances; 10.0 kernels/instance; 23548 us measured vs 9783 us ideal per instance (17.834 GB external); gap 798.4 ms (2.10% of step).
- Where: `layers.*.feed_forward.routed_experts, layers.*.feed_forward.routed_experts.w2`; nodes `silu_3, mul_21, _grouped_mm_1`; source torchtitan/torchtitan/distributed/local_compile.py:96; torchtitan/torchtitan/models/common/linear/linear.py:438.
- Ops: _grouped_mm.default, mul.Tensor, silu.default. Shapes: bfloat16[N_routed, 2048], bfloat16[N_routed, 7168].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #15, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 81.5 ms | GEMM _grouped_mm_1: 15288.4 us measured vs 9783.0 us flops-ideal (GEMM-own gap 319.3 ms of this region's gap); glue part 8259.6 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 479.1 ms. | silu_3/mul_21 read the strided halves of the w13 output [N,2,2048] and feed only the w2 grouped GEMM. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: routed-token dimension is data-dependent (unbacked SymInts from .item() on all_to_all split sizes): kernels need dynamic shapes. needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-010, G-019

### G-015 Routed-expert SwiGLU backward: mul, mul, silu_backward, stack -> one kernel
- L1; phases backward; scope intra-module; 58 instances; 18.0 kernels/instance; 12574 us measured vs 2811 us ideal per instance (19.327 GB external); gap 566.3 ms (1.49% of step).
- Where: `layers.*.feed_forward.routed_experts`; nodes `mul_1001, mul_1002, silu_backward_1, stack_1`; source torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: mul.Tensor, silu_backward.default, stack.default. Shapes: bfloat16[N_routed, 2048], bfloat16[N_routed, 2, 2048].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #17, L1). Reads dH (w2 dgrad output) and the recomputed w13 halves; writes the stacked [N,2,2048] grad for the w13 GEMMs. Four kernels with intermediates [N,2048] each; inputs are strided unbind views.
- Blockers: routed-token dimension is data-dependent (unbacked SymInts from .item() on all_to_all split sizes): kernels need dynamic shapes.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-009

### G-016 T3: backward permutation gather of incoming expert grads fused into the w2 dgrad/wgrad grouped GEMM prologue
- T3-pro; phases backward; scope cross-module; 58 instances; 10.0 kernels/instance; 19310 us measured vs 9783 us ideal per instance (25.894 GB external); gap 552.6 ms (1.45% of step).
- Where: `layers.*.feed_forward.routed_experts, layers.*.feed_forward.routed_experts.w2`; nodes `index_351, _grouped_mm_117`; source torchtitan/torchtitan/models/common/moe/token_dispatcher.py:601; torchtitan/torchtitan/models/common/linear/linear.py:438.
- Ops: _grouped_mm.default, index.Tensor. Shapes: bfloat16[N_routed, 7168], bfloat16[N_routed, 2048].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #19, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 190.2 ms | GEMM _grouped_mm_117: 14901.2 us measured vs 9783.0 us flops-ideal (GEMM-own gap 296.9 ms of this region's gap); glue part 4409.1 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 255.7 ms. | index_351 gathers the all_to_all-backward output into expert order (permutation from sort/cumsum, [N,7168] copy, 4.4 ms/layer) and feeds only grouped_mm_117 (dgrad) and, through a transpose view, the w2 wgrad GEMM; a gather-GEMM prologue removes the copy. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: routed-token dimension is data-dependent (unbacked SymInts from .item() on all_to_all split sizes): kernels need dynamic shapes. needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-009

### G-017 T3: w13 wgrad grouped GEMM epilogue emits fp32 directly into the reduce-scatter bucket
- T3-epi; phases backward; scope intra-module; 58 instances; 5.0 kernels/instance; 27861 us measured vs 19566 us ideal per instance (18.656 GB external); gap 481.1 ms (1.26% of step).
- Where: `layers.*.feed_forward.routed_experts.w13`; nodes `_grouped_mm_118, _to_copy_1477, _pre_bucket_reduce_scatter_119`; source torchtitan/torchtitan/models/common/linear/linear.py:437-438.
- Ops: _grouped_mm.default, _to_copy.default, bucketing._pre_bucket_reduce_scatter.default. Shapes: bfloat16[4, 4096, 7168], float32[4, 2, 2048, 7168], float32[176160768].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #20, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 9.9 ms | GEMM _grouped_mm_118: 27243.4 us measured vs 19565.9 us flops-ideal (GEMM-own gap 445.3 ms of this region's gap); glue part 617.5 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 35.8 ms. | grouped_mm_118 (wgrad, bf16) -> _to_copy fp32 -> bucketing._pre_bucket_reduce_scatter copy. An fp32 out_dtype epilogue writing into the bucket slot removes two passes over the expert weight grads. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops). the pack is a custom bucketing op (bucketing._pre_bucket_*); casts must be folded into it or into the producer's epilogue.
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-080, G-086

### G-018 MoE router forward: sigmoid/bias/group-limited top-k/normalize + aux-loss stats + dispatch sort
- XM-max; phases forward; scope cross-module; 57 instances; 82.0 kernels/instance; 9655 us measured vs 1864 us ideal per instance (12.820 GB external); gap 444.1 ms (1.17% of step).
- Where: `layers.*.feed_forward.routed_experts, layers.*.feed_forward.router, layers.*.feed_forward.router.aux_loss`; nodes `sigmoid_1, add_77, topk_3, sum_11, topk_4, ones_like_1, scatter__2, masked_fill_1`; source torchtitan/torchtitan/models/common/moe/moe.py:256-606; torchtitan/torchtitan/distributed/local_compile.py:96; torchtitan/torchtitan/models/common/moe/token_dispatcher.py:294-403.
- Ops: _to_copy.default, add.Tensor, add_.Tensor, bitwise_and.Tensor, bitwise_not.default, clamp_min.default, div.Tensor, floor_divide.default, gather.default, index.Tensor, linalg_vector_norm.default, masked_fill.Scalar, mul.Tensor, ones_like.default, reciprocal.default, scatter_.value, sigmoid.default, sort.stable, sum.default, sum.dim_IntList, topk.default, zeros_like.default. Shapes: float32[256], float32[98304, 256], float32[98304, 8], float32[].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #23, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 2.5 ms | Everything between the router gate GEMM and the token-count all_to_all at one fusion stage: three topk (group top-2 over [98304,8,32], group top-4, final top-8 over 256: 2.1-2.4 ms each), masks, gather, normalization, aux-loss bincount/sums, expert-bias count update, stable sort of the 786432 expert ids and the routed-score gather. A per-token router kernel (256 scores in registers) plus one sort is the target.
- Blockers: top-k / stable sort selection: tie-breaking order must match; produces data-dependent indices. [786432,7168] tensors exceed 2^31 elements: eager TensorIterator splits each op into 16 launches (32-bit indexing).
- Tier T2: order-sensitive data movement (scatter_.value, sort.stable, topk): accumulation / selection order must match eager exactly.
- Overlaps with: G-022, G-023, G-033, G-084, G-085, G-111

### G-019 Routed-expert SwiGLU forward: silu + mul on strided w13 halves (1 kernel instead of 2)
- L1; phases forward; scope intra-module; 58 instances; 8.0 kernels/instance; 8260 us measured vs 1405 us ideal per instance (9.664 GB external); gap 397.5 ms (1.04% of step).
- Where: `layers.*.feed_forward.routed_experts`; nodes `silu_3, mul_21`; source torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: mul.Tensor, silu.default. Shapes: bfloat16[N_routed, 2048].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #25, L1). silu_3 and mul_21 run at ~1.7 TB/s on the non-contiguous unbind views of the [N,2,2048] w13 output.
- Blockers: routed-token dimension is data-dependent (unbacked SymInts from .item() on all_to_all split sizes): kernels need dynamic shapes.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-014

### G-020 Routed-expert SwiGLU forward, recompute copy: silu + mul
- L1; phases recompute; scope intra-module; 58 instances; 8.0 kernels/instance; 8544 us measured vs 1874 us ideal per instance (12.885 GB external); gap 386.9 ms (1.02% of step).
- Where: `layers.*.feed_forward.routed_experts`; nodes `silu_117_recomputed, mul_990_recomputed`; source torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: mul.Tensor, silu.default. Shapes: bfloat16[N_routed, 2048].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #27, L1).
- Blockers: routed-token dimension is data-dependent (unbacked SymInts from .item() on all_to_all split sizes): kernels need dynamic shapes.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.

### G-021 MoE router + dispatch prologue, recompute copy (sigmoid/topk/normalize/sort/gather)
- XM-max; phases recompute; scope cross-module; 58 instances; 63.0 kernels/instance; 7123 us measured vs 1883 us ideal per instance (12.947 GB external); gap 304.0 ms (0.80% of step).
- Where: `layers.*.feed_forward.routed_experts, layers.*.feed_forward.router, layers.*.feed_forward.router.aux_loss`; nodes `sigmoid_57_recomputed, add_3997_recomputed, masked_fill_57_recomputed, topk_173_recomputed, sort_57_recomputed, floor_divide_57_recomputed, index_347_recomputed, gather_57_recomputed`; source torchtitan/torchtitan/models/common/moe/moe.py:256-606; torchtitan/torchtitan/models/common/moe/token_dispatcher.py:403; torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: _to_copy.default, add.Tensor, bitwise_not.default, clamp_min.default, div.Tensor, floor_divide.default, gather.default, index.Tensor, linalg_vector_norm.default, masked_fill.Scalar, mul.Tensor, reciprocal.default, sigmoid.default, sort.stable, sum.default, sum.dim_IntList, topk.default. Shapes: float32[98304, 256], float32[256], float32[], float32[98304, 8].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #29, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 1.0 ms
- Blockers: top-k / stable sort selection: tie-breaking order must match; produces data-dependent indices. [786432,7168] tensors exceed 2^31 elements: eager TensorIterator splits each op into 16 launches (32-bit indexing).
- Tier T2: order-sensitive data movement (sort.stable, topk): accumulation / selection order must match eager exactly.
- Overlaps with: G-032, G-034, G-035, G-040, G-041, G-052, G-082

### G-022 T3: router gate GEMM (mm.dtype fp32 out) + router top-k epilogue (forward)
- T3-epi; phases forward; scope cross-module; 58 instances; 25.0 kernels/instance; 4690 us measured vs 229 us ideal per instance (1.577 GB external); gap 258.7 ms (0.68% of step).
- Where: `layers.*.feed_forward.router, layers.*.feed_forward.router.gate`; nodes `mm_26, sigmoid, add_7, topk, sum_1, topk_1, ones_like, scatter_`; source torchtitan/torchtitan/models/common/linear/hi_mid_lo_linear.py:117; torchtitan/torchtitan/models/common/moe/moe.py:256-313.
- Ops: add.Tensor, add_.Tensor, bitwise_and.Tensor, bitwise_not.default, div.Tensor, gather.default, masked_fill.Scalar, mm.dtype, mul.Tensor, ones_like.default, scatter_.value, sigmoid.default, sum.dim_IntList, topk.default, zeros_like.default. Shapes: float32[98304, 8], float32[98304, 256], bool[98304, 256], bool[98304, 8].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #33, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.0 ms | GEMM mm_26: 254.0 us measured vs 152.9 us flops-ideal (GEMM-own gap 5.9 ms of this region's gap); glue part 4435.9 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 257.3 ms. | The gate GEMM output [98304,256] fp32 is tiny; the region is dominated by the topk kernels, so a GEMM epilogue gains nothing beyond the router-kernel fusion (incremental gap <= 0). | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: top-k / stable sort selection: tie-breaking order must match; produces data-dependent indices. needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-018, G-023, G-118

### G-023 MoE router selection: sigmoid + bias + 3x topk + masks + gather + normalize (forward)
- L1; phases forward; scope intra-module; 58 instances; 24.0 kernels/instance; 4436 us measured vs 39 us ideal per instance (0.265 GB external); gap 255.0 ms (0.67% of step).
- Where: `layers.*.feed_forward.router`; nodes `sigmoid, add_7, topk, sum_1, topk_1, ones_like, scatter_, masked_fill`; source torchtitan/torchtitan/models/common/moe/moe.py:256-313.
- Ops: add.Tensor, add_.Tensor, bitwise_and.Tensor, bitwise_not.default, div.Tensor, gather.default, masked_fill.Scalar, mul.Tensor, ones_like.default, scatter_.value, sigmoid.default, sum.dim_IntList, topk.default, zeros_like.default. Shapes: float32[98304, 8], bool[98304, 256], float32[98304, 256], bool[98304, 8].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #34, L1). 20 kernels per layer on [98304,256] fp32 (100 MB): 4.4 ms measured vs 39 us of bytes; the three aten.topk calls on short rows dominate (2.4 + 0.24 + 1.8 ms).
- Blockers: top-k / stable sort selection: tie-breaking order must match; produces data-dependent indices.
- Tier T2: order-sensitive data movement (scatter_.value, topk): accumulation / selection order must match eager exactly.
- Overlaps with: G-018, G-022, G-118

### G-024 T3: w2 wgrad grouped GEMM epilogue emits fp32 (drop _to_copy)
- T3-epi; phases backward; scope intra-module; 58 instances; 3.0 kernels/instance; 14145 us measured vs 9783 us ideal per instance (14.730 GB external); gap 253.0 ms (0.67% of step).
- Where: `layers.*.feed_forward.routed_experts.w2`; nodes `_grouped_mm_116, _to_copy_1476`; source torchtitan/torchtitan/models/common/linear/linear.py:438.
- Ops: _grouped_mm.default, _to_copy.default. Shapes: bfloat16[4, 7168, 2048], float32[4, 7168, 2048].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #35, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 3.0 ms | GEMM _grouped_mm_116: 14096.4 us measured vs 9783.0 us flops-ideal (GEMM-own gap 250.2 ms of this region's gap); glue part 48.5 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 2.8 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-080

### G-025 Q RoPE backward + grad concat: fp32 casts, conj-mul, cast, cat (backward)
- XM-max; phases backward; scope cross-module; 61 instances; 17.0 kernels/instance; 5388 us measured vs 1409 us ideal per instance (9.689 GB external); gap 242.7 ms (0.64% of step).
- Where: `layers.*.attention, layers.*.attention.rope`; nodes `_to_copy_1489, clone_118, mul_1015, _to_copy_1490, cat_187`; source torchtitan/torchtitan/models/common/rope.py:171; torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: _to_copy.default, cat.default, clone.default, mul.Tensor. Shapes: float32[98304, 128, 64], complex64[98304, 1, 32], complex64[98304, 128, 32], bfloat16[98304, 128, 64].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #37, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 28.6 ms | Backward of q_pe RoPE (complex mul by conj(freqs)) with fp32 round trip, then cat of q_nope/q_pe grads into the wq_b dgrad input [98304,128,192].
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-028, G-029, G-038

### G-026 Q RoPE + Q concat, recompute copy (fp32 cast, complex mul, cast, cat)
- XM-max; phases recompute; scope cross-module; 61 instances; 16.0 kernels/instance; 5366 us measured vs 1409 us ideal per instance (9.689 GB external); gap 241.4 ms (0.63% of step).
- Where: `layers.*.attention, layers.*.attention.rope`; nodes `_to_copy_1419_recomputed, mul_978_recomputed, _to_copy_1420_recomputed, cat_180_recomputed`; source torchtitan/torchtitan/models/common/rope.py:171; torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: _to_copy.default, cat.default, mul.Tensor. Shapes: float32[98304, 128, 64], complex64[98304, 128, 32], bfloat16[98304, 128, 64], bfloat16[98304, 128, 192].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #40, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 28.6 ms
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-030, G-037

### G-027 Q RoPE + Q concat (forward): fp32 cast, complex mul, bf16 cast, cat(q_nope, q_pe)
- XM-max; phases forward; scope cross-module; 61 instances; 16.0 kernels/instance; 5353 us measured vs 1409 us ideal per instance (9.689 GB external); gap 240.6 ms (0.63% of step).
- Where: `layers.*.attention, layers.*.attention.rope`; nodes `_to_copy_6, mul, _to_copy_7, cat`; source torchtitan/torchtitan/models/common/rope.py:171; torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: _to_copy.default, cat.default, mul.Tensor. Shapes: float32[98304, 128, 64], complex64[98304, 128, 32], bfloat16[98304, 128, 64], bfloat16[98304, 128, 192].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #43, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 28.6 ms | q_pe [98304,128,64] is upcast to fp32 (2 ms), multiplied as complex64 by freqs (1.25 ms), cast back (0.67 ms) and concatenated with q_nope into q [98304,128,192] (1.43 ms). One kernel reading the wq_b output and writing q is ~1.4 ms.
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-031, G-036

### G-028 RoPE frequency gather (recompute) + RoPE backward (q and k paths)
- RB; phases backward/recompute; scope intra-module, recompute-backward; 122 instances; 7.5 kernels/instance; 2011 us measured vs 240 us ideal per instance (1.650 GB external); gap 216.1 ms (0.57% of step).
- Where: `layers.*.attention.rope`; nodes `index_345_recomputed, _to_copy_1489, clone_118, mul_1015, _to_copy_1490`; source torchtitan/torchtitan/models/common/rope.py:170-171.
- Ops: _to_copy.default, clone.default, index.Tensor, mul.Tensor. Shapes: complex64[98304, 32], float32[98304, 128, 64], complex64[98304, 1, 32], complex64[98304, 128, 32].
- Dataflow / notes: RB pair: a recompute L1 component feeding a backward L1 component at the same fusion stage (candidate #44, RB). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.4 ms | merged signature variants (first/last-layer differences): candidates [191] | index_*_recomputed gathers freqs_cis by position in the recompute region and is consumed directly by the RoPE backward conj-mul; a fused backward kernel can load freqs by position itself.
- Blockers: recompute duplicates are inserted by selective_activation_remat_pass right before their first backward consumer; fusing them into the backward kernel needs a remat-aware fusion step (no such pass in the default pipeline).
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-025, G-029, G-109

### G-029 RoPE backward (q_pe and k_pe): fp32 cast + conj-mul + cast (backward)
- L1; phases backward; scope intra-module; 122 instances; 6.5 kernels/instance; 1993 us measured vs 240 us ideal per instance (1.648 GB external); gap 213.9 ms (0.56% of step).
- Where: `layers.*.attention.rope`; nodes `_to_copy_1489, clone_118, mul_1015, _to_copy_1490`; source torchtitan/torchtitan/models/common/rope.py:171.
- Ops: _to_copy.default, clone.default, mul.Tensor. Shapes: float32[98304, 128, 64], complex64[98304, 1, 32], complex64[98304, 128, 32], bfloat16[98304, 128, 64].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #45, L1). merged signature variants (first/last-layer differences): candidates [218]
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-025, G-028, G-109

### G-030 RoPE (q_pe and k_pe), recompute copy: fp32 cast + complex mul + cast
- L1; phases recompute; scope intra-module; 122 instances; 5.5 kernels/instance; 1984 us measured vs 240 us ideal per instance (1.648 GB external); gap 212.7 ms (0.56% of step).
- Where: `layers.*.attention.rope`; nodes `_to_copy_1419_recomputed, mul_978_recomputed, _to_copy_1420_recomputed`; source torchtitan/torchtitan/models/common/rope.py:171.
- Ops: _to_copy.default, mul.Tensor. Shapes: float32[98304, 128, 64], complex64[98304, 128, 32], bfloat16[98304, 128, 64].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #47, L1). merged signature variants (first/last-layer differences): candidates [205]
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-026, G-125

### G-031 RoPE (q_pe and k_pe) forward: fp32 cast + complex mul + cast
- L1; phases forward; scope intra-module; 122 instances; 5.5 kernels/instance; 1978 us measured vs 240 us ideal per instance (1.648 GB external); gap 212.0 ms (0.56% of step).
- Where: `layers.*.attention.rope`; nodes `_to_copy_6, mul, _to_copy_7`; source torchtitan/torchtitan/models/common/rope.py:171.
- Ops: _to_copy.default, mul.Tensor. Shapes: float32[98304, 128, 64], complex64[98304, 128, 32], bfloat16[98304, 128, 64].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #48, L1). merged signature variants (first/last-layer differences): candidates [208]
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-027, G-126

### G-032 MoE dispatch, recompute copy: stable sort + floor_divide + routed-score and token gathers
- L1; phases recompute; scope intra-module; 58 instances; 32.0 kernels/instance; 4686 us measured vs 1849 us ideal per instance (12.715 GB external); gap 164.6 ms (0.43% of step).
- Where: `layers.*.feed_forward.routed_experts`; nodes `sort_57_recomputed, floor_divide_57_recomputed, index_347_recomputed, index_346_recomputed`; source torchtitan/torchtitan/models/common/moe/token_dispatcher.py:403.
- Ops: floor_divide.default, index.Tensor, sort.stable. Shapes: int64[786432], bfloat16[786432, 7168], float32[786432].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #49, L1).
- Blockers: top-k / stable sort selection: tie-breaking order must match; produces data-dependent indices. [786432,7168] tensors exceed 2^31 elements: eager TensorIterator splits each op into 16 launches (32-bit indexing).
- Tier T2: order-sensitive data movement (sort.stable): accumulation / selection order must match eager exactly.
- Overlaps with: G-021

### G-033 MoE dispatch: stable sort of expert ids + floor_divide + routed-score gather + token gather [786432,7168]
- L1; phases forward; scope intra-module; 58 instances; 32.0 kernels/instance; 4564 us measured vs 1847 us ideal per instance (12.702 GB external); gap 157.6 ms (0.41% of step).
- Where: `layers.*.feed_forward.routed_experts`; nodes `sort, index_4, floor_divide, index_5`; source torchtitan/torchtitan/models/common/moe/token_dispatcher.py:403.
- Ops: floor_divide.default, index.Tensor, sort.stable. Shapes: int64[786432], float32[786432], bfloat16[786432, 7168].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #50, L1). index_5 copies every token 8x into expert order (11.3 GB written, 4.6 ms) before the dispatch all_to_all. Fusing it with the sort consumers or into the all_to_all send buffer packing removes a pass.
- Blockers: top-k / stable sort selection: tie-breaking order must match; produces data-dependent indices. [786432,7168] tensors exceed 2^31 elements: eager TensorIterator splits each op into 16 launches (32-bit indexing).
- Tier T2: order-sensitive data movement (sort.stable): accumulation / selection order must match eager exactly.
- Overlaps with: G-018, G-118

### G-034 Router + aux-loss recompute fused into their backward (score grads)
- RB; phases backward/recompute; scope cross-module, recompute-backward; 58 instances; 26.0 kernels/instance; 2518 us measured vs 47 us ideal per instance (0.320 GB external); gap 143.4 ms (0.38% of step).
- Where: `layers.*.feed_forward.router, layers.*.feed_forward.router.aux_loss`; nodes `sigmoid_57_recomputed, add_3997_recomputed, masked_fill_57_recomputed, topk_173_recomputed, gather_57_recomputed, sum_572_recomputed, add_3998_recomputed, div_171_recomputed`; source torchtitan/torchtitan/models/common/moe/moe.py:256-606; torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: add.Tensor, div.Tensor, gather.default, gt.Scalar, masked_fill.Scalar, mul.Tensor, neg.default, ones_like.default, scalar_tensor.default, sgn.default, sigmoid.default, sum.dim_IntList, topk.default, where.self. Shapes: float32[98304, 256], float32[98304, 8], float32[98304, 1], float32[].
- Dataflow / notes: RB pair: a recompute L1 component feeding a backward L1 component at the same fusion stage (candidate #51, RB). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.8 ms
- Blockers: top-k / stable sort selection: tie-breaking order must match; produces data-dependent indices. recompute duplicates are inserted by selective_activation_remat_pass right before their first backward consumer; fusing them into the backward kernel needs a remat-aware fusion step (no such pass in the default pipeline).
- Tier T2: order-sensitive data movement (topk): accumulation / selection order must match eager exactly.
- Overlaps with: G-001, G-021, G-035, G-040, G-041, G-052, G-075

### G-035 Router recompute (sigmoid/topk/normalize) fused into router backward
- RB; phases backward/recompute; scope intra-module, recompute-backward; 58 instances; 56.0 kernels/instance; 2425 us measured vs 61 us ideal per instance (0.419 GB external); gap 137.1 ms (0.36% of step).
- Where: `layers.*.feed_forward.router`; nodes `sigmoid_57_recomputed, add_3997_recomputed, masked_fill_57_recomputed, topk_173_recomputed, gather_57_recomputed, sum_572_recomputed, add_3998_recomputed, div_171_recomputed`; source torchtitan/torchtitan/models/common/moe/moe.py:256-291.
- Ops: add.Tensor, div.Tensor, gather.default, masked_fill.Scalar, mul.Tensor, neg.default, new_zeros.default, scatter_add.default, sigmoid.default, sigmoid_backward.default, sum.dim_IntList, topk.default. Shapes: float32[98304, 8], float32[98304, 256], float32[98304, 1], float32[98304, 8, 32].
- Dataflow / notes: RB pair: a recompute L1 component feeding a backward L1 component at the same fusion stage (candidate #53, RB). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 1.0 ms
- Blockers: scatter_add accumulation order (atomics) must be fixed for bitwise results. top-k / stable sort selection: tie-breaking order must match; produces data-dependent indices. recompute duplicates are inserted by selective_activation_remat_pass right before their first backward consumer; fusing them into the backward kernel needs a remat-aware fusion step (no such pass in the default pipeline).
- Tier T2: order-sensitive data movement (scatter_add, topk): accumulation / selection order must match eager exactly.
- Overlaps with: G-001, G-021, G-034, G-040, G-041, G-087

### G-036 T3: wq_b GEMM epilogue writes q directly (RoPE + cat into the GEMM output layout)
- T3-epi; phases forward; scope cross-module; 61 instances; 9.0 kernels/instance; 5144 us measured vs 3145 us ideal per instance (11.652 GB external); gap 122.0 ms (0.32% of step).
- Where: `layers.*.attention, layers.*.attention.wq_b`; nodes `mm_1, cat`; source torchtitan/torchtitan/models/common/linear/linear.py:115; torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: cat.default, mm.default. Shapes: bfloat16[98304, 24576], bfloat16[98304, 128, 192].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #54, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 85.7 ms | GEMM mm_1: 3715.6 us measured vs 3144.5 us flops-ideal (GEMM-own gap 34.8 ms of this region's gap); glue part 1428.6 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 87.1 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-027, G-064

### G-037 T3: wq_b GEMM epilogue writes q directly, recompute copy
- T3-epi; phases recompute; scope cross-module; 61 instances; 9.0 kernels/instance; 5124 us measured vs 3145 us ideal per instance (11.652 GB external); gap 120.8 ms (0.32% of step).
- Where: `layers.*.attention, layers.*.attention.wq_b`; nodes `mm_478_recomputed, cat_180_recomputed`; source torchtitan/torchtitan/models/common/linear/linear.py:115; torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: cat.default, mm.default. Shapes: bfloat16[98304, 24576], bfloat16[98304, 128, 192].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #55, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 85.7 ms | GEMM mm_478_recomputed: 3694.1 us measured vs 3144.5 us flops-ideal (GEMM-own gap 33.5 ms of this region's gap); glue part 1430.2 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 87.2 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-026

### G-038 T3: q-grad concat (dq_nope, dq_pe) folded into the wq_b dgrad GEMM prologue (backward)
- T3-pro; phases backward; scope cross-module; 61 instances; 9.0 kernels/instance; 5116 us measured vs 3145 us ideal per instance (10.041 GB external); gap 120.3 ms (0.32% of step).
- Where: `layers.*.attention, layers.*.attention.wq_b`; nodes `cat_187, mm_524`; source torchtitan/torchtitan/distributed/local_compile.py:96; torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: cat.default, mm.default. Shapes: bfloat16[98304, 128, 192], bfloat16[98304, 1536].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #56, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 85.7 ms | GEMM mm_524: 3688.6 us measured vs 3144.5 us flops-ideal (GEMM-own gap 33.2 ms of this region's gap); glue part 1427.6 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 87.1 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-025, G-060

### G-039 T3: attention.wo wgrad GEMM epilogue emits fp32 (drop _to_copy)
- T3-epi; phases backward; scope intra-module; 61 instances; 2.0 kernels/instance; 11718 us measured vs 9783 us ideal per instance (5.100 GB external); gap 118.0 ms (0.31% of step).
- Where: `layers.*.attention.wo`; nodes `mm_517, _to_copy_1484`; source torchtitan/torchtitan/models/common/linear/linear.py:360.
- Ops: _to_copy.default, mm.default. Shapes: bfloat16[7168, 16384], float32[7168, 16384].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #57, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 6.3 ms | GEMM mm_517: 11620.8 us measured vs 9783.0 us flops-ideal (GEMM-own gap 112.1 ms of this region's gap); glue part 97.2 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 5.9 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-067, G-120

### G-040 T3: router gate GEMM + router top-k epilogue, recompute copy
- T3-epi; phases recompute; scope cross-module; 58 instances; 11.0 kernels/instance; 2196 us measured vs 223 us ideal per instance (1.531 GB external); gap 114.5 ms (0.30% of step).
- Where: `layers.*.feed_forward.router, layers.*.feed_forward.router.gate`; nodes `mm_482_recomputed, sigmoid_57_recomputed, add_3997_recomputed, masked_fill_57_recomputed, topk_173_recomputed, gather_57_recomputed, sum_572_recomputed, add_3998_recomputed`; source torchtitan/torchtitan/models/common/linear/hi_mid_lo_linear.py:117; torchtitan/torchtitan/models/common/moe/moe.py:256-291.
- Ops: add.Tensor, div.Tensor, gather.default, masked_fill.Scalar, mm.dtype, mul.Tensor, sigmoid.default, sum.dim_IntList, topk.default. Shapes: float32[98304, 8], float32[98304, 256], float32[98304, 1], float32[98304, 8, 32].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #58, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.0 ms | GEMM mm_482_recomputed: 253.8 us measured vs 152.9 us flops-ideal (GEMM-own gap 5.9 ms of this region's gap); glue part 1942.3 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 112.7 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: top-k / stable sort selection: tie-breaking order must match; produces data-dependent indices. needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-021, G-034, G-035, G-041

### G-041 MoE router selection, recompute copy (sigmoid, bias, topk, gather, normalize)
- L1; phases recompute; scope intra-module; 58 instances; 10.0 kernels/instance; 1942 us measured vs 32 us ideal per instance (0.218 GB external); gap 110.8 ms (0.29% of step).
- Where: `layers.*.feed_forward.router`; nodes `sigmoid_57_recomputed, add_3997_recomputed, masked_fill_57_recomputed, topk_173_recomputed, gather_57_recomputed, sum_572_recomputed, add_3998_recomputed, div_171_recomputed`; source torchtitan/torchtitan/models/common/moe/moe.py:256-291.
- Ops: add.Tensor, div.Tensor, gather.default, masked_fill.Scalar, mul.Tensor, sigmoid.default, sum.dim_IntList, topk.default. Shapes: float32[98304, 8], float32[98304, 256], float32[98304, 1], float32[98304, 8, 32].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #59, L1).
- Blockers: top-k / stable sort selection: tie-breaking order must match; produces data-dependent indices.
- Tier T2: order-sensitive data movement (topk): accumulation / selection order must match eager exactly.
- Overlaps with: G-021, G-034, G-035, G-040

### G-042 T3: kv-grad concat folded into the wkv_b dgrad GEMM prologue (backward)
- T3-pro; phases backward; scope cross-module; 61 instances; 9.0 kernels/instance; 3657 us measured vs 1893 us ideal per instance (13.019 GB external); gap 107.6 ms (0.28% of step).
- Where: `layers.*.attention.inner_attention, layers.*.attention.inner_attention.wkv_b`; nodes `cat_186, mm_520`; source torchtitan/torchtitan/models/common/attention/mla.py:119; torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: cat.default, mm.default. Shapes: bfloat16[98304, 128, 256], bfloat16[98304, 512].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #60, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 84.1 ms | GEMM mm_520: 1803.7 us measured vs 1397.6 us flops-ideal (GEMM-own gap 24.8 ms of this region's gap); glue part 1853.8 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 107.6 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-078, G-109

### G-043 T3: shared-expert w2 dgrad GEMM + SwiGLU backward epilogue
- T3-epi; phases backward; scope cross-module; 58 instances; 5.0 kernels/instance; 2930 us measured vs 1223 us ideal per instance (3.452 GB external); gap 99.0 ms (0.26% of step).
- Where: `layers.*.feed_forward.shared_experts, layers.*.feed_forward.shared_experts.w2`; nodes `mm_510, mul_995, mul_996, silu_backward, stack`; source torchtitan/torchtitan/models/common/linear/linear.py:360; torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: mm.default, mul.Tensor, silu_backward.default, stack.default. Shapes: bfloat16[98304, 2048], bfloat16[98304, 2, 2048].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #61, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 20.4 ms | GEMM mm_510: 1481.3 us measured vs 1222.9 us flops-ideal (GEMM-own gap 15.0 ms of this region's gap); glue part 1448.8 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 84.0 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-050

### G-044 T3: wkv_b GEMM epilogue writes K = cat(k_nope, broadcast k_pe) directly
- T3-epi; phases forward; scope cross-module; 61 instances; 9.0 kernels/instance; 3073 us measured vs 1661 us ideal per instance (11.421 GB external); gap 86.2 ms (0.23% of step).
- Where: `layers.*.attention.inner_attention, layers.*.attention.inner_attention.wkv_b`; nodes `mm_3, cat_1`; source torchtitan/torchtitan/models/common/linear/linear.py:115; torchtitan/torchtitan/models/common/attention/mla.py:119.
- Ops: cat.default, mm.default. Shapes: bfloat16[98304, 32768], bfloat16[98304, 128, 192].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #62, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 55.5 ms | GEMM mm_3: 1882.7 us measured vs 1397.6 us flops-ideal (GEMM-own gap 29.6 ms of this region's gap); glue part 1190.5 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 72.6 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-072, G-126

### G-045 T3: wkv_b GEMM epilogue writes K directly, recompute copy
- T3-epi; phases recompute; scope cross-module; 61 instances; 9.0 kernels/instance; 3069 us measured vs 1661 us ideal per instance (11.421 GB external); gap 85.9 ms (0.23% of step).
- Where: `layers.*.attention.inner_attention, layers.*.attention.inner_attention.wkv_b`; nodes `mm_480_recomputed, cat_181_recomputed`; source torchtitan/torchtitan/models/common/linear/linear.py:115; torchtitan/torchtitan/models/common/attention/mla.py:119.
- Ops: cat.default, mm.default. Shapes: bfloat16[98304, 32768], bfloat16[98304, 128, 192].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #63, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 55.5 ms | GEMM mm_480_recomputed: 1878.4 us measured vs 1397.6 us flops-ideal (GEMM-own gap 29.3 ms of this region's gap); glue part 1190.8 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 72.6 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-125

### G-046 T3: attention.wo GEMM epilogue adds the residual, recompute copy
- T3-epi; phases recompute; scope cross-module; 61 instances; 2.0 kernels/instance; 11087 us measured vs 9783 us ideal per instance (6.275 GB external); gap 79.6 ms (0.21% of step).
- Where: `layers.*, layers.*.attention.wo`; nodes `mm_481_recomputed, add_3996_recomputed`; source torchtitan/torchtitan/models/common/linear/linear.py:360; torchtitan/torchtitan/models/deepseek_v3/model.py:59.
- Ops: add.Tensor, mm.default. Shapes: bfloat16[98304, 7168].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #64, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 37.5 ms | GEMM mm_481_recomputed: 10501.0 us measured vs 9783.0 us flops-ideal (GEMM-own gap 43.8 ms of this region's gap); glue part 586.4 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 35.8 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-112

### G-047 T3: attention.wo GEMM epilogue adds the residual (forward)
- T3-epi; phases forward; scope cross-module; 61 instances; 2.0 kernels/instance; 10998 us measured vs 9783 us ideal per instance (6.275 GB external); gap 74.1 ms (0.20% of step).
- Where: `layers.*, layers.*.attention.wo`; nodes `mm_4, add`; source torchtitan/torchtitan/models/common/linear/linear.py:360; torchtitan/torchtitan/models/deepseek_v3/model.py:59.
- Ops: add.Tensor, mm.default. Shapes: bfloat16[98304, 7168].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #65, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 37.5 ms | GEMM mm_4: 10412.8 us measured vs 9783.0 us flops-ideal (GEMM-own gap 38.4 ms of this region's gap); glue part 584.9 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 35.7 ms. | add(x, wo(attn)) as a GEMM beta=1 / epilogue add saves one [98304,7168] write+read. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-114

### G-048 T3: shared-expert SwiGLU fused into w2 GEMM prologue
- T3-pro; phases forward; scope cross-module; 58 instances; 3.0 kernels/instance; 2373 us measured vs 1223 us ideal per instance (2.244 GB external); gap 66.7 ms (0.17% of step).
- Where: `layers.*.feed_forward.shared_experts, layers.*.feed_forward.shared_experts.w2`; nodes `silu_4, mul_25, mm_28`; source torchtitan/torchtitan/distributed/local_compile.py:96; torchtitan/torchtitan/models/common/linear/linear.py:360.
- Ops: mm.default, mul.Tensor, silu.default. Shapes: bfloat16[98304, 2048], bfloat16[98304, 7168].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #67, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 10.2 ms | GEMM mm_28: 1422.6 us measured vs 1222.9 us flops-ideal (GEMM-own gap 11.6 ms of this region's gap); glue part 950.8 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 55.1 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-058, G-059

### G-049 T3: shared-expert w13 dgrad GEMM accumulates into the token grad (epilogue add)
- T3-epi; phases backward; scope intra-module; 58 instances; 2.0 kernels/instance; 3554 us measured vs 2446 us ideal per instance (3.683 GB external); gap 64.3 ms (0.17% of step).
- Where: `layers.*.feed_forward.shared_experts.w13`; nodes `mm_512, add_4074`; source torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: add.Tensor, mm.default. Shapes: bfloat16[98304, 7168].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #71, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 35.7 ms | GEMM mm_512: 2967.2 us measured vs 2445.7 us flops-ideal (GEMM-own gap 30.2 ms of this region's gap); glue part 587.0 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 34.0 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-008, G-054, G-076

### G-050 Shared-expert SwiGLU backward: mul, mul, silu_backward, stack -> one kernel
- L1; phases backward; scope intra-module; 58 instances; 4.0 kernels/instance; 1449 us measured vs 351 us ideal per instance (2.416 GB external); gap 63.7 ms (0.17% of step).
- Where: `layers.*.feed_forward.shared_experts`; nodes `mul_995, mul_996, silu_backward, stack`; source torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: mul.Tensor, silu_backward.default, stack.default. Shapes: bfloat16[98304, 2048], bfloat16[98304, 2, 2048].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #72, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-043

### G-051 T3: router gate hi/mid/lo bf16 split (casts, shifts, subs) fused into the gate GEMM prologue (backward)
- T3-pro; phases backward; scope intra-module; 174 instances; 12.0 kernels/instance; 589 us measured vs 243 us ideal per instance (1.668 GB external); gap 60.2 ms (0.16% of step).
- Where: `layers.*.feed_forward.router.gate`; nodes `_to_copy_1478, _to_copy_1479, lshift, sub_174, _to_copy_1480, _to_copy_1481, lshift_1, sub_175`; source torchtitan/torchtitan/models/common/linear/hi_mid_lo_linear.py:117.
- Ops: __lshift__.Scalar, _to_copy.default, cat.default, mm.dtype, sub.Tensor. Shapes: int32[98304, 256], bfloat16[98304, 256], float32[98304, 256], bfloat16[98304, 768].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #74, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.0 ms | GEMM mm_514: 251.3 us measured vs 152.9 us flops-ideal (GEMM-own gap 17.1 ms of this region's gap); glue part 337.5 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 58.7 ms. | identical kernel set as candidate(s) [128] | hi_mid_lo_linear.py:117 emulates the fp32 router gate with three bf16 GEMMs; in backward each fp32 operand is split into hi/mid/lo bf16 parts by _to_copy/__lshift__/sub chains (10 kernels) feeding 3 GEMMs. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-001, G-099, G-123

### G-052 Aux-loss statistics recompute fused into aux-loss backward
- RB; phases backward/recompute; scope intra-module, recompute-backward; 58 instances; 37.0 kernels/instance; 1071 us measured vs 33 us ideal per instance (0.227 GB external); gap 60.2 ms (0.16% of step).
- Where: `layers.*.feed_forward.router.aux_loss`; nodes `_to_copy_1428_recomputed, sum_574_recomputed, linalg_vector_norm_114_recomputed, clamp_min_171_recomputed, div_172_recomputed, bitwise_not_115_recomputed, mul_981_recomputed, sum_575_recomputed`; source torchtitan/torchtitan/distributed/local_compile.py:96; torchtitan/torchtitan/models/common/moe/moe.py:597-606.
- Ops: _to_copy.default, add.Tensor, bitwise_not.default, clamp_min.default, div.Tensor, gt.Scalar, linalg_vector_norm.default, mul.Tensor, neg.default, ones_like.default, reciprocal.default, scalar_tensor.default, sgn.default, sum.default, sum.dim_IntList, where.self. Shapes: float32[98304, 256], float32[], float32[256], float32[98304, 1].
- Dataflow / notes: RB pair: a recompute L1 component feeding a backward L1 component at the same fusion stage (candidate #75, RB). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.9 ms
- Blockers: recompute duplicates are inserted by selective_activation_remat_pass right before their first backward consumer; fusing them into the backward kernel needs a remat-aware fusion step (no such pass in the default pipeline).
- Tier T1: contains reductions (linalg_vector_norm, sum, sum.dim_IntList): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-001, G-021, G-034, G-075, G-082

### G-053 T3: chunked loss backward (nll/log_softmax backward + cast) fused into lm_head dgrad GEMM prologue
- T3-pro; phases backward; scope cross-module; 8 instances; 9.0 kernels/instance; 16987 us measured vs 9649 us ideal per instance (17.916 GB external); gap 58.7 ms (0.15% of step).
- Where: `lm_head, loss`; nodes `div_175, nll_loss_backward, _log_softmax_backward_data, _to_copy_1441, mm_487`; source torchtitan/torchtitan/distributed/local_compile.py:96; torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: _log_softmax_backward_data.default, _to_copy.default, div.Tensor, mm.default, nll_loss_backward.default. Shapes: float32[12288, 129280], float32[], bfloat16[12288, 129280], bfloat16[12288, 7168].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #76, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 18.5 ms | GEMM mm_487: 10798.5 us measured vs 9649.2 us flops-ideal (GEMM-own gap 9.2 ms of this region's gap); glue part 6188.8 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 49.5 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-077

### G-054 Backward residual-grad adds around ffn_norm: add(dgrads) + RMSNorm backward + add (cross-module)
- XM-max; phases backward; scope cross-module; 58 instances; 5.0 kernels/instance; 2408 us measured vs 1428 us ideal per instance (9.817 GB external); gap 56.9 ms (0.15% of step).
- Where: `layers.*.feed_forward.router.gate, layers.*.feed_forward.shared_experts.w13, layers.*.ffn_norm`; nodes `add_4082, add_4086, _fused_rms_norm_backward_5, add_4087`; source torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm_backward.default, add.Tensor. Shapes: bfloat16[98304, 7168], bfloat16[7168].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #77, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 47.5 ms | merged signature variants (first/last-layer differences): candidates [226] | Sums the dgrads of router.gate and shared_experts.w13 into the ffn_norm output grad, runs _fused_rms_norm_backward and adds the residual grad: four [98304,7168] passes at one stage.
- Blockers: RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T1: contains reductions (_fused_rms_norm_backward): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-008, G-049, G-065, G-088, G-105, G-138, G-140

### G-055 T3: dense FFN (layers 0-2) w2 dgrad GEMM + SwiGLU backward epilogue
- T3-epi; phases backward; scope cross-module; 3 instances; 19.0 kernels/instance; 27954 us measured vs 11006 us ideal per instance (19.793 GB external); gap 50.8 ms (0.13% of step).
- Where: `layers.*.feed_forward, layers.*.feed_forward.w2`; nodes `mm_1554, mul_2213, mul_2214, silu_backward_116, stack_116`; source torchtitan/torchtitan/models/common/linear/linear.py:360; torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: mm.default, mul.Tensor, silu_backward.default, stack.default. Shapes: bfloat16[98304, 18432], bfloat16[98304, 2, 18432].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #78, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 9.5 ms | GEMM mm_1554: 13816.8 us measured vs 11005.8 us flops-ideal (GEMM-own gap 8.4 ms of this region's gap); glue part 14137.2 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 42.4 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-073

### G-056 T3: wq_a/wkv_a dgrad GEMM accumulates into the token grad (epilogue add)
- T3-epi; phases backward; scope cross-module; 61 instances; 2.0 kernels/instance; 1690 us measured vs 917 us ideal per instance (3.143 GB external); gap 47.1 ms (0.12% of step).
- Where: `layers.*.attention.wkv_a, layers.*.attention.wq_a`; nodes `mm_526, add_4080`; source torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: add.Tensor, mm.default. Shapes: bfloat16[98304, 7168].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #79, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 37.5 ms | GEMM mm_526: 1101.7 us measured vs 917.2 us flops-ideal (GEMM-own gap 11.3 ms of this region's gap); glue part 588.3 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 35.9 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-066, G-068

### G-057 Cross-layer: final norm backward fused into last layer's combine-backward gather
- XL; phases backward; scope cross-layer; 1 instances; 113.0 kernels/instance; 50763 us measured vs 3897 us ideal per instance (26.796 GB external); gap 46.9 ms (0.12% of step).
- Where: `layers.*.feed_forward.routed_experts, norm`; nodes `_fused_rms_norm_backward, gather_58, _to_copy_1474, mul_997, mul_998, _to_copy_1475, sum_581, new_zeros_3`; source torchtitan/torchtitan/models/common/nn_modules.py:155; torchtitan/torchtitan/models/common/moe/token_dispatcher.py:403-601.
- Ops: _fused_rms_norm_backward.default, _to_copy.default, gather.default, index_put.default, mul.Tensor, new_zeros.default, sum.dim_IntList. Shapes: float32[786432, 7168], bfloat16[786432, 7168], float32[786432], bfloat16[98304, 7168].
- Dataflow / notes: XL pair: two L1 components of one phase and stage joined by an edge across a layer/root-module boundary (candidate #80, XL). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.2 ms
- Blockers: aten.index_put(_) under deterministic mode runs a sort-based path (~21-22 kernels incl. memset/memcpy of the full destination); accumulate=True needs a deterministic segmented reduction. [786432,7168] tensors exceed 2^31 elements: eager TensorIterator splits each op into 16 launches (32-bit indexing). fp32 [N,7168] intermediates from bf16 x fp32-score type promotion double the bytes of every pass. crosses a decoder-layer boundary in backward: the residual gradient handed from layer l+1 to layer l is also read by layer l's other dgrad branch, so the fused kernel must still write it; graph passes treat layers as units (per-layer FSDP reduce-scatter buckets), no compute dependency blocks it. RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T2: order-sensitive data movement (index_put): accumulation / selection order must match eager exactly.
- Overlaps with: G-001, G-002, G-136

### G-058 T3: shared-expert w2 GEMM epilogue adds the routed-expert output
- T3-epi; phases forward; scope cross-module; 58 instances; 2.0 kernels/instance; 2010 us measured vs 1223 us ideal per instance (3.251 GB external); gap 45.6 ms (0.12% of step).
- Where: `layers.*.feed_forward, layers.*.feed_forward.shared_experts.w2`; nodes `mm_28, add_74`; source torchtitan/torchtitan/models/common/linear/linear.py:360; torchtitan/torchtitan/models/common/moe/moe.py:825.
- Ops: add.Tensor, mm.default. Shapes: bfloat16[98304, 7168].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #81, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 35.7 ms | GEMM mm_28: 1422.6 us measured vs 1222.9 us flops-ideal (GEMM-own gap 11.6 ms of this region's gap); glue part 587.1 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 34.1 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-004, G-048, G-096

### G-059 Shared-expert SwiGLU forward: silu + mul
- L1; phases forward; scope intra-module; 58 instances; 2.0 kernels/instance; 951 us measured vs 176 us ideal per instance (1.208 GB external); gap 45.0 ms (0.12% of step).
- Where: `layers.*.feed_forward.shared_experts`; nodes `silu_4, mul_25`; source torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: mul.Tensor, silu.default. Shapes: bfloat16[98304, 2048].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #82, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-048

### G-060 T3: wq_b dgrad GEMM + q_norm RMSNorm backward epilogue
- T3-epi; phases backward; scope cross-module; 61 instances; 3.0 kernels/instance; 3857 us measured vs 3145 us ideal per instance (5.512 GB external); gap 43.5 ms (0.11% of step).
- Where: `layers.*.attention.q_norm, layers.*.attention.wq_b`; nodes `mm_524, _fused_rms_norm_backward_3`; source torchtitan/torchtitan/models/common/linear/linear.py:115; torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm_backward.default, mm.default. Shapes: bfloat16[98304, 1536], bfloat16[1536].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #85, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 8.0 ms | GEMM mm_524: 3688.6 us measured vs 3144.5 us flops-ideal (GEMM-own gap 33.2 ms of this region's gap); glue part 168.8 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 10.3 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops). RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-038, G-095, G-124

### G-061 Shared-expert SwiGLU forward, recompute copy
- L1; phases recompute; scope intra-module; 58 instances; 2.0 kernels/instance; 971 us measured vs 234 us ideal per instance (1.611 GB external); gap 42.8 ms (0.11% of step).
- Where: `layers.*.feed_forward.shared_experts`; nodes `silu_118_recomputed, mul_994_recomputed`; source torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: mul.Tensor, silu.default. Shapes: bfloat16[98304, 2048].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #86, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.

### G-062 T3: wq_a wgrad GEMM epilogue emits fp32 into the reduce-scatter bucket
- T3-epi; phases backward; scope intra-module; 61 instances; 4.0 kernels/instance; 1606 us measured vs 917 us ideal per instance (3.675 GB external); gap 42.0 ms (0.11% of step).
- Where: `layers.*.attention.wq_a`; nodes `mm_525, _to_copy_1494, _pre_bucket_reduce_scatter_118`; source torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: _to_copy.default, mm.default, bucketing._pre_bucket_reduce_scatter.default. Shapes: bfloat16[1536, 7168], float32[1536, 7168], float32[234373120].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #97, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 17.6 ms | GEMM mm_525: 1111.9 us measured vs 917.2 us flops-ideal (GEMM-own gap 11.9 ms of this region's gap); glue part 493.9 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 30.1 ms. | merged signature variants (first/last-layer differences): candidates [183] | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops). the pack is a custom bucketing op (bucketing._pre_bucket_*); casts must be folded into it or into the producer's epilogue.
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-067, G-108, G-120

### G-063 T3: chunked loss: lm_head GEMM + fp32 upcast + log_softmax + nll (fused linear cross-entropy)
- T3-epi; phases forward; scope cross-module; 1 instances; 67.0 kernels/instance; 52806 us measured vs 11127 us ideal per instance (76.515 GB external); gap 41.7 ms (0.11% of step).
- Where: `lm_head, loss`; nodes `new_zeros, _to_copy_1440, _log_softmax, nll_loss_forward, div_174, add_4066, _to_copy_1444, _log_softmax_1`; source torchtitan/torchtitan/distributed/local_compile.py:96; torchtitan/torchtitan/models/common/linear/linear.py:115; torchtitan/torchtitan/experiments/graph_trainer/chunked_loss.py:107.
- Ops: _log_softmax.default, _to_copy.default, add.Tensor, clone.default, div.Tensor, mm.default, new_zeros.default, nll_loss_forward.default. Shapes: float32[], float32[12288, 129280], bfloat16[12288, 129280].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #88, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 9.8 ms | GEMM mm_494: 10985.7 us measured vs 9649.2 us flops-ideal (GEMM-own gap 1.3 ms of this region's gap); glue part 41820.2 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 41.7 ms. | identical kernel set as candidate(s) [89, 90, 91, 92, 93, 94, 95] | Eight 12288-token chunks; each upcasts logits [12288,129280] to fp32 (1.3 ms), log_softmax (3.8 ms) and nll. A fused linear-cross-entropy kernel never materializes the fp32 logits. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-079

### G-064 T3: q_norm RMSNorm fused into the wq_b GEMM prologue (forward)
- T3-pro; phases forward; scope cross-module; 61 instances; 2.0 kernels/instance; 3801 us measured vs 3145 us ideal per instance (5.209 GB external); gap 40.0 ms (0.10% of step).
- Where: `layers.*.attention.q_norm, layers.*.attention.wq_b`; nodes `_fused_rms_norm_1, mm_1`; source torchtitan/torchtitan/models/common/nn_modules.py:155; torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: _fused_rms_norm.default, mm.default. Shapes: bfloat16[98304, 1536], float32[98304, 1], bfloat16[98304, 24576].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #96, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 5.4 ms | GEMM mm_1: 3715.6 us measured vs 3144.5 us flops-ideal (GEMM-own gap 34.8 ms of this region's gap); glue part 85.3 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 5.2 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-036, G-100

### G-065 T3: router-gate dgrad GEMM accumulates into the token grad (epilogue add)
- T3-epi; phases backward; scope intra-module; 58 instances; 2.0 kernels/instance; 1114 us measured vs 459 us ideal per instance (2.981 GB external); gap 38.0 ms (0.10% of step).
- Where: `layers.*.feed_forward.router.gate`; nodes `mm_513, add_4078`; source torchtitan/torchtitan/models/common/linear/hi_mid_lo_linear.py:117.
- Ops: add.Tensor, mm.default. Shapes: bfloat16[98304, 7168].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #98, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 35.7 ms | GEMM mm_513: 522.1 us measured vs 458.6 us flops-ideal (GEMM-own gap 3.7 ms of this region's gap); glue part 591.4 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 34.3 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-054, G-127

### G-066 T3: wkv_a dgrad GEMM accumulates into the token grad (epilogue add)
- T3-epi; phases backward; scope intra-module; 61 instances; 2.0 kernels/instance; 1044 us measured vs 428 us ideal per instance (2.940 GB external); gap 37.6 ms (0.10% of step).
- Where: `layers.*.attention.wkv_a`; nodes `mm_522, add_4080`; source torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: add.Tensor, mm.default. Shapes: bfloat16[98304, 7168].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #99, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 32.4 ms | GEMM mm_522: 455.9 us measured vs 343.9 us flops-ideal (GEMM-own gap 6.8 ms of this region's gap); glue part 588.3 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 35.9 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-056, G-068, G-117

### G-067 Attention/shared-expert weight-grad casts, shard re-cat and bucket pack (backward)
- XM-max; phases backward; scope cross-module; 58 instances; 19.0 kernels/instance; 852 us measured vs 209 us ideal per instance (1.437 GB external); gap 37.3 ms (0.10% of step).
- Where: `layers.*.attention.inner_attention.wkv_b, layers.*.attention.kv_norm, layers.*.attention.q_norm, layers.*.attention.wkv_a, layers.*.attention.wo, layers.*.attention.wq_a, layers.*.attention.wq_b, layers.*.attention_norm, layers.*.feed_forward.router.gate, layers.*.feed_forward.shared_experts.w13, layers.*.feed_forward.shared_experts.w2, layers.*.ffn_norm`; nodes `_to_copy_1472, _to_copy_1473, cat_183, add__123, add__124, _to_copy_1483, _to_copy_1484, _to_copy_1485`; source torchtitan/torchtitan/models/common/linear/linear.py:115-360; torchtitan/torchtitan/models/common/linear/hi_mid_lo_linear.py:117; torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _to_copy.default, add_.Tensor, cat.default, bucketing._pre_bucket_reduce_scatter.default. Shapes: float32[256, 7168], float32[7168], float32[7168, 2048], float32[2, 2048, 7168].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #100, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 14.9 ms
- Blockers: the pack is a custom bucketing op (bucketing._pre_bucket_*); casts must be folded into it or into the producer's epilogue.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-039, G-062, G-070, G-081, G-083, G-102, G-108, G-110, G-116, G-123, G-143

### G-068 Backward: dgrad add + attention_norm RMSNorm backward + residual-grad add
- XM-max; phases backward; scope cross-module; 61 instances; 4.0 kernels/instance; 1823 us measured vs 1230 us ideal per instance (8.456 GB external); gap 36.2 ms (0.10% of step).
- Where: `layers.*.attention.wkv_a, layers.*.attention_norm`; nodes `add_4080, _fused_rms_norm_backward_4, add_4081`; source torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm_backward.default, add.Tensor. Shapes: bfloat16[98304, 7168], bfloat16[7168].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #101, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 25.0 ms
- Blockers: RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T1: contains reductions (_fused_rms_norm_backward): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-003, G-056, G-066, G-089, G-107, G-119

### G-069 T3: attention_norm RMSNorm fused into the wq_a GEMM prologue (forward)
- T3-pro; phases forward; scope cross-module; 61 instances; 2.0 kernels/instance; 1501 us measured vs 917 us ideal per instance (3.143 GB external); gap 35.6 ms (0.09% of step).
- Where: `layers.*.attention.wq_a, layers.*.attention_norm`; nodes `_fused_rms_norm, mm`; source torchtitan/torchtitan/models/common/nn_modules.py:155; torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: _fused_rms_norm.default, mm.default. Shapes: bfloat16[98304, 7168], float32[98304, 1], bfloat16[98304, 1536].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #103, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 25.0 ms | GEMM mm: 1101.1 us measured vs 917.2 us flops-ideal (GEMM-own gap 11.2 ms of this region's gap); glue part 400.3 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 24.4 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-074, G-100, G-113, G-142

### G-070 T3 GEMM epilogue (backward): feed_forward.shared_experts.w13: mm + _to_copy + cat
- T3-epi; phases backward; scope intra-module; 58 instances; 6.0 kernels/instance; 3055 us measured vs 2446 us ideal per instance (2.332 GB external); gap 35.3 ms (0.09% of step).
- Where: `layers.*.feed_forward.shared_experts.w13`; nodes `mm_511, _to_copy_1473, cat_183`; source torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: _to_copy.default, cat.default, mm.default. Shapes: bfloat16[4096, 7168], float32[2, 2048, 7168], float32[512, 8, 7168].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #104, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 1.5 ms | GEMM mm_511: 2836.2 us measured vs 2445.7 us flops-ideal (GEMM-own gap 22.6 ms of this region's gap); glue part 219.0 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 12.7 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-067, G-110

### G-071 T3 GEMM prologue (forward): feed_forward, feed_forward.w2: silu + mul + mm
- T3-pro; phases forward; scope cross-module; 3 instances; 9.0 kernels/instance; 22650 us measured vs 11006 us ideal per instance (8.921 GB external); gap 34.9 ms (0.09% of step).
- Where: `layers.*.feed_forward, layers.*.feed_forward.w2`; nodes `silu, mul_2, mm_6`; source torchtitan/torchtitan/distributed/local_compile.py:96; torchtitan/torchtitan/models/common/linear/linear.py:360.
- Ops: mm.default, mul.Tensor, silu.default. Shapes: bfloat16[98304, 18432], bfloat16[98304, 7168].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #105, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 4.7 ms | GEMM mm_6: 13603.7 us measured vs 11005.8 us flops-ideal (GEMM-own gap 7.8 ms of this region's gap); glue part 9046.5 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 27.1 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-093, G-115

### G-072 T3 GEMM prologue (forward): attention.inner_attention.wkv_b, attention.kv_norm: _fused_rms_norm + mm
- T3-pro; phases forward; scope cross-module; 61 instances; 3.0 kernels/instance; 1945 us measured vs 1398 us ideal per instance (6.577 GB external); gap 33.4 ms (0.09% of step).
- Where: `layers.*.attention.inner_attention.wkv_b, layers.*.attention.kv_norm`; nodes `_fused_rms_norm_2, mm_3`; source torchtitan/torchtitan/models/common/nn_modules.py:155; torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: _fused_rms_norm.default, mm.default. Shapes: bfloat16[98304, 512], float32[98304, 1], bfloat16[98304, 32768].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #107, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 1.8 ms | GEMM mm_3: 1882.7 us measured vs 1397.6 us flops-ideal (GEMM-own gap 29.6 ms of this region's gap); glue part 62.1 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 3.8 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-044, G-126

### G-073 Intra-module (backward): feed_forward: mul + silu_backward + stack
- L1; phases backward; scope intra-module; 3 instances; 18.0 kernels/instance; 14137 us measured vs 3162 us ideal per instance (21.743 GB external); gap 32.9 ms (0.09% of step).
- Where: `layers.*.feed_forward`; nodes `mul_2213, mul_2214, silu_backward_116, stack_116`; source torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: mul.Tensor, silu_backward.default, stack.default. Shapes: bfloat16[98304, 18432], bfloat16[98304, 2, 18432].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #108, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-055

### G-074 T3: attention_norm RMSNorm fused into the wkv_a GEMM prologue (forward)
- T3-pro; phases forward; scope cross-module; 61 instances; 2.0 kernels/instance; 949 us measured vs 428 us ideal per instance (2.940 GB external); gap 31.8 ms (0.08% of step).
- Where: `layers.*.attention.wkv_a, layers.*.attention_norm`; nodes `_fused_rms_norm, mm_2`; source torchtitan/torchtitan/models/common/nn_modules.py:155; torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: _fused_rms_norm.default, mm.default. Shapes: bfloat16[98304, 7168], float32[98304, 1], bfloat16[98304, 576].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #110, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 19.9 ms | GEMM mm_2: 549.0 us measured vs 343.9 us flops-ideal (GEMM-own gap 12.5 ms of this region's gap); glue part 400.3 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 24.4 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-069, G-113, G-142

### G-075 Aux-loss backward: ~16 tiny kernels on [256] / [98304,256]
- L1; phases backward; scope intra-module; 58 instances; 16.0 kernels/instance; 576 us measured vs 29 us ideal per instance (0.202 GB external); gap 31.7 ms (0.08% of step).
- Where: `layers.*.feed_forward.router.aux_loss`; nodes `ones_like_67, mul_1007, mul_1008, mul_1009, neg, div_190, div_191, mul_1010`; source torchtitan/torchtitan/models/common/moe/moe.py:605-606; torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: add.Tensor, div.Tensor, gt.Scalar, mul.Tensor, neg.default, ones_like.default, scalar_tensor.default, sgn.default, sum.dim_IntList, where.self. Shapes: float32[98304, 256], float32[], float32[98304, 1], float32[256].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #111, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T1: contains reductions (sum.dim_IntList): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-001, G-034, G-052

### G-076 T3 GEMM prologue (backward/recompute): feed_forward.shared_experts.w13: cat + mm
- T3-pro; phases backward/recompute; scope intra-module, recompute-backward; 58 instances; 3.0 kernels/instance; 2990 us measured vs 2446 us ideal per instance (2.332 GB external); gap 31.5 ms (0.08% of step).
- Where: `layers.*.feed_forward.shared_experts.w13`; nodes `cat_182_recomputed, mm_512`; source torchtitan/torchtitan/experiments/graph_trainer/simple_fsdp.py:366; torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: cat.default, mm.default. Shapes: bfloat16[2, 2048, 7168], bfloat16[98304, 7168].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #112, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 1.0 ms | GEMM mm_512: 2967.2 us measured vs 2445.7 us flops-ideal (GEMM-own gap 30.2 ms of this region's gap); glue part 22.4 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 1.3 ms. | identical kernel set as candidate(s) [134] | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: recompute duplicates are inserted by selective_activation_remat_pass right before their first backward consumer; fusing them into the backward kernel needs a remat-aware fusion step (no such pass in the default pipeline). needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-049

### G-077 Chunked loss backward: nll_loss_backward + log_softmax_backward + bf16 cast
- L1; phases backward; scope intra-module; 8 instances; 8.0 kernels/instance; 6189 us measured vs 2310 us ideal per instance (15.886 GB external); gap 31.0 ms (0.08% of step).
- Where: `loss`; nodes `div_175, nll_loss_backward, _log_softmax_backward_data, _to_copy_1441`; source torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: _log_softmax_backward_data.default, _to_copy.default, div.Tensor, nll_loss_backward.default. Shapes: float32[12288, 129280], float32[], bfloat16[12288, 129280].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #113, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T1: contains reductions (_log_softmax_backward_data, nll_loss_backward): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-053

### G-078 T3 GEMM epilogue (backward): attention.inner_attention.wkv_b, attention.kv_norm: mm + _fused_rms_norm_backward
- T3-epi; phases backward; scope cross-module; 61 instances; 4.0 kernels/instance; 1901 us measured vs 1398 us ideal per instance (6.678 GB external); gap 30.7 ms (0.08% of step).
- Where: `layers.*.attention.inner_attention.wkv_b, layers.*.attention.kv_norm`; nodes `mm_520, _fused_rms_norm_backward_2`; source torchtitan/torchtitan/models/common/linear/linear.py:115; torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm_backward.default, mm.default. Shapes: bfloat16[98304, 512], bfloat16[512].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #115, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 2.7 ms | GEMM mm_520: 1803.7 us measured vs 1397.6 us flops-ideal (GEMM-own gap 24.8 ms of this region's gap); glue part 97.1 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 5.9 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops). RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-042, G-109, G-122

### G-079 Chunked loss forward: fp32 upcast + log_softmax + nll + scaling (8 chunks, one stage)
- L1; phases forward; scope intra-module; 1 instances; 66.0 kernels/instance; 41820 us measured vs 11294 us ideal per instance (77.663 GB external); gap 30.5 ms (0.08% of step).
- Where: `loss`; nodes `new_zeros, _to_copy_1440, _log_softmax, nll_loss_forward, div_174, add_4066, _to_copy_1444, _log_softmax_1`; source torchtitan/torchtitan/distributed/local_compile.py:96; torchtitan/torchtitan/experiments/graph_trainer/chunked_loss.py:107.
- Ops: _log_softmax.default, _to_copy.default, add.Tensor, clone.default, div.Tensor, new_zeros.default, nll_loss_forward.default. Shapes: float32[], float32[12288, 129280].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #116, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T1: contains reductions (_log_softmax, nll_loss_forward): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-063

### G-080 Routed-expert weight-grad fp32 casts + reduce-scatter bucket pack
- XM-max; phases backward; scope cross-module; 58 instances; 4.0 kernels/instance; 666 us measured vs 154 us ideal per instance (1.057 GB external); gap 29.7 ms (0.08% of step).
- Where: `layers.*.feed_forward.routed_experts.w13, layers.*.feed_forward.routed_experts.w2`; nodes `_to_copy_1476, _to_copy_1477, _pre_bucket_reduce_scatter_119`; source torchtitan/torchtitan/models/common/linear/linear.py:437-438.
- Ops: _to_copy.default, bucketing._pre_bucket_reduce_scatter.default. Shapes: float32[4, 7168, 2048], float32[4, 2, 2048, 7168], float32[176160768].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #117, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 4.0 ms
- Blockers: the pack is a custom bucketing op (bucketing._pre_bucket_*); casts must be folded into it or into the producer's epilogue.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-017, G-024, G-086

### G-081 T3 GEMM epilogue (backward): attention.wq_b: mm + _to_copy
- T3-epi; phases backward; scope intra-module; 61 instances; 2.0 kernels/instance; 3629 us measured vs 3145 us ideal per instance (5.285 GB external); gap 29.5 ms (0.08% of step).
- Where: `layers.*.attention.wq_b`; nodes `mm_523, _to_copy_1492`; source torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: _to_copy.default, mm.default. Shapes: bfloat16[24576, 1536], float32[24576, 1536].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #118, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 2.0 ms | GEMM mm_523: 3598.9 us measured vs 3144.5 us flops-ideal (GEMM-own gap 27.7 ms of this region's gap); glue part 29.7 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 1.8 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-067, G-120

### G-082 Aux-loss statistics, recompute copy
- L1; phases recompute; scope intra-module; 58 instances; 21.0 kernels/instance; 495 us measured vs 18 us ideal per instance (0.127 GB external); gap 27.6 ms (0.07% of step).
- Where: `layers.*.feed_forward.router.aux_loss`; nodes `_to_copy_1428_recomputed, sum_574_recomputed, linalg_vector_norm_114_recomputed, clamp_min_171_recomputed, div_172_recomputed, bitwise_not_115_recomputed, mul_981_recomputed, sum_575_recomputed`; source torchtitan/torchtitan/distributed/local_compile.py:96; torchtitan/torchtitan/models/common/moe/moe.py:597-606.
- Ops: _to_copy.default, bitwise_not.default, clamp_min.default, div.Tensor, linalg_vector_norm.default, mul.Tensor, reciprocal.default, sum.default, sum.dim_IntList. Shapes: float32[256], float32[], float32[98304, 256], float32[98304, 1].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #119, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T1: contains reductions (linalg_vector_norm, sum, sum.dim_IntList): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-021, G-052

### G-083 T3 GEMM epilogue (backward): attention.inner_attention.wkv_b: mm + _to_copy
- T3-epi; phases backward; scope intra-module; 61 instances; 2.0 kernels/instance; 1844 us measured vs 1398 us ideal per instance (6.610 GB external); gap 27.2 ms (0.07% of step).
- Where: `layers.*.attention.inner_attention.wkv_b`; nodes `mm_519, _to_copy_1485`; source torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: _to_copy.default, mm.default. Shapes: bfloat16[32768, 512], float32[32768, 512].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #120, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.9 ms | GEMM mm_519: 1831.6 us measured vs 1397.6 us flops-ideal (GEMM-own gap 26.5 ms of this region's gap); glue part 12.6 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 0.8 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-067, G-120

### G-084 Cross-layer: aux-loss scalar accumulation chained across layers
- XL; phases forward; scope cross-layer; 57 instances; 24.0 kernels/instance; 489 us measured vs 18 us ideal per instance (0.126 GB external); gap 26.8 ms (0.07% of step).
- Where: `layers.*.feed_forward.router.aux_loss`; nodes `clamp_min_2, reciprocal, mul_15, _to_copy_1164, sum_464, linalg_vector_norm_92, clamp_min_138, div_139`; source torchtitan/torchtitan/models/common/moe/moe.py:597-606; torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: _to_copy.default, add_.Tensor, bitwise_not.default, clamp_min.default, div.Tensor, linalg_vector_norm.default, mul.Tensor, reciprocal.default, sum.default, sum.dim_IntList. Shapes: float32[], float32[256], float32[98304, 256], int64[].
- Dataflow / notes: XL pair: two L1 components of one phase and stage joined by an edge across a layer/root-module boundary (candidate #121, XL). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.0 ms
- Blockers: crosses a decoder-layer boundary: the layer output is MUST_SAVE for backward (full-recompute policy saves layer outputs), so the fused kernel must still write it; graph passes treat layers as units (per-layer FSDP all-gather buckets and reordering), no compute dependency blocks it.
- Tier T1: contains reductions (linalg_vector_norm, sum, sum.dim_IntList): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-018, G-085, G-145

### G-085 Aux-loss forward statistics: casts, sums, norms, clamps (~19 tiny kernels)
- L1; phases forward; scope intra-module; 57 instances; 21.0 kernels/instance; 481 us measured vs 18 us ideal per instance (0.126 GB external); gap 26.4 ms (0.07% of step).
- Where: `layers.*.feed_forward.router.aux_loss`; nodes `_to_copy_84, sum_14, linalg_vector_norm_2, clamp_min_3, div_4, bitwise_not_3, mul_29, sum_15`; source torchtitan/torchtitan/distributed/local_compile.py:96; torchtitan/torchtitan/models/common/moe/moe.py:597-606.
- Ops: _to_copy.default, add_.Tensor, bitwise_not.default, clamp_min.default, div.Tensor, linalg_vector_norm.default, mul.Tensor, reciprocal.default, sum.default, sum.dim_IntList. Shapes: float32[256], float32[], float32[98304, 256], float32[98304, 1].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #122, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T1: contains reductions (linalg_vector_norm, sum, sum.dim_IntList): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-018, G-084

### G-086 w13 weight-grad fp32 cast + reduce-scatter bucket pack
- L1; phases backward; scope intra-module; 58 instances; 3.0 kernels/instance; 618 us measured vs 171 us ideal per instance (1.174 GB external); gap 25.9 ms (0.07% of step).
- Where: `layers.*.feed_forward.routed_experts.w13`; nodes `_to_copy_1477, _pre_bucket_reduce_scatter_119`; source torchtitan/torchtitan/models/common/linear/linear.py:437.
- Ops: _to_copy.default, bucketing._pre_bucket_reduce_scatter.default. Shapes: float32[4, 2, 2048, 7168], float32[176160768].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #123, L1).
- Blockers: the pack is a custom bucketing op (bucketing._pre_bucket_*); casts must be folded into it or into the producer's epilogue.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-017, G-080

### G-087 Router backward: score-normalization grads + scatter_add into [98304,256] + sigmoid_backward
- L1; phases backward; scope intra-module; 58 instances; 46.0 kernels/instance; 483 us measured vs 46 us ideal per instance (0.315 GB external); gap 25.4 ms (0.07% of step).
- Where: `layers.*.feed_forward.router`; nodes `mul_1012, neg_1, div_193, div_194, mul_1013, div_195, sum_583, add_4076`; source torchtitan/torchtitan/models/common/moe/moe.py:256-291.
- Ops: add.Tensor, div.Tensor, mul.Tensor, neg.default, new_zeros.default, scatter_add.default, sigmoid_backward.default, sum.dim_IntList. Shapes: float32[98304, 8], float32[98304, 256], float32[98304, 1].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #124, L1).
- Blockers: scatter_add accumulation order (atomics) must be fixed for bitwise results.
- Tier T2: order-sensitive data movement (scatter_add): accumulation / selection order must match eager exactly.
- Overlaps with: G-001, G-035

### G-088 Recomputed ffn_norm fused with its RMSNorm backward + residual-grad add
- RB; phases backward/recompute; scope intra-module, recompute-backward; 60 instances; 4.0 kernels/instance; 1647 us measured vs 1230 us ideal per instance (8.456 GB external); gap 25.0 ms (0.07% of step).
- Where: `layers.*.ffn_norm`; nodes `_fused_rms_norm_239_recomputed, _fused_rms_norm_backward_5, add_4087`; source torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm.default, _fused_rms_norm_backward.default, add.Tensor. Shapes: bfloat16[98304, 7168], float32[98304, 1], bfloat16[7168].
- Dataflow / notes: RB pair: a recompute L1 component feeding a backward L1 component at the same fusion stage (candidate #125, RB). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 12.3 ms
- Blockers: recompute duplicates are inserted by selective_activation_remat_pass right before their first backward consumer; fusing them into the backward kernel needs a remat-aware fusion step (no such pass in the default pipeline). RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T1: contains reductions (_fused_rms_norm, _fused_rms_norm_backward): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-054, G-092, G-105, G-112

### G-089 Recomputed attention_norm fused with its RMSNorm backward + residual-grad add
- RB; phases backward/recompute; scope intra-module, recompute-backward; 61 instances; 4.0 kernels/instance; 1637 us measured vs 1230 us ideal per instance (8.456 GB external); gap 24.8 ms (0.07% of step).
- Where: `layers.*.attention_norm`; nodes `_fused_rms_norm_240_recomputed, _fused_rms_norm_backward_4, add_4081`; source torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm.default, _fused_rms_norm_backward.default, add.Tensor. Shapes: bfloat16[98304, 7168], float32[98304, 1], bfloat16[7168].
- Dataflow / notes: RB pair: a recompute L1 component feeding a backward L1 component at the same fusion stage (candidate #126, RB). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 12.5 ms
- Blockers: recompute duplicates are inserted by selective_activation_remat_pass right before their first backward consumer; fusing them into the backward kernel needs a remat-aware fusion step (no such pass in the default pipeline). RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T1: contains reductions (_fused_rms_norm, _fused_rms_norm_backward): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-003, G-068, G-107, G-119

### G-090 T3 GEMM epilogue (backward): feed_forward.w13: mm + _to_copy + cat
- T3-epi; phases backward; scope intra-module; 3 instances; 6.0 kernels/instance; 29773 us measured vs 22012 us ideal per instance (9.714 GB external); gap 23.3 ms (0.06% of step).
- Where: `layers.*.feed_forward.w13`; nodes `mm_1555, _to_copy_2865, cat_589`; source torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: _to_copy.default, cat.default, mm.default. Shapes: bfloat16[36864, 7168], float32[2, 18432, 7168], float32[512, 72, 7168].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #127, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.7 ms | GEMM mm_1555: 28279.1 us measured vs 22011.7 us flops-ideal (GEMM-own gap 18.8 ms of this region's gap); glue part 1493.7 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 4.5 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-120, G-129

### G-091 Intra-module (recompute): feed_forward: silu + mul
- L1; phases recompute; scope intra-module; 3 instances; 8.0 kernels/instance; 9649 us measured vs 2108 us ideal per instance (14.496 GB external); gap 22.6 ms (0.06% of step).
- Where: `layers.*.feed_forward`; nodes `silu_2_recomputed, mul_8_recomputed`; source torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: mul.Tensor, silu.default. Shapes: bfloat16[98304, 18432].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #129, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.

### G-092 T3 GEMM epilogue (backward): feed_forward.w13, ffn_norm: mm + _fused_rms_norm_backward + add
- T3-epi; phases backward; scope cross-module; 3 instances; 4.0 kernels/instance; 29512 us measured vs 22012 us ideal per instance (13.414 GB external); gap 22.5 ms (0.06% of step).
- Where: `layers.*.feed_forward.w13, layers.*.ffn_norm`; nodes `mm_1556, _fused_rms_norm_backward_233, add_4538`; source torchtitan/torchtitan/models/common/linear/linear.py:115; torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm_backward.default, add.Tensor, mm.default. Shapes: bfloat16[98304, 7168], bfloat16[7168].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #131, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 3.1 ms | GEMM mm_1556: 28149.6 us measured vs 22011.7 us flops-ideal (GEMM-own gap 18.4 ms of this region's gap); glue part 1362.9 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 4.1 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops). RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-088, G-097, G-105

### G-093 Intra-module (forward): feed_forward: silu + mul
- L1; phases forward; scope intra-module; 3 instances; 8.0 kernels/instance; 9047 us measured vs 1581 us ideal per instance (10.872 GB external); gap 22.4 ms (0.06% of step).
- Where: `layers.*.feed_forward`; nodes `silu, mul_2`; source torchtitan/torchtitan/distributed/local_compile.py:96.
- Ops: mul.Tensor, silu.default. Shapes: bfloat16[98304, 18432].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #132, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-071

### G-094 T3 GEMM prologue (forward): feed_forward.shared_experts.w13: cat + mm
- T3-pro; phases forward; scope intra-module; 58 instances; 3.0 kernels/instance; 2816 us measured vs 2446 us ideal per instance (2.273 GB external); gap 21.5 ms (0.06% of step).
- Where: `layers.*.feed_forward.shared_experts.w13`; nodes `cat_11, mm_27`; source torchtitan/torchtitan/experiments/graph_trainer/simple_fsdp.py:366; torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: cat.default, mm.default. Shapes: bfloat16[2, 2048, 7168], bfloat16[98304, 4096].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #135, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 1.0 ms | GEMM mm_27: 2796.9 us measured vs 2445.7 us flops-ideal (GEMM-own gap 20.4 ms of this region's gap); glue part 19.6 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 1.1 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.

### G-095 T3 GEMM epilogue (backward/recompute): attention.q_norm, attention.wq_a: mm + _fused_rms_norm_backward
- T3-epi; phases backward/recompute; scope cross-module, recompute-backward; 61 instances; 3.0 kernels/instance; 1265 us measured vs 917 us ideal per instance (2.338 GB external); gap 21.2 ms (0.06% of step).
- Where: `layers.*.attention.q_norm, layers.*.attention.wq_a`; nodes `mm_477_recomputed, _fused_rms_norm_backward_3`; source torchtitan/torchtitan/models/common/linear/linear.py:115; torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm_backward.default, mm.default. Shapes: bfloat16[98304, 1536], bfloat16[1536].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #136, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 8.0 ms | GEMM mm_477_recomputed: 1096.4 us measured vs 917.2 us flops-ideal (GEMM-own gap 10.9 ms of this region's gap); glue part 168.8 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 10.3 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: recompute duplicates are inserted by selective_activation_remat_pass right before their first backward consumer; fusing them into the backward kernel needs a remat-aware fusion step (no such pass in the default pipeline). needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops). RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-060, G-101, G-124

### G-096 Routed + shared expert sum + residual add (two adds, cross-module)
- XM-max; phases forward; scope cross-module; 58 instances; 2.0 kernels/instance; 1174 us measured vs 820 us ideal per instance (5.637 GB external); gap 20.5 ms (0.05% of step).
- Where: `layers.*, layers.*.feed_forward`; nodes `add_74, add_75`; source torchtitan/torchtitan/models/common/moe/moe.py:825; torchtitan/torchtitan/models/deepseek_v3/model.py:70.
- Ops: add.Tensor. Shapes: bfloat16[98304, 7168].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #139, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 23.8 ms
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-004, G-058, G-113, G-141

### G-097 T3 GEMM prologue (backward/recompute): feed_forward.w13: cat + mm
- T3-pro; phases backward/recompute; scope intra-module, recompute-backward; 3 instances; 3.0 kernels/instance; 28310 us measured vs 22012 us ideal per instance (9.714 GB external); gap 18.9 ms (0.05% of step).
- Where: `layers.*.feed_forward.w13`; nodes `cat_8_recomputed, mm_1556`; source torchtitan/torchtitan/experiments/graph_trainer/simple_fsdp.py:366; torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: cat.default, mm.default. Shapes: bfloat16[2, 18432, 7168], bfloat16[98304, 7168].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #140, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.5 ms | GEMM mm_1556: 28149.6 us measured vs 22011.7 us flops-ideal (GEMM-own gap 18.4 ms of this region's gap); glue part 160.5 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 0.5 ms. | identical kernel set as candidate(s) [148] | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: recompute duplicates are inserted by selective_activation_remat_pass right before their first backward consumer; fusing them into the backward kernel needs a remat-aware fusion step (no such pass in the default pipeline). needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-092

### G-098 T3 GEMM epilogue (backward): lm_head: mm + _to_copy
- T3-epi; phases backward; scope intra-module; 8 instances; 3.0 kernels/instance; 11895 us measured vs 9649 us ideal per instance (7.060 GB external); gap 18.0 ms (0.05% of step).
- Where: `lm_head`; nodes `mm_486, _to_copy_1442`; source torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: _to_copy.default, mm.default. Shapes: bfloat16[129280, 7168], float32[129280, 7168].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #141, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 6.5 ms | GEMM mm_486: 11117.1 us measured vs 9649.2 us flops-ideal (GEMM-own gap 11.7 ms of this region's gap); glue part 778.2 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 6.2 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.

### G-099 Router gate hi/mid/lo split chain (casts, shifts, subs), backward
- L1; phases backward; scope intra-module; 58 instances; 10.0 kernels/instance; 337 us measured vs 37 us ideal per instance (0.252 GB external); gap 17.5 ms (0.05% of step).
- Where: `layers.*.feed_forward.router.gate`; nodes `_to_copy_1478, _to_copy_1479, lshift, sub_174, _to_copy_1480, _to_copy_1481, lshift_1, sub_175`; source torchtitan/torchtitan/models/common/linear/hi_mid_lo_linear.py:117.
- Ops: __lshift__.Scalar, _to_copy.default, cat.default, sub.Tensor. Shapes: int32[98304, 256], bfloat16[98304, 256], float32[98304, 256], bfloat16[98304, 768].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #142, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-001, G-051

### G-100 T3 GEMM epilogue (forward): attention.q_norm, attention.wq_a: mm + _fused_rms_norm
- T3-epi; phases forward; scope cross-module; 61 instances; 2.0 kernels/instance; 1186 us measured vs 917 us ideal per instance (1.734 GB external); gap 16.4 ms (0.04% of step).
- Where: `layers.*.attention.q_norm, layers.*.attention.wq_a`; nodes `mm, _fused_rms_norm_1`; source torchtitan/torchtitan/models/common/linear/linear.py:115; torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm.default, mm.default. Shapes: bfloat16[98304, 1536], float32[98304, 1].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #143, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 5.4 ms | GEMM mm: 1101.1 us measured vs 917.2 us flops-ideal (GEMM-own gap 11.2 ms of this region's gap); glue part 85.3 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 5.2 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-064, G-069

### G-101 T3 GEMM epilogue (recompute): attention.q_norm, attention.wq_a: mm + _fused_rms_norm
- T3-epi; phases recompute; scope cross-module; 61 instances; 2.0 kernels/instance; 1182 us measured vs 917 us ideal per instance (2.036 GB external); gap 16.1 ms (0.04% of step).
- Where: `layers.*.attention.q_norm, layers.*.attention.wq_a`; nodes `mm_477_recomputed, _fused_rms_norm_241_recomputed`; source torchtitan/torchtitan/models/common/linear/linear.py:115; torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm.default, mm.default. Shapes: bfloat16[98304, 1536], float32[98304, 1].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #144, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 5.4 ms | GEMM mm_477_recomputed: 1096.4 us measured vs 917.2 us flops-ideal (GEMM-own gap 10.9 ms of this region's gap); glue part 85.2 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 5.2 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-095, G-124

### G-102 T3 GEMM epilogue (backward): feed_forward.shared_experts.w2: mm + _to_copy
- T3-epi; phases backward; scope intra-module; 58 instances; 2.0 kernels/instance; 1467 us measured vs 1223 us ideal per instance (1.871 GB external); gap 14.1 ms (0.04% of step).
- Where: `layers.*.feed_forward.shared_experts.w2`; nodes `mm_509, _to_copy_1472`; source torchtitan/torchtitan/models/common/linear/linear.py:360.
- Ops: _to_copy.default, mm.default. Shapes: bfloat16[7168, 2048], float32[7168, 2048].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #145, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.7 ms | GEMM mm_509: 1455.7 us measured vs 1222.9 us flops-ideal (GEMM-own gap 13.5 ms of this region's gap); glue part 11.1 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 0.6 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-067

### G-103 T3 GEMM prologue (forward): feed_forward.w13, ffn_norm: _fused_rms_norm + mm
- T3-pro; phases forward; scope cross-module; 3 instances; 2.0 kernels/instance; 26629 us measured vs 22012 us ideal per instance (9.186 GB external); gap 13.9 ms (0.04% of step).
- Where: `layers.*.feed_forward.w13, layers.*.ffn_norm`; nodes `_fused_rms_norm_3, mm_5`; source torchtitan/torchtitan/models/common/nn_modules.py:155; torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: _fused_rms_norm.default, mm.default. Shapes: bfloat16[98304, 7168], float32[98304, 1], bfloat16[98304, 36864].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #146, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 1.2 ms | GEMM mm_5: 26230.6 us measured vs 22011.7 us flops-ideal (GEMM-own gap 12.7 ms of this region's gap); glue part 398.7 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 1.2 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-106, G-114

### G-104 T3 GEMM epilogue (backward): feed_forward.w2: mm + _to_copy
- T3-epi; phases backward; scope intra-module; 3 instances; 2.0 kernels/instance; 15498 us measured vs 11006 us ideal per instance (5.562 GB external); gap 13.5 ms (0.04% of step).
- Where: `layers.*.feed_forward.w2`; nodes `mm_1553, _to_copy_2864`; source torchtitan/torchtitan/models/common/linear/linear.py:360.
- Ops: _to_copy.default, mm.default. Shapes: bfloat16[7168, 18432], float32[7168, 18432].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #147, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.3 ms | GEMM mm_1553: 15387.5 us measured vs 11005.8 us flops-ideal (GEMM-own gap 13.1 ms of this region's gap); glue part 111.0 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 0.3 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-120

### G-105 Intra-module (backward): ffn_norm: _fused_rms_norm_backward + add
- L1; phases backward; scope intra-module; 60 instances; 3.0 kernels/instance; 1246 us measured vs 1025 us ideal per instance (7.047 GB external); gap 13.3 ms (0.04% of step).
- Where: `layers.*.ffn_norm`; nodes `_fused_rms_norm_backward_5, add_4087`; source torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm_backward.default, add.Tensor. Shapes: bfloat16[98304, 7168], bfloat16[7168].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #149, L1).
- Blockers: RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T1: contains reductions (_fused_rms_norm_backward): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-054, G-088, G-092

### G-106 T3 GEMM prologue (forward): feed_forward.w13: cat + mm
- T3-pro; phases forward; scope intra-module; 3 instances; 3.0 kernels/instance; 26391 us measured vs 22012 us ideal per instance (9.186 GB external); gap 13.1 ms (0.04% of step).
- Where: `layers.*.feed_forward.w13`; nodes `cat_2, mm_5`; source torchtitan/torchtitan/experiments/graph_trainer/simple_fsdp.py:366; torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: cat.default, mm.default. Shapes: bfloat16[2, 18432, 7168], bfloat16[98304, 36864].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #151, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.5 ms | GEMM mm_5: 26230.6 us measured vs 22011.7 us flops-ideal (GEMM-own gap 12.7 ms of this region's gap); glue part 160.0 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 0.5 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-103

### G-107 Intra-module (backward): attention_norm: _fused_rms_norm_backward + add
- L1; phases backward; scope intra-module; 61 instances; 3.0 kernels/instance; 1234 us measured vs 1025 us ideal per instance (7.047 GB external); gap 12.8 ms (0.03% of step).
- Where: `layers.*.attention_norm`; nodes `_fused_rms_norm_backward_4, add_4081`; source torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm_backward.default, add.Tensor. Shapes: bfloat16[98304, 7168], bfloat16[7168].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #152, L1).
- Blockers: RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T1: contains reductions (_fused_rms_norm_backward): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-003, G-068, G-089, G-119

### G-108 Attention weight-grad fp32 cast + reduce-scatter bucket pack
- L1; phases backward; scope intra-module; 61 instances; 3.0 kernels/instance; 494 us measured vs 289 us ideal per instance (1.985 GB external); gap 12.5 ms (0.03% of step).
- Where: `layers.*.attention.wq_a`; nodes `_to_copy_1494, _pre_bucket_reduce_scatter_118`; source torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: _to_copy.default, bucketing._pre_bucket_reduce_scatter.default. Shapes: float32[1536, 7168], float32[234373120].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #158, L1). merged signature variants (first/last-layer differences): candidates [217]
- Blockers: the pack is a custom bucketing op (bucketing._pre_bucket_*); casts must be folded into it or into the producer's epilogue.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-062, G-067, G-120

### G-109 Cross-module (backward): attention, attention.inner_attention, attention.kv_norm, attention.rope: sum + cat + _fused_rms_norm_backward + _to_copy + clone + mul
- XM-max; phases backward; scope cross-module; 61 instances; 20.0 kernels/instance; 2359 us measured vs 2172 us ideal per instance (14.936 GB external); gap 11.4 ms (0.03% of step).
- Where: `layers.*.attention, layers.*.attention.inner_attention, layers.*.attention.kv_norm, layers.*.attention.rope`; nodes `sum_584, cat_186, _fused_rms_norm_backward_2, _to_copy_1487, clone_117, mul_1014, _to_copy_1488, cat_188`; source torchtitan/torchtitan/models/common/attention/mla.py:119; torchtitan/torchtitan/models/common/nn_modules.py:155; torchtitan/torchtitan/models/common/rope.py:171; torchtitan/torchtitan/models/deepseek_v3/attention.py:133.
- Ops: _fused_rms_norm_backward.default, _to_copy.default, cat.default, clone.default, mul.Tensor, sum.dim_IntList. Shapes: bfloat16[98304, 1, 64], complex64[98304, 1, 32], bfloat16[98304, 128, 256], bfloat16[98304, 512].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #155, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 1.3 ms
- Blockers: RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T1: contains reductions (_fused_rms_norm_backward, sum.dim_IntList): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-028, G-029, G-042, G-078, G-117, G-122

### G-110 Intra-module (backward): feed_forward.shared_experts.w13: _to_copy + cat
- L1; phases backward; scope intra-module; 58 instances; 5.0 kernels/instance; 219 us measured vs 26 us ideal per instance (0.176 GB external); gap 11.2 ms (0.03% of step).
- Where: `layers.*.feed_forward.shared_experts.w13`; nodes `_to_copy_1473, cat_183`; source torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: _to_copy.default, cat.default. Shapes: float32[2, 2048, 7168], float32[512, 8, 7168].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #157, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-067, G-070

### G-111 MoE token counts: per-expert sums + cast before the count all_to_all
- L1; phases forward; scope intra-module; 58 instances; 5.0 kernels/instance; 185 us measured vs 4 us ideal per instance (0.025 GB external); gap 10.5 ms (0.03% of step).
- Where: `layers.*.feed_forward.routed_experts`; nodes `sum_7, sum_8, _to_copy_61`; source torchtitan/torchtitan/models/common/moe/token_dispatcher.py:294-394.
- Ops: _to_copy.default, sum.dim_IntList. Shapes: int64[64], int64[256].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #159, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T1: contains reductions (sum.dim_IntList): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-018, G-118

### G-112 Residual add after attention + ffn_norm RMSNorm (recompute copy)
- XM-max; phases recompute; scope cross-module; 61 instances; 2.0 kernels/instance; 987 us measured vs 820 us ideal per instance (5.638 GB external); gap 10.2 ms (0.03% of step).
- Where: `layers.*, layers.*.ffn_norm`; nodes `add_3996_recomputed, _fused_rms_norm_243_recomputed`; source torchtitan/torchtitan/models/deepseek_v3/model.py:59; torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm.default, add.Tensor. Shapes: bfloat16[98304, 7168], float32[98304, 1].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #161, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 12.5 ms
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T1: contains reductions (_fused_rms_norm): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-046, G-088, G-138

### G-113 Cross-layer: residual add (end of layer l) + attention_norm RMSNorm (layer l+1)
- XL; phases forward; scope cross-layer; 60 instances; 2.0 kernels/instance; 988 us measured vs 820 us ideal per instance (5.638 GB external); gap 10.1 ms (0.03% of step).
- Where: `layers.*, layers.*.attention_norm`; nodes `add_1, _fused_rms_norm_4`; source torchtitan/torchtitan/models/deepseek_v3/model.py:70; torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm.default, add.Tensor. Shapes: bfloat16[98304, 7168], float32[98304, 1].
- Dataflow / notes: XL pair: two L1 components of one phase and stage joined by an edge across a layer/root-module boundary (candidate #162, XL). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 12.3 ms
- Blockers: crosses a decoder-layer boundary: the layer output is MUST_SAVE for backward (full-recompute policy saves layer outputs), so the fused kernel must still write it; graph passes treat layers as units (per-layer FSDP all-gather buckets and reordering), no compute dependency blocks it.
- Tier T1: contains reductions (_fused_rms_norm): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-069, G-074, G-096, G-115

### G-114 Residual add after attention + ffn_norm RMSNorm (forward)
- XM-max; phases forward; scope cross-module; 61 instances; 2.0 kernels/instance; 982 us measured vs 820 us ideal per instance (5.638 GB external); gap 9.9 ms (0.03% of step).
- Where: `layers.*, layers.*.ffn_norm`; nodes `add, _fused_rms_norm_3`; source torchtitan/torchtitan/models/deepseek_v3/model.py:59; torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm.default, add.Tensor. Shapes: bfloat16[98304, 7168], float32[98304, 1].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #163, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 12.5 ms
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T1: contains reductions (_fused_rms_norm): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-047, G-103

### G-115 T3 GEMM epilogue (forward): layers.*, feed_forward.w2: mm + add
- T3-epi; phases forward; scope cross-module; 3 instances; 2.0 kernels/instance; 14207 us measured vs 11006 us ideal per instance (6.707 GB external); gap 9.6 ms (0.03% of step).
- Where: `layers.*, layers.*.feed_forward.w2`; nodes `mm_6, add_1`; source torchtitan/torchtitan/models/common/linear/linear.py:360; torchtitan/torchtitan/models/deepseek_v3/model.py:70.
- Ops: add.Tensor, mm.default. Shapes: bfloat16[98304, 7168].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #165, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 1.8 ms | GEMM mm_6: 13603.7 us measured vs 11005.8 us flops-ideal (GEMM-own gap 7.8 ms of this region's gap); glue part 603.7 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 1.8 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-071, G-113

### G-116 T3 GEMM epilogue (backward): attention.wkv_a: mm + _to_copy
- T3-epi; phases backward; scope intra-module; 61 instances; 3.0 kernels/instance; 493 us measured vs 344 us ideal per instance (1.539 GB external); gap 9.1 ms (0.02% of step).
- Where: `layers.*.attention.wkv_a`; nodes `mm_521, _to_copy_1491`; source torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: _to_copy.default, mm.default. Shapes: bfloat16[576, 7168], float32[576, 7168].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #166, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.2 ms | GEMM mm_521: 487.3 us measured vs 343.9 us flops-ideal (GEMM-own gap 8.7 ms of this region's gap); glue part 6.0 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 0.4 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-067, G-120

### G-117 T3 GEMM prologue (backward): attention, attention.wkv_a: cat + mm
- T3-pro; phases backward; scope cross-module; 61 instances; 2.0 kernels/instance; 491 us measured vs 344 us ideal per instance (1.644 GB external); gap 9.0 ms (0.02% of step).
- Where: `layers.*.attention, layers.*.attention.wkv_a`; nodes `cat_188, mm_522`; source torchtitan/torchtitan/models/deepseek_v3/attention.py:133; torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: cat.default, mm.default. Shapes: bfloat16[98304, 576], bfloat16[98304, 7168].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #167, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 2.0 ms | GEMM mm_522: 455.9 us measured vs 343.9 us flops-ideal (GEMM-own gap 6.8 ms of this region's gap); glue part 35.4 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 2.2 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-066, G-109

### G-118 Cross-module (forward): feed_forward.routed_experts, feed_forward.router, feed_forward.router.aux_loss: sigmoid + add + topk + sum + ones_like + scatter_.value + masked_fill.Scalar + gather + di
- XM-max; phases forward; scope cross-module; 1 instances; 79.0 kernels/instance; 10366 us measured vs 1864 us ideal per instance (12.820 GB external); gap 8.5 ms (0.02% of step).
- Where: `layers.*.feed_forward.routed_experts, layers.*.feed_forward.router, layers.*.feed_forward.router.aux_loss`; nodes `sigmoid, add_7, topk, sum_1, topk_1, ones_like, scatter_, masked_fill`; source torchtitan/torchtitan/models/common/moe/moe.py:256-606; torchtitan/torchtitan/distributed/local_compile.py:96; torchtitan/torchtitan/models/common/moe/token_dispatcher.py:294-403.
- Ops: _to_copy.default, add.Tensor, add_.Tensor, bitwise_and.Tensor, bitwise_not.default, clamp_min.default, div.Tensor, floor_divide.default, gather.default, index.Tensor, linalg_vector_norm.default, masked_fill.Scalar, mul.Tensor, ones_like.default, scatter_.value, sigmoid.default, sort.stable, sum.default, sum.dim_IntList, topk.default, zeros_like.default. Shapes: float32[256], float32[98304, 256], float32[98304, 8], float32[98304, 1].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #168, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.0 ms
- Blockers: top-k / stable sort selection: tie-breaking order must match; produces data-dependent indices. [786432,7168] tensors exceed 2^31 elements: eager TensorIterator splits each op into 16 launches (32-bit indexing).
- Tier T2: order-sensitive data movement (scatter_.value, sort.stable, topk): accumulation / selection order must match eager exactly.
- Overlaps with: G-022, G-023, G-033, G-111, G-135

### G-119 Cross-layer (backward): attention_norm, tok_embeddings: _fused_rms_norm_backward + add + embedding_dense_backward + _to_copy + bkt._pre_bucket_red
- XL; phases backward; scope cross-layer; 1 instances; 28.0 kernels/instance; 8676 us measured vs 1359 us ideal per instance (9.345 GB external); gap 7.3 ms (0.02% of step).
- Where: `layers.*.attention_norm, tok_embeddings`; nodes `_fused_rms_norm_backward_244, add_4546, embedding_dense_backward, _to_copy_2909, _pre_bucket_reduce_scatter`; source torchtitan/torchtitan/models/common/nn_modules.py:155; torchtitan/torchtitan/models/common/embedding.py:41.
- Ops: _fused_rms_norm_backward.default, _to_copy.default, add.Tensor, embedding_dense_backward.default, bucketing._pre_bucket_reduce_scatter.default. Shapes: bfloat16[98304, 7168], bfloat16[7168], bfloat16[129280, 7168], float32[129280, 7168].
- Dataflow / notes: XL pair: two L1 components of one phase and stage joined by an edge across a layer/root-module boundary (candidate #169, XL). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.4 ms
- Blockers: crosses a decoder-layer boundary in backward: the residual gradient handed from layer l+1 to layer l is also read by layer l's other dgrad branch, so the fused kernel must still write it; graph passes treat layers as units (per-layer FSDP reduce-scatter buckets), no compute dependency blocks it. the pack is a custom bucketing op (bucketing._pre_bucket_*); casts must be folded into it or into the producer's epilogue. RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T2: order-sensitive data movement (embedding_dense_backward): accumulation / selection order must match eager exactly.
- Overlaps with: G-068, G-089, G-107, G-121

### G-120 Cross-module (backward): attention.inner_attention.wkv_b, attention.kv_norm, attention.q_norm, attention.: _to_copy + cat + bkt._pre_bucket_reduce_scatter
- XM-max; phases backward; scope cross-module; 3 instances; 17.0 kernels/instance; 2874 us measured vs 510 us ideal per instance (3.506 GB external); gap 7.1 ms (0.02% of step).
- Where: `layers.*.attention.inner_attention.wkv_b, layers.*.attention.kv_norm, layers.*.attention.q_norm, layers.*.attention.wkv_a, layers.*.attention.wo, layers.*.attention.wq_a, layers.*.attention.wq_b, layers.*.attention_norm, layers.*.feed_forward.w13, layers.*.feed_forward.w2, layers.*.ffn_norm`; nodes `_to_copy_2864, _to_copy_2865, cat_589, _to_copy_2866, _to_copy_2867, _to_copy_2868, _to_copy_2869, _to_copy_2874`; source torchtitan/torchtitan/models/common/linear/linear.py:115-360; torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _to_copy.default, cat.default, bucketing._pre_bucket_reduce_scatter.default. Shapes: float32[7168], float32[7168, 18432], float32[2, 18432, 7168], float32[512, 72, 7168].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #170, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 2.0 ms
- Blockers: the pack is a custom bucketing op (bucketing._pre_bucket_*); casts must be folded into it or into the producer's epilogue.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-039, G-062, G-081, G-083, G-090, G-104, G-108, G-116, G-129

### G-121 Embedding backward (dense scatter-add) + fp32 cast + bucket pack
- L1; phases backward; scope intra-module; 1 instances; 25.0 kernels/instance; 7437 us measured vs 744 us ideal per instance (5.117 GB external); gap 6.7 ms (0.02% of step).
- Where: `tok_embeddings`; nodes `embedding_dense_backward, _to_copy_2909, _pre_bucket_reduce_scatter`; source torchtitan/torchtitan/models/common/embedding.py:41.
- Ops: _to_copy.default, embedding_dense_backward.default, bucketing._pre_bucket_reduce_scatter.default. Shapes: bfloat16[129280, 7168], float32[129280, 7168], float32[926679040].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #174, L1).
- Blockers: the pack is a custom bucketing op (bucketing._pre_bucket_*); casts must be folded into it or into the producer's epilogue.
- Tier T2: order-sensitive data movement (embedding_dense_backward): accumulation / selection order must match eager exactly.
- Overlaps with: G-119

### G-122 Recompute->backward (backward/recompute): attention.kv_norm: _fused_rms_norm + _fused_rms_norm_backward
- RB; phases backward/recompute; scope intra-module, recompute-backward; 61 instances; 5.0 kernels/instance; 159 us measured vs 59 us ideal per instance (0.403 GB external); gap 6.1 ms (0.02% of step).
- Where: `layers.*.attention.kv_norm`; nodes `_fused_rms_norm_242_recomputed, _fused_rms_norm_backward_2`; source torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm.default, _fused_rms_norm_backward.default. Shapes: bfloat16[98304, 512], float32[98304, 1], bfloat16[512].
- Dataflow / notes: RB pair: a recompute L1 component feeding a backward L1 component at the same fusion stage (candidate #175, RB). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.9 ms
- Blockers: recompute duplicates are inserted by selective_activation_remat_pass right before their first backward consumer; fusing them into the backward kernel needs a remat-aware fusion step (no such pass in the default pipeline). RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T1: contains reductions (_fused_rms_norm, _fused_rms_norm_backward): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-078, G-109, G-125

### G-123 T3 GEMM epilogue (backward): feed_forward.router.gate: mm.dtype + add_
- T3-epi; phases backward; scope intra-module; 116 instances; 4.0 kernels/instance; 260 us measured vs 217 us ideal per instance (1.489 GB external); gap 5.1 ms (0.01% of step).
- Where: `layers.*.feed_forward.router.gate`; nodes `mm_514, add__123, add__124`; source torchtitan/torchtitan/models/common/linear/hi_mid_lo_linear.py:117.
- Ops: add_.Tensor, mm.dtype. Shapes: float32[256, 7168].
- Dataflow / notes: T3 epilogue: L1 component consuming a GEMM output (candidate #177, T3-epi). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.0 ms | GEMM mm_514: 252.8 us measured vs 152.9 us flops-ideal (GEMM-own gap 11.6 ms of this region's gap); glue part 7.5 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 0.9 ms. | identical kernel set as candidate(s) [188] | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-051, G-067, G-143

### G-124 Recompute->backward (backward/recompute): attention.q_norm: _fused_rms_norm + _fused_rms_norm_backward
- RB; phases backward/recompute; scope intra-module, recompute-backward; 61 instances; 3.0 kernels/instance; 254 us measured vs 176 us ideal per instance (1.208 GB external); gap 4.8 ms (0.01% of step).
- Where: `layers.*.attention.q_norm`; nodes `_fused_rms_norm_241_recomputed, _fused_rms_norm_backward_3`; source torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm.default, _fused_rms_norm_backward.default. Shapes: bfloat16[98304, 1536], float32[98304, 1], bfloat16[1536].
- Dataflow / notes: RB pair: a recompute L1 component feeding a backward L1 component at the same fusion stage (candidate #178, RB). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 2.7 ms
- Blockers: recompute duplicates are inserted by selective_activation_remat_pass right before their first backward consumer; fusing them into the backward kernel needs a remat-aware fusion step (no such pass in the default pipeline). RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T1: contains reductions (_fused_rms_norm, _fused_rms_norm_backward): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-060, G-095, G-101

### G-125 Cross-module (recompute): attention.inner_attention, attention.kv_norm, attention.rope: _to_copy + mul + _fused_rms_norm + cat
- XM-max; phases recompute; scope cross-module; 61 instances; 13.0 kernels/instance; 1284 us measured vs 1206 us ideal per instance (8.293 GB external); gap 4.8 ms (0.01% of step).
- Where: `layers.*.attention.inner_attention, layers.*.attention.kv_norm, layers.*.attention.rope`; nodes `_to_copy_1421_recomputed, mul_979_recomputed, _to_copy_1422_recomputed, _fused_rms_norm_242_recomputed, cat_181_recomputed`; source torchtitan/torchtitan/models/common/rope.py:171; torchtitan/torchtitan/models/common/nn_modules.py:155; torchtitan/torchtitan/models/common/attention/mla.py:119.
- Ops: _fused_rms_norm.default, _to_copy.default, cat.default, mul.Tensor. Shapes: float32[98304, 1, 64], complex64[98304, 1, 32], bfloat16[98304, 1, 64], bfloat16[98304, 512].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #179, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.2 ms
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T1: contains reductions (_fused_rms_norm): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-030, G-045, G-122

### G-126 Cross-module (forward): attention.inner_attention, attention.kv_norm, attention.rope: _to_copy + mul + _fused_rms_norm + cat
- XM-max; phases forward; scope cross-module; 61 instances; 13.0 kernels/instance; 1284 us measured vs 1206 us ideal per instance (8.293 GB external); gap 4.7 ms (0.01% of step).
- Where: `layers.*.attention.inner_attention, layers.*.attention.kv_norm, layers.*.attention.rope`; nodes `_to_copy_8, mul_1, _to_copy_9, _fused_rms_norm_2, cat_1`; source torchtitan/torchtitan/models/common/rope.py:171; torchtitan/torchtitan/models/common/nn_modules.py:155; torchtitan/torchtitan/models/common/attention/mla.py:119.
- Ops: _fused_rms_norm.default, _to_copy.default, cat.default, mul.Tensor. Shapes: float32[98304, 1, 64], complex64[98304, 1, 32], bfloat16[98304, 1, 64], bfloat16[98304, 512].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #180, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.2 ms
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T1: contains reductions (_fused_rms_norm): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-031, G-044, G-072

### G-127 T3 GEMM prologue (backward): feed_forward.router.gate: cat + mm
- T3-pro; phases backward; scope intra-module; 58 instances; 2.0 kernels/instance; 527 us measured vs 459 us ideal per instance (1.564 GB external); gap 4.0 ms (0.01% of step).
- Where: `layers.*.feed_forward.router.gate`; nodes `cat_185, mm_513`; source torchtitan/torchtitan/models/common/linear/hi_mid_lo_linear.py:117.
- Ops: cat.default, mm.default. Shapes: bfloat16[768, 7168], bfloat16[98304, 7168].
- Dataflow / notes: T3 prologue: L1 component whose outputs feed only GEMMs (candidate #182, T3-pro). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.1 ms | GEMM mm_513: 522.1 us measured vs 458.6 us flops-ideal (GEMM-own gap 3.7 ms of this region's gap); glue part 5.0 us/instance. Fusion-attributable gap (glue time not hidden behind the GEMM's measured time) = 0.3 ms. | estimated: ideal uses flops/peak for the GEMM (zero-operand peak, see setup)
- Blockers: needs a GEMM with a custom prologue/epilogue (grouped GEMM: CUTLASS/Triton template; cuBLASLt epilogues do not cover these ops).
- Tier T3: contains a GEMM/attention kernel: bitwise identity requires the same GEMM algorithm/split-K as eager plus an epilogue that rounds at the same points.
- Overlaps with: G-065

### G-128 Intra-module (recompute): feed_forward.routed_experts: arange + repeat_interleave + index_select + index + add + sub
- L1; phases recompute; scope intra-module; 58 instances; 10.0 kernels/instance; 67 us measured vs 1 us ideal per instance (0.006 GB external); gap 3.8 ms (0.01% of step).
- Where: `layers.*.feed_forward.routed_experts`; nodes `arange_114_recomputed, repeat_interleave_57_recomputed, index_select_57_recomputed, index_348_recomputed, arange_115_recomputed, add_3999_recomputed, index_349_recomputed, sub_173_recomputed`; source torchtitan/torchtitan/models/common/moe/token_dispatcher.py:403.
- Ops: add.Tensor, arange.default, index.Tensor, index_select.default, repeat_interleave.Tensor, sub.Tensor. Shapes: int64[N_routed], int64[256].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #184, L1).
- Blockers: routed-token dimension is data-dependent (unbacked SymInts from .item() on all_to_all split sizes): kernels need dynamic shapes.
- Tier T2: order-sensitive data movement (repeat_interleave.Tensor): accumulation / selection order must match eager exactly.

### G-129 Intra-module (backward): feed_forward.w13: _to_copy + cat
- L1; phases backward; scope intra-module; 3 instances; 5.0 kernels/instance; 1494 us measured vs 231 us ideal per instance (1.585 GB external); gap 3.8 ms (0.01% of step).
- Where: `layers.*.feed_forward.w13`; nodes `_to_copy_2865, cat_589`; source torchtitan/torchtitan/models/common/linear/linear.py:115.
- Ops: _to_copy.default, cat.default. Shapes: float32[2, 18432, 7168], float32[512, 72, 7168].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #185, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-090, G-120

### G-130 Intra-module (forward): feed_forward.routed_experts: arange + repeat_interleave + index_select + index + add + sub
- L1; phases forward; scope intra-module; 58 instances; 10.0 kernels/instance; 64 us measured vs 2 us ideal per instance (0.013 GB external); gap 3.6 ms (0.01% of step).
- Where: `layers.*.feed_forward.routed_experts`; nodes `arange, repeat_interleave, index_select, index_6, arange_1, add_9, index_7, sub_2`; source torchtitan/torchtitan/models/common/moe/token_dispatcher.py:403.
- Ops: add.Tensor, arange.default, index.Tensor, index_select.default, repeat_interleave.Tensor, sub.Tensor. Shapes: int64[N_routed], int64[256].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #186, L1).
- Blockers: routed-token dimension is data-dependent (unbacked SymInts from .item() on all_to_all split sizes): kernels need dynamic shapes.
- Tier T2: order-sensitive data movement (repeat_interleave.Tensor): accumulation / selection order must match eager exactly.

### G-131 Intra-module (forward): feed_forward.routed_experts: sum + _to_copy + cumsum + sub + clone
- L1; phases forward; scope intra-module; 58 instances; 14.0 kernels/instance; 31 us measured vs 0 us ideal per instance (0.000 GB external); gap 1.8 ms (0.01% of step).
- Where: `layers.*.feed_forward.routed_experts`; nodes `sum_9, _to_copy_62, cumsum, sub, clone, clone_1, cumsum_1, sub_1`; source torchtitan/torchtitan/models/common/moe/token_dispatcher.py:300-403; torchtitan/torchtitan/models/common/moe/moe.py:136.
- Ops: _to_copy.default, clone.default, cumsum.default, sub.Tensor, sum.dim_IntList. Shapes: int64[256], int64[64], int64[4, 64], int64[4].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #196, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T1: contains reductions (cumsum, sum.dim_IntList): the fused kernel must keep eager's reduction order/tree to be bitwise identical.

### G-132 Recompute->backward (backward/recompute): attention.wkv_a, attention.wq_a: _to_copy + constant_pad_nd + bkt._pre_bucket_all_gather
- RB; phases backward/recompute; scope cross-module, recompute-backward; 61 instances; 4.0 kernels/instance; 31 us measured vs 1 us ideal per instance (0.008 GB external); gap 1.8 ms (0.01% of step).
- Where: `layers.*.attention.wkv_a, layers.*.attention.wq_a`; nodes `_to_copy_1418_recomputed, constant_pad_nd_60_recomputed, _pre_bucket_all_gather_236`; source torchtitan/torchtitan/experiments/graph_trainer/simple_fsdp.py:366.
- Ops: _to_copy.default, constant_pad_nd.default, bucketing._pre_bucket_all_gather.default. Shapes: bfloat16[3, 7168], bfloat16[234373120].
- Dataflow / notes: RB pair: a recompute L1 component feeding a backward L1 component at the same fusion stage (candidate #197, RB). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.0 ms | merged signature variants (first/last-layer differences): candidates [256]
- Blockers: recompute duplicates are inserted by selective_activation_remat_pass right before their first backward consumer; fusing them into the backward kernel needs a remat-aware fusion step (no such pass in the default pipeline). the pack is a custom bucketing op (bucketing._pre_bucket_*); casts must be folded into it or into the producer's epilogue.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-137

### G-133 Cross-module (forward): attention.wkv_a, attention.wq_a: _to_copy + constant_pad_nd + bkt._pre_bucket_all_gather
- XM-max; phases forward; scope cross-module; 61 instances; 4.0 kernels/instance; 30 us measured vs 1 us ideal per instance (0.008 GB external); gap 1.7 ms (0.01% of step).
- Where: `layers.*.attention.wkv_a, layers.*.attention.wq_a`; nodes `_to_copy_50, constant_pad_nd_3, _pre_bucket_all_gather_7`; source torchtitan/torchtitan/experiments/graph_trainer/simple_fsdp.py:366.
- Ops: _to_copy.default, constant_pad_nd.default, bucketing._pre_bucket_all_gather.default. Shapes: bfloat16[3, 7168], bfloat16[234373120].
- Dataflow / notes: XM-max component: L1 components of one phase/layer/stage joined across module FQNs (barrier-free edges) (candidate #199, XM-max). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.0 ms | merged signature variants (first/last-layer differences): candidates [258]
- Blockers: the pack is a custom bucketing op (bucketing._pre_bucket_*); casts must be folded into it or into the producer's epilogue.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-139

### G-134 Intra-module (recompute): feed_forward.routed_experts: cumsum + sub + clone + sum
- L1; phases recompute; scope intra-module; 58 instances; 12.0 kernels/instance; 28 us measured vs 0 us ideal per instance (0.000 GB external); gap 1.6 ms (0.00% of step).
- Where: `layers.*.feed_forward.routed_experts`; nodes `cumsum_171_recomputed, sub_171_recomputed, clone_114_recomputed, clone_115_recomputed, cumsum_172_recomputed, sub_172_recomputed, sum_580_recomputed, cumsum_173_recomputed`; source torchtitan/torchtitan/models/common/moe/token_dispatcher.py:403; torchtitan/torchtitan/models/common/moe/moe.py:136.
- Ops: clone.default, cumsum.default, sub.Tensor, sum.dim_IntList. Shapes: int64[256], int64[4, 64], int64[4], int32[4].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #200, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T1: contains reductions (cumsum, sum.dim_IntList): the fused kernel must keep eager's reduction order/tree to be bitwise identical.

### G-135 Intra-module (forward): feed_forward.router.aux_loss: _to_copy + sum + linalg_vector_norm + clamp_min + div + bitwise_not + mul + add_
- L1; phases forward; scope intra-module; 1 instances; 18.0 kernels/instance; 514 us measured vs 18 us ideal per instance (0.126 GB external); gap 0.5 ms (0.00% of step).
- Where: `layers.*.feed_forward.router.aux_loss`; nodes `_to_copy_60, sum_4, linalg_vector_norm, clamp_min, div_1, bitwise_not_1, mul_12, sum_5`; source torchtitan/torchtitan/distributed/local_compile.py:96; torchtitan/torchtitan/models/common/moe/moe.py:597-606.
- Ops: _to_copy.default, add_.Tensor, bitwise_not.default, clamp_min.default, div.Tensor, linalg_vector_norm.default, mul.Tensor, sum.default, sum.dim_IntList. Shapes: float32[256], float32[98304, 256], float32[], float32[98304, 1].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #228, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T1: contains reductions (linalg_vector_norm, sum, sum.dim_IntList): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-118

### G-136 Recompute->backward (backward/recompute): norm: _fused_rms_norm + _fused_rms_norm_backward
- RB; phases backward/recompute; scope intra-module, recompute-backward; 1 instances; 3.0 kernels/instance; 1108 us measured vs 615 us ideal per instance (4.228 GB external); gap 0.5 ms (0.00% of step).
- Where: `norm`; nodes `_fused_rms_norm_244_recomputed, _fused_rms_norm_backward`; source torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm.default, _fused_rms_norm_backward.default. Shapes: bfloat16[98304, 7168], float32[98304, 1], bfloat16[7168].
- Dataflow / notes: RB pair: a recompute L1 component feeding a backward L1 component at the same fusion stage (candidate #229, RB). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.4 ms
- Blockers: recompute duplicates are inserted by selective_activation_remat_pass right before their first backward consumer; fusing them into the backward kernel needs a remat-aware fusion step (no such pass in the default pipeline). RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T1: contains reductions (_fused_rms_norm, _fused_rms_norm_backward): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-057

### G-137 Intra-module (recompute): attention.wkv_a: _to_copy + constant_pad_nd
- L1; phases recompute; scope intra-module; 61 instances; 2.0 kernels/instance; 4 us measured vs 0 us ideal per instance (0.000 GB external); gap 0.2 ms (0.00% of step).
- Where: `layers.*.attention.wkv_a`; nodes `_to_copy_1418_recomputed, constant_pad_nd_60_recomputed`; source torchtitan/torchtitan/experiments/graph_trainer/simple_fsdp.py:366.
- Ops: _to_copy.default, constant_pad_nd.default. Shapes: bfloat16[3, 7168].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #237, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-132

### G-138 Recompute->backward (backward/recompute): ffn_norm: _fused_rms_norm + _fused_rms_norm_backward
- RB; phases backward/recompute; scope intra-module, recompute-backward; 1 instances; 3.0 kernels/instance; 1043 us measured vs 820 us ideal per instance (5.638 GB external); gap 0.2 ms (0.00% of step).
- Where: `layers.*.ffn_norm`; nodes `_fused_rms_norm_243_recomputed, _fused_rms_norm_backward_1`; source torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm.default, _fused_rms_norm_backward.default. Shapes: bfloat16[98304, 7168], float32[98304, 1], bfloat16[7168].
- Dataflow / notes: RB pair: a recompute L1 component feeding a backward L1 component at the same fusion stage (candidate #238, RB). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.2 ms
- Blockers: recompute duplicates are inserted by selective_activation_remat_pass right before their first backward consumer; fusing them into the backward kernel needs a remat-aware fusion step (no such pass in the default pipeline). RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T1: contains reductions (_fused_rms_norm, _fused_rms_norm_backward): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-054, G-112, G-140

### G-139 Intra-module (forward): attention.wkv_a: _to_copy + constant_pad_nd
- L1; phases forward; scope intra-module; 61 instances; 2.0 kernels/instance; 4 us measured vs 0 us ideal per instance (0.000 GB external); gap 0.2 ms (0.00% of step).
- Where: `layers.*.attention.wkv_a`; nodes `_to_copy_5, constant_pad_nd`; source torchtitan/torchtitan/experiments/graph_trainer/simple_fsdp.py:366.
- Ops: _to_copy.default, constant_pad_nd.default. Shapes: bfloat16[3, 7168].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #239, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-133

### G-140 Cross-layer (backward): ffn_norm, norm: _fused_rms_norm_backward + add
- XL; phases backward; scope cross-layer; 1 instances; 3.0 kernels/instance; 1227 us measured vs 1025 us ideal per instance (7.047 GB external); gap 0.2 ms (0.00% of step).
- Where: `layers.*.ffn_norm, norm`; nodes `_fused_rms_norm_backward_1, add_4079`; source torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm_backward.default, add.Tensor. Shapes: bfloat16[98304, 7168], bfloat16[7168].
- Dataflow / notes: XL pair: two L1 components of one phase and stage joined by an edge across a layer/root-module boundary (candidate #240, XL). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.2 ms
- Blockers: crosses a decoder-layer boundary in backward: the residual gradient handed from layer l+1 to layer l is also read by layer l's other dgrad branch, so the fused kernel must still write it; graph passes treat layers as units (per-layer FSDP reduce-scatter buckets), no compute dependency blocks it. RMSNorm backward includes a weight-grad column reduction over all 98304 rows.
- Tier T1: contains reductions (_fused_rms_norm_backward): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-054, G-138

### G-141 Cross-layer (forward): layers.*, norm: add + _fused_rms_norm
- XL; phases forward; scope cross-layer; 1 instances; 2.0 kernels/instance; 981 us measured vs 820 us ideal per instance (5.638 GB external); gap 0.2 ms (0.00% of step).
- Where: `layers.*, norm`; nodes `add_4065, _fused_rms_norm_244`; source torchtitan/torchtitan/models/deepseek_v3/model.py:70; torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm.default, add.Tensor. Shapes: bfloat16[98304, 7168], float32[98304, 1].
- Dataflow / notes: XL pair: two L1 components of one phase and stage joined by an edge across a layer/root-module boundary (candidate #243, XL). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.2 ms
- Blockers: crosses a decoder-layer boundary: the layer output is MUST_SAVE for backward (full-recompute policy saves layer outputs), so the fused kernel must still write it; graph passes treat layers as units (per-layer FSDP all-gather buckets and reordering), no compute dependency blocks it.
- Tier T1: contains reductions (_fused_rms_norm): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-096

### G-142 Cross-layer (forward): attention_norm, tok_embeddings: embedding + _fused_rms_norm
- XL; phases forward; scope cross-layer; 1 instances; 2.0 kernels/instance; 819 us measured vs 680 us ideal per instance (4.673 GB external); gap 0.1 ms (0.00% of step).
- Where: `layers.*.attention_norm, tok_embeddings`; nodes `embedding, _fused_rms_norm`; source torchtitan/torchtitan/models/common/embedding.py:41; torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _fused_rms_norm.default, embedding.default. Shapes: bfloat16[98304, 7168], float32[98304, 1].
- Dataflow / notes: XL pair: two L1 components of one phase and stage joined by an edge across a layer/root-module boundary (candidate #248, XL). incremental gap over its parts (bytes of intermediates no longer materialized, at HBM roofline) = 0.2 ms
- Blockers: crosses a decoder-layer boundary: the layer output is MUST_SAVE for backward (full-recompute policy saves layer outputs), so the fused kernel must still write it; graph passes treat layers as units (per-layer FSDP all-gather buckets and reordering), no compute dependency blocks it.
- Tier T1: contains reductions (_fused_rms_norm): the fused kernel must keep eager's reduction order/tree to be bitwise identical.
- Overlaps with: G-069, G-074

### G-143 Intra-module (backward): feed_forward.router.gate: add_
- L1; phases backward; scope intra-module; 58 instances; 2.0 kernels/instance; 8 us measured vs 5 us ideal per instance (0.037 GB external); gap 0.1 ms (0.00% of step).
- Where: `layers.*.feed_forward.router.gate`; nodes `add__123, add__124`; source torchtitan/torchtitan/models/common/linear/hi_mid_lo_linear.py:117.
- Ops: add_.Tensor. Shapes: float32[256, 7168].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #249, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-067, G-123

### G-144 Intra-module (backward): loss: add_
- L1; phases backward; scope intra-module; 1 instances; 7.0 kernels/instance; 44 us measured vs 32 us ideal per instance (0.217 GB external); gap 0.0 ms (0.00% of step).
- Where: `loss`; nodes `add__116, add__117, add__118, add__119, add__120, add__121, add__122`; source (no torchtitan frame in stack_trace).
- Ops: add_.Tensor. Shapes: float32[505, 7168].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #271, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.

### G-145 Intra-module (forward): feed_forward.router.aux_loss: clamp_min + reciprocal + mul
- L1; phases forward; scope intra-module; 1 instances; 3.0 kernels/instance; 7 us measured vs 0 us ideal per instance (0.000 GB external); gap 0.0 ms (0.00% of step).
- Where: `layers.*.feed_forward.router.aux_loss`; nodes `clamp_min_2, reciprocal, mul_15`; source torchtitan/torchtitan/models/common/moe/moe.py:606.
- Ops: clamp_min.default, mul.Tensor, reciprocal.default. Shapes: float32[], int64[].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #272, L1).
- Blockers: none structural: same phase, same fusion stage, no barrier op; standard Inductor-style pointwise/reduction fusion applies.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.
- Overlaps with: G-084

### G-146 Intra-module (backward): norm: _to_copy + bkt._pre_bucket_reduce_scatter
- L1; phases backward; scope intra-module; 1 instances; 3.0 kernels/instance; 7 us measured vs 0 us ideal per instance (0.000 GB external); gap 0.0 ms (0.00% of step).
- Where: `norm`; nodes `_to_copy_1471, _pre_bucket_reduce_scatter_120`; source torchtitan/torchtitan/models/common/nn_modules.py:155.
- Ops: _to_copy.default, bucketing._pre_bucket_reduce_scatter.default. Shapes: float32[7168].
- Dataflow / notes: L1 component: glue nodes joined by tensor edges within one phase, layer, module FQN and fusion stage (candidate #273, L1).
- Blockers: the pack is a custom bucketing op (bucketing._pre_bucket_*); casts must be folded into it or into the producer's epilogue.
- Tier T0: pointwise/views/casts only: elementwise results are bitwise reproducible if the same fp32 math and rounding points are kept.

## 5. No opportunity

Memory-bound node classes (op target, phase, module pattern) not covered by any region: 68.0 ms in total (includes the eager optimizer kernels outside the graph).

| what | ms | reason |
|---|---:|---|
| optimizer step + grad-norm clipping (eager, outside the traced graph) | 16.84 | runs outside the FX graph (fused AdamW multi_tensor_apply + foreach norm/mul); invisible to this substrate's graph, already multi-tensor fused. |
| other eager kernels outside the graph (loss logging, grad clones) | 10.99 | outside the traced graph; not attributable to FX nodes. |
| split_with_sizes_copy.out [forward] layers.*.attention.wq_a (x61) | 9.41 | FSDP all-gather unpack: single copy kernel straight after the (fake) collective; consumers are GEMMs. Could only be removed by gathering into the final layout (comm-side change), not by fusion. |
| split_with_sizes_copy.out [backward] layers.*.attention.wq_a (x61) | 9.39 | FSDP all-gather unpack: single copy kernel straight after the (fake) collective; consumers are GEMMs. Could only be removed by gathering into the final layout (comm-side change), not by fusion. |
| split_with_sizes_copy.out [forward] layers.*.feed_forward.routed_experts.w13 (x58) | 6.42 | FSDP all-gather unpack: single copy kernel straight after the (fake) collective; consumers are GEMMs. Could only be removed by gathering into the final layout (comm-side change), not by fusion. |
| split_with_sizes_copy.out [backward] layers.*.feed_forward.routed_experts.w13 (x58) | 6.41 | FSDP all-gather unpack: single copy kernel straight after the (fake) collective; consumers are GEMMs. Could only be removed by gathering into the final layout (comm-side change), not by fusion. |
| bkt._pre_bucket_all_gather [forward] layers.*.feed_forward.routed_experts.w13 (x58) | 3.09 | FSDP all-gather pack of parameter shards (single copy kernel feeding the fake all_gather); its producers are placeholders, nothing to fuse in-graph. |
| bkt._pre_bucket_all_gather [backward] layers.*.feed_forward.routed_experts.w13 (x58) | 3.08 | FSDP all-gather pack of parameter shards (single copy kernel feeding the fake all_gather); its producers are placeholders, nothing to fuse in-graph. |
| index.Tensor [forward] layers.*.attention.rope (x61) | 1.07 | RoPE frequency gather (freqs_cis rows by position) on graph inputs, read by both the q and k RoPE groups at different fusion stages; could be folded into each RoPE kernel (load freqs by position), not into one group. |
| split_with_sizes_copy.out [forward] norm (x1) | 0.56 | FSDP all-gather unpack: single copy kernel straight after the (fake) collective; consumers are GEMMs. Could only be removed by gathering into the final layout (comm-side change), not by fusion. |
| copy_ [forward] loss (x8) | 0.45 | single kernel with no same-stage glue neighbour (neighbours are GEMM/attention/collective); 0.72x of its roofline. |
| zeros_like [forward] loss (x1) | 0.21 | single kernel with no same-stage glue neighbour (neighbours are GEMM/attention/collective); 0.51x of its roofline. |
| bkt._pre_bucket_all_gather [forward] norm (x1) | 0.02 | FSDP all-gather pack of parameter shards (single copy kernel feeding the fake all_gather); its producers are placeholders, nothing to fuse in-graph. |
| bkt._pre_bucket_all_gather [forward] tok_embeddings (x1) | 0.02 | FSDP all-gather pack of parameter shards (single copy kernel feeding the fake all_gather); its producers are placeholders, nothing to fuse in-graph. |
| ones_like [forward] loss (x8) | 0.01 | tiny launch-bound kernel (1.6 us each, ~0 bytes) with no same-stage glue neighbour. |
| bkt._pre_bucket_all_gather [backward] norm (x1) | 0.00 | tiny launch-bound kernel (2.4 us each, ~0 bytes) with no same-stage glue neighbour. |

Also not a fusion opportunity: fake-collective kernels (2050 ms) and the 901 ms of GPU idle (Section 2).

## 6. Substrate observations


**What the graph made easy**

- *Exact dataflow and sizes.* Every candidate is a set of FX nodes; producers/consumers come from node args and
  every tensor has `val` shape/dtype/stride. External bytes (inputs read once, outputs written once) and legality
  are mechanical. Legality uses a *fusion stage* computed in one topological pass (number of GEMM/attention/
  collective nodes on the longest path): glue nodes with equal stage have no non-glue op on any path between them,
  so a component of equal-stage nodes is a legal single-kernel fusion. 5226 same-layer glue edges were cut by this
  rule (e.g. the dispatch-side index tensors reused by the combine scatter after two grouped GEMMs and two
  all_to_alls).
- *Phase provenance is explicit* (see below), so recompute->backward fusions are just edges (1243 same-stage RB
  edges; 14 regions carry the recompute-backward tag).
- *Attribution is exact.* `custom_codegen_pass` labels every node; kernel -> CUDA runtime launch (correlation id)
  -> enclosing label on the main thread mapped 99.93% of kernel time to FX nodes with no name matching.
- *Instances for free.* Normalizing `layers.<i>` in `custom.module_fqn` and abstracting unbacked sizes gives the
  repeating units (58 MoE layers, 61 attention blocks, 3 dense FFNs, 8 loss chunks).
- *Things an eager view hides, read directly from the graph:*
  - autograd emits `index_put(..., accumulate=True)` for the backward of the expert-order permutation gather even
    though its indices come from `sort.stable`/`cumsum` and form a permutation; with `debug.deterministic` this runs
    torch's sort-based path (~22 kernels, ~22 ms/layer). The forward un-permute `index_put_` (accumulate=False,
    permutation indices) takes the same sort path (11 ms/layer). A permutation-aware rewrite (inverse-permutation
    gather) is visible only because the index provenance is in the graph.
  - 14848 `_local_scalar_dense` nodes (256 per MoE layer) on CPU split sizes after one D2H `_to_copy`: a host sync
    per MoE layer that causes ~275 ms of GPU idle.
  - FSDP bucket pack/unpack copies, fp32 casts of every weight grad before the reduce-scatter pack, and the
    hi/mid/lo bf16 split of the fp32 router gate (`hi_mid_lo_linear.py`) appear as ordinary nodes with sizes.
  - Memory-policy decisions are inspectable: the OOM root cause (Pass 1b of `tag_sac_policy` saving the combine
    all_to_all output, 10.5 GiB per MoE layer) was found by listing forward nodes consumed by recompute/backward.

**What was hard / evidence that needed workarounds**

- `custom_codegen_pass` is not in the default pass list. Least invasive hook: register a pass pipeline from a
  module under the output directory (`PASS_PIPELINE_REGISTRY`, `compile.pass_pipeline="fusion_finder"`); no
  tracked file was edited.
- The fixed setup does not run on HEAD (OOM, above); fixed with an extra retagging pass in the same pipeline.
- Node metadata is inconsistent: `nn_module_stack` is sometimes a string (crashed the first dump); 474 backward
  accumulation nodes (residual-gradient adds, grad sums) have no `module_fqn` and were assigned the FQN of their
  first annotated input; FSDP bucketing nodes carry the FQN of the bucket's first parameter (e.g. views of RMSNorm
  weights labelled `attention.wq_a`).
- Out-variant writers (`split_with_sizes_copy.out`) write into views of `empty` buffers with no FX edge to the
  readers, and the reduce-scatter pack consumes every weight grad of a layer. Both create false connectivity; I cut
  edges out of allocation nodes, into shape-only consumers (`zeros_like`, `new_zeros`, ...), and into
  `bucketing._pre_bucket_*` except from casts that feed only the pack.
- Routed-token shapes are unbacked expressions (`u512 + ... + u575`). Sizes were resolved from the runtime
  `Input Dims` that `_RecordFunctionFast` records for each label (232 expressions); no region kept unknown bytes.
- Attention FLOPs depend on `cu_seqlens` values, which the graph does not hold, so attention nodes have a
  bytes-only ideal (not used by any region).
- Several costs are only visible in the trace, not the graph: TensorIterator splits every op on [786432,7168]
  (> 2^31 elements) into 16 launches; deterministic `index_put` launches memset+memcpy of the full destination;
  fake collectives launch DtoD copies/memsets.
- GEMM speed is data dependent: under the fake process group large GEMMs run at 2.34 PFLOP/s (zero-operand power
  regime) vs 1.81-1.87 PFLOP/s with random operands; the GEMM peak had to be recalibrated with zero operands.
- The eager optimizer / grad-norm kernels (17 ms) are outside the graph and were attributed by CPU op only.
- `regional_inductor_pass` is a no-op for this model (attention is `torch_attn._varlen_attn`, not FlexAttention),
  so the pre-Inductor dump and the executed graph are identical (128357 nodes).

**How recompute and backward provenance was identified**

- Backward: `node.meta["autograd_backward"] == True` (set by make_fx when tracing `torch.autograd.grad`).
- Recompute: `selective_activation_remat_pass` copies each MUST_RECOMPUTE forward node in front of its first
  backward consumer, names the copy `<orig>_recomputed` and sets `autograd_backward=True`; the copy keeps the
  `recompute=MUST_RECOMPUTE` tag, while `tag_sac_policy` never tags backward nodes. Phase rule: name matches
  `_recomputed(_\d+)?$` -> recompute; else `autograd_backward` -> backward; else forward. Cross-check: the set
  {autograd_backward and has a recompute tag} equals the suffix set exactly (25835 nodes; 0 backward nodes carry a
  tag). Forward originals that still have forward consumers stay in the forward phase with a MUST_RECOMPUTE tag.

**False leads discarded**

- A first union of all connected glue produced whole-layer clusters: the FSDP unpack buffers (`empty` -> views ->
  RMSNorm/gate weights) and the reduce-scatter pack tied unrelated activation paths together; the residual-gradient
  add chain without FQNs formed one 122-add cluster spanning all layers.
- Dispatch and combine glue of one MoE layer appeared as one component through shared index tensors
  (`floor_divide` used by the forward token gather and by the combine scatter); the fusion-stage rule separates them.
- `aten.mm.dtype` (router gate, fp32 out) and `torch_attn._varlen_attn*` were first classified as glue (3.1 s of
  attention counted as memory-bound).
- Byte accounting bugs that produced negative gaps: a split view of the wkv_b output counted the full tensor; an
  expanded int64 index ([786432,7168] stride 0) counted 45 GB; the FSDP pack's output `val` is the whole all-gather
  buffer, not the shard it writes.
- The 14848 `_local_scalar_dense` nodes looked like D2H kernels; they run on a CPU tensor. The single D2H is the
  preceding `_to_copy`; the cost is GPU idle, not kernel time.
- Single adds / RMSNorm kernels with slightly negative gaps (they run at the 6.9-7.0 TB/s pointwise rate, a bit
  above the D2D copy roofline) are at roofline: no opportunity on their own, only through cross-boundary fusion.

**Opportunity types this method can miss**

- Horizontal (sibling) fusion of independent ops with no dataflow edge (e.g. the q-path and kv-path RMSNorms, the
  ~30 per-layer weight-grad casts except through the shared pack, independent aux-loss reductions).
- Fusions that need reordering across a GEMM/collective (the stage rule is conservative: an edge whose endpoints
  differ in stage because of an *unrelated* longer GEMM chain is cut) and fusions with collectives (packing the
  token gather into the all_to_all send buffer, unpacking the all_gather directly into the parameter layout).
- Anything outside the traced graph (optimizer, grad clipping, data loading, H2D input copies).
- Single-op kernel inefficiencies are only reported when inside a multi-op region (e.g. `aten.topk` on 32-wide rows
  at ~2.4 ms is inside the router regions); a lone slow kernel lands in `no_opportunity`.
- Opportunities that change save/recompute decisions (e.g. saving the routed scores or the w13 output instead of
  recomputing) are out of scope of a fusion inventory but visible in the same graph.
- T3 gaps depend on the GEMM peak assumption; with the random-operand peak they would shrink by the GEMM part.

**Effort**

- Wall time: 10:56 start; capture attempts at 11:00 (dump crash after 4 min), 11:05 (OOM in step 1 after 15 min),
  11:13-11:26 (good run, ~13 min: 3 min trace, 2 min passes + dumps, 1 min codegen, 7 min of 10 steps, 10 s trace
  export); calibration, analysis and report until ~11:55. Total ~1 hour. The analysis pipeline reruns in ~2 minutes.
- Scripts (all under `scripts/`): `profile_dsv3_fake_graph.sh`, `fusion_finder_passes.py`, `calibrate_roofline.py`,
  `calibrate_gemm_data_dependence.py`, `parse_trace.py`, `graph_lib.py`, `build_node_table.py`, `find_regions.py`,
  `curate_regions.py`, `region_annotations.py`, `idle_breakdown.py`, `dump_region_details.py`, `write_report.py`,
  `report_text.py`, `validate_outputs.py`.


## 7. Reproduction


All paths are relative to `outputs/fusion_finder/graph/` (repo root `/home/bahuang/local/torchtitan`).

1. Calibrate (GPU 2 idle):
   `CUDA_VISIBLE_DEVICES=2 python scripts/calibrate_roofline.py --out calibration/calibration_before.json` and
   `CUDA_VISIBLE_DEVICES=2 python scripts/calibrate_gemm_data_dependence.py calibration/gemm_data_dependence.json`;
   `calibration/roofline.json` holds the numbers used (6876.3 GB/s, 2360.2 TFLOP/s).
2. Capture (from the repo root): `CUDA_VISIBLE_DEVICES=2 outputs/fusion_finder/graph/scripts/profile_dsv3_fake_graph.sh`.
   Writes `capture/recipe/` (generated recipe), `capture/graph_dump/{pre_inductor,final}_{nodes.jsonl.gz,graph.txt.gz}`
   (node records: name, op, target, args, users, val, stack_trace, nn_module_stack, custom, recompute tag,
   autograd_backward, seq_nr), `capture/codegen/fx_*_rank0.py` (profiled forward with one `_RecordFunctionFast`
   per node), `capture/pass_list.txt`, `capture/retagged_index_put_nodes.json`,
   `capture/iteration_10/rank0_trace.json.gz` (Kineto, CPU+CUDA) and `capture/profile.log`.
3. Analysis (CPU only, ~2 min):
   ```
   python scripts/parse_trace.py --trace capture/iteration_10/rank0_trace.json.gz --out analysis
   python scripts/build_node_table.py --graph capture/graph_dump/final_nodes.jsonl.gz --analysis analysis --calibration calibration/roofline.json
   python scripts/idle_breakdown.py analysis
   python scripts/find_regions.py --graph capture/graph_dump/final_nodes.jsonl.gz --analysis analysis --calibration calibration/roofline.json
   python scripts/curate_regions.py --root .
   python scripts/write_report.py .
   python scripts/validate_outputs.py .
   ```
   Outputs: `analysis/kernels.{pkl,csv.gz}` (every GPU activity with its FX node), `analysis/node_table.{pkl,csv.gz}`
   (node-level table), `analysis/step_budget.json`, `analysis/candidates.json` (all 300+ discovery candidates),
   `analysis/clusters.pkl`, `regions.json`, `report.md`.
