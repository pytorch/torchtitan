# GraphTrainer benchmarks matched to Megatron-Bridge 26.08.01 (GB300, MXFP8)

Recipes: `torchtitan_recipes/graph_trainer/megatron_bridge_benchmarks.py`

| Megatron-Bridge recipe (`perf_recipes.*.gb300`) | torchtitan recipe |
|---|---|
| `deepseek_v3_pretrain_256gpu_gb300_fp8mx_config` | `graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_256gpu` |
| `qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_config` | `graph_trainer_qwen3_235b_a22b_dist_moe_mxfp8_256gpu` |
| `qwen3_30b_a3b_pretrain_8gpu_gb300_fp8mx_config` | `graph_trainer_qwen3_30b_a3b_dist_moe_mxfp8_8gpu` |
| `gpt_oss_120b_pretrain_64gpu_gb300_fp8mx_config` | `graph_trainer_gpt_oss_120b_mxfp8_64gpu` |

## Sources

* The performance summary page:
  https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html
* Megatron-Bridge **v0.6.2**, the 26.08 release that refreshed the 26.08.01
  summary (release notes: "docs(performance): refresh summary for 26.08.01").
* Megatron-Core at the commit v0.6.2 pins (`3rdparty/Megatron-LM`,
  `a0f793dfa4e776d99a8aa63bed74e0fafc91b0db`).

For each recipe, the full config chain was traced at that tag:

* the perf recipe function
* `perf_recipes/_common.py` (`_benchmark_common`, `_perf_precision`)
* the family helpers (`_deepseek_v3_common`, `_enable_deepseek_full_iteration_mxfp8`,
  `_enable_hybridep_full_iteration_mxfp8`, `_apply_gpt_oss_120b_full_iter_fp8mx_configs`)
* the H100 base recipes they start from
* `training/mixed_precision.py` (`bf16_with_mxfp8_mixed`)
* the `scripts/performance` launcher defaults: `--data mock`, and the
  tokenizer and token-drop overrides are no-ops by default

Megatron-Bridge `main` (0.7.0+, the 26.10 release) has diverged: it switches
these recipes to the NCCL EP dispatcher and adds `moe_use_grouped_tensor`.
Neither change is in the configuration that produced the published numbers.

torchtitan model shapes match the HF configs Megatron instantiates
(`AutoBridge.from_hf_pretrained`): layer count, MoE layer count, MTP, vocab,
expert count, top-k, and expert dims.

## 1. Summary

* All four recipes run 3 steps under fake PG on one GB300 (276.6 GiB usable)
  at the full logical world size, with 73% to 91% peak reserved memory.
* Global batch size, micro batch size, sequence length, and GPU count match
  Megatron exactly. Tokens per optimizer step are identical.
* No PP. Megatron's PP x VP ranks become FSDP ranks, so each rank accumulates
  GBS / (DP x MBS) microbatches per step.
* **CUDA graphs: on** for DeepSeek V3, Qwen3 235B, and Qwen3 30B. **Off** for
  GPT-OSS 120B, which cannot use Dist-MoE (section 5). Megatron uses
  full-iteration CUDA graphs for all four.
* MXFP8 covers the same GEMMs as Megatron, except the attention core (BF16
  here) and the GPT-OSS routed experts (BF16 here). Section 4 has the
  region-by-region breakdown.
* Compare **tokens/sec/GPU**. torchtitan's `tflops` uses a different attention
  FLOP convention. For Megatron-equivalent TFLOP/s/GPU, multiply torchtitan's
  tps by Megatron's GFLOP/token (section 2).

## 2. Megatron reference numbers (GB300, MXFP8, 26.08.01)

| Model | GPUs | GBS | MBS | Seq | TP | PP | VP | EP | Tokens/s/GPU | TFLOP/s/GPU | GFLOP/token (derived) | Step time (derived) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek V3 671B | 256 | 4096 | 1 | 4096 | 1 | 2 | 8 | 32 | 6288 | 1636 | 260.2 | 10.42 s |
| Qwen3 235B-A22B | 256 | 8192 | 2 | 4096 | 1 | 4 | 12 | 32 | 8832 | 1306 | 147.9 | 14.84 s |
| Qwen3 30B-A3B | 8 | 512 | 8 | 4096 | 1 | 1 | - | 8 | 44544 | 1023 | 22.97 | 5.89 s |
| GPT-OSS 120B | 64 | 1280 | 4 | 4096 | 1 | 1 | - | 16 | 33024 | 1077 | 32.61 | 2.48 s |

GFLOP/token = TFLOP/s / tokens/s. Step time = GBS x seq / (tokens/s/GPU x GPUs).

## 3. Geometry

| | DeepSeek V3: Megatron | DeepSeek V3: proposed | Qwen3 235B: Megatron | Qwen3 235B: proposed | Qwen3 30B: Megatron | Qwen3 30B: proposed | GPT-OSS 120B: Megatron | GPT-OSS 120B: proposed |
|---|---|---|---|---|---|---|---|---|
| GPUs | 256 | 256 | 256 | 256 | 8 | 8 | 64 | 64 |
| GBS (sequences) | 4096 | 4096 | 8192 | 8192 | 512 | 512 | 1280 | 1280 |
| Tokens / step | 16,777,216 | 16,777,216 | 33,554,432 | 33,554,432 | 2,097,152 | 2,097,152 | 5,242,880 | 5,242,880 |
| MBS (seqs) / tokens per microbatch | 1 / 4096 | 1 / 4096 | 2 / 8192 | 2 / 8192 | 8 / 32768 | 8 / 32768 | 4 / 16384 | 4 / 16384 |
| Seq len | 4096 | 4096 | 4096 | 4096 | 4096 | 4096 | 4096 | 4096 |
| TP / CP | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| PP x VP | 2 x 8 | none | 4 x 12 | none | none | none | none | none |
| EP | 32 | **64** | 32 | 32 | 8 | 8 | 16 | 16 |
| DP | 128 | 256 (FSDP) | 64 | 256 (FSDP) | 8 | 8 (FSDP) | 64 | 64 (FSDP) |
| Expert DP (DP x TP x CP / EP) | 4 | 4 | 2 | 8 | 1 | 1 | 4 | 4 |
| Microbatches per step per rank | 32 | **16** | 64 | **16** | 8 | 8 | 5 | 5 |

Weight and gradient residency: Megatron's distributed optimizer keeps the
local PP stage's parameters and a BF16 gradient buffer, with optimizer state
sharded over DP. The proposed recipes use FSDP that unshards once on the first
microbatch and reduces once on the last (`spmd_gradient_accumulation`
defaults), with optimizer state sharded.

Per-GPU work per step is the same. With PP, Megatron runs more microbatches
per rank, each over fewer layers; without PP, each rank runs fewer
microbatches over all layers.

## 4. MXFP8 coverage, region by region

Megatron: Transformer Engine `bf16_with_mxfp8_mixed` (`fp8="e4m3"`,
`fp8_recipe="mxfp8"`, `fp8_param_gather=True`,
`reuse_grad_buf_for_mxfp8_param_ag=True`, `first_last_layers_bf16=False`).
Every Transformer Engine linear inside every transformer layer runs under the
FP8 context, and Transformer Engine quantizes the forward, dgrad, and wgrad
GEMMs.

Proposed: torchtitan `MXFP8Linear` (forward, dgrad, and wgrad as MXFP8
`scaled_mm`) and Dist-MoE `BlockScaledConfig(MXFP8_E4M3, pipeline="staged")`
(block-scaled FPROP, DGRAD, and WGRAD). Coverage was listed from the final
model configs after overrides; `fused_mla` and `fused_swiglu` preserve the
converted linears.

| Region | Megatron | Proposed | Match |
|---|---|---|---|
| DeepSeek V3 MLA projections (q down/up, kv down/up, out) | MXFP8 | MXFP8 (`wq_a`, `wq_b`, `wkv_a`, `wkv_b`, `wo`) | yes |
| Qwen3 / GPT-OSS QKV and output projections | MXFP8 | MXFP8 (`wqkv`, `wo`) | yes |
| DeepSeek V3 dense FFN (first 3 layers) and shared experts | MXFP8 | MXFP8 | yes |
| Routed experts | MXFP8 (Transformer Engine grouped GEMM) | MXFP8 Dist-MoE (DeepSeek V3, both Qwen3); **BF16** (GPT-OSS) | GPT-OSS no |
| DeepSeek V3 MTP layer (attention, experts, shared experts) | MXFP8 | MXFP8 | yes |
| DeepSeek V3 MTP `eh_proj` | MXFP8 (inside the MTP FP8 context) | MXFP8 | yes |
| LM head | MXFP8 for DeepSeek V3 and GPT-OSS (`fp8_output_proj=True`); BF16 for Qwen3 | Same | yes |
| Router gate | Not quantized (`router_gating_linear` in BF16 or `moe_router_dtype`) | Not quantized (`HiMidLoLinear`) | yes |
| Embedding | Not quantized | Not quantized | yes |
| Attention core (QK^T, PV) | **FP8** (`fp8_dot_product_attention=True`, all four) | **BF16** (varlen flash attention) | no |
| Parameter storage / all-gather | MXFP8 | BF16 FSDP all-gather, quantized after the gather | no |
| Saved linear inputs for backward | Quantized by Transformer Engine | BF16 by default; DeepSeek V3 saves MXFP8 for `wkv_b`, `w2`, and shared-expert `w2` | memory only |

Other precision settings:

| | Megatron | Proposed |
|---|---|---|
| Main weights / gradient reduction | FP32 / BF16 (`grad_reduce_in_fp32=False`) | FP32 / BF16 |
| Adam moments | DeepSeek V3: BF16 (precision-aware optimizer); others: FP32 | DeepSeek V3: BF16; **Qwen3 30B: BF16**; Qwen3 235B and GPT-OSS: FP32 |

## 5. MoE, memory, and execution

| | Megatron | Proposed |
|---|---|---|
| Token dispatch | Flex dispatcher, HybridEP backend (32 SMs, MNNVL across the 72-GPU NVLink domain) | Dist-MoE fused dispatch + grouped GEMM + combine (DeepSeek V3, both Qwen3); stock all-to-all with host-synced splits (GPT-OSS) |
| Expert receive capacity | `moe_expert_rank_capacity_factor=1.5` | Dist-MoE `scratch_capacity_factor=1.5` (GPT-OSS: dynamic, no cap) |
| Routing | `moe_router_force_load_balancing=True` (random logits) | `RoundRobinTokenChoiceTopKRouter` (exact balance) |
| Activation recompute | None | None: `compile.memory_policy="none"`, Dist-MoE `activation_slot_capacity_factor=1.0` keeps every expert intermediate |
| MoE activation storage | Paged stash (CUDA factor 1.2, CPU spill factor 1.0) | Dist-MoE activation slot sized for balanced routing |
| CUDA graphs | Full-iteration, all four | **Yes** for DeepSeek V3 and both Qwen3 (whole forward/backward step); **no** for GPT-OSS |
| EP comm overlap | `overlap_moe_expert_parallel_comm`, `delay_wgrad_compute`, high-priority A2A stream | Pipelining inside Dist-MoE; graph_trainer `ep_overlap` not enabled |
| Fused kernels | Transformer Engine op fuser, fused router, RoPE fusion, Transformer Engine fused cross entropy, CuTe DSL fused grouped MLP, cuDNN norms (DeepSeek V3) | graph_trainer regional Inductor; DeepSeek V3 `fused_mla` + `fused_swiglu` overrides; chunked LM head + loss |
| MTP | DeepSeek V3: 1 layer | DeepSeek V3: 1 layer |
| Data | Mock data, full causal attention over each 4096-token sequence (`reset_attention_mask=False`) | `c4_test` sample + test tokenizer, packed with **document masking** |
| Iterations | 50 (10 warmup) | 50 (10 warmup) |

Deliberate deviations:

1. **No PP.** With unshard-once/reduce-once FSDP, each rank keeps the
   unsharded BF16 parameters and gradients of *all* layers for the whole step,
   about twice what Megatron's PP2 holds for DeepSeek V3.
2. **DeepSeek V3 uses EP64 instead of EP32.** EP32 ran out of memory under fake
   PG (274 GiB on step 2). EP64 halves the local experts. Megatron uses EP64
   for this model on GB200, and expert DP stays 4, matching Megatron's GB300
   run.
3. **Qwen3 30B uses BF16 Adam moments** (Megatron: FP32). EP8 spans all 8 GPUs,
   so FSDP cannot shard expert optimizer state. BF16 moments cut peak memory
   from 267 to 253 GiB without recompute; the optimizer step is a negligible
   share of step time.
4. **GPT-OSS experts are BF16 and run without CUDA graphs.**
   * Dist-MoE rejects GPT-OSS because its experts have biases
     (`GptOssGroupedLinear`).
   * torchao's MXFP8 grouped GEMM requires K % 128 == 0, and GPT-OSS dim is
     2880 (the MXFP8 format itself needs only K % 32).
   * The stock all-to-all dispatcher syncs split sizes to the host, so
     torchtitan refuses CUDA graphs with it.
   * HybridEP, the only other graph-safe dispatcher, needs a DeepEP build and
     runs only with real PG, so it cannot be validated with fake PG.
5. **Chunked LM head + loss** (`ChunkedLossWrapper`). Without it, Qwen3 30B's
   32K-token microbatch materializes 18.55 GiB of FP32 logits and runs out of
   memory. Megatron's Transformer Engine fused cross entropy avoids the same
   allocation.

## 6. Caveats that affect comparability (largest first)

1. **Attention work differs.** torchtitan always masks attention at document
   boundaries, and `c4_test` documents are short. Megatron's mock data runs full
   4096-token causal attention. At full causal 4096, the attention core is
   roughly 12% (DeepSeek V3), 13% (Qwen3 235B), 21% (Qwen3 30B), and 6%
   (GPT-OSS) of per-token FLOPs (estimates with average context 2048, not
   Megatron's formula). torchtitan therefore does less attention work, which
   flatters its tokens/sec, most of all on Qwen3 30B. Closing this needs a data
   source whose documents fill the 4096-token row.
2. **FP8 attention.** Megatron runs MXFP8 dot-product attention; torchtitan
   runs it in BF16.
3. **GPT-OSS** experts are BF16 and it runs without CUDA graphs, so its result
   is a lower bound, not like-for-like.
4. **FLOP accounting.** torchtitan's `tflops` ignores causal sparsity and uses
   its own formula. Use tokens/sec/GPU, or tps x Megatron GFLOP/token.
5. **Parameter all-gather.** Megatron stores weights and all-gathers them in
   MXFP8, halving parameter-gather traffic. torchtitan's FSDP gathers BF16 and
   quantizes after the gather. With one gather per step, this matters most
   where FSDP spans racks (the 256-GPU recipes).
6. **Routing.** Round-robin routing is perfectly balanced; Megatron's random
   logits are balanced only in expectation. Dist-MoE receive buffers are sized
   for 1.5x, like Megatron's capacity bound, but receive exactly 1.0x.

## 7. Fake-PG validation

One GB300, 3 steps, `PYTORCH_ALLOC_CONF=expandable_segments:True`, validated on
`main` at `da9de4d59`.

| Recipe | Logical NGPU | Result | Peak reserved | Dist-MoE device buffer | Model memory after init | CUDA graph |
|---|---|---|---|---|---|---|
| DeepSeek V3 671B (EP64) | 256 | pass | 227.5 GiB (82%) | 60.0 GiB (58.0 activation slot + 1.9 scratch) | 13.6 GiB | recorded |
| Qwen3 235B-A22B | 256 | pass | 243.0 GiB (88%) | 118.5 GiB (116.0 + 2.5) | 4.1 GiB | recorded |
| Qwen3 30B-A3B | 8 | pass | 252.8 GiB (91%) | 123.5 GiB (118.5 + 5.0) | 14.5 GiB | recorded |
| GPT-OSS 120B | 64 | pass | 202.6 GiB (73%) | - | 9.1 GiB | off by design |

* Fake PG does not model NCCL communicator memory, transport scratch, or
  collective overlap. Qwen3 30B's 91% leaves about 24 GiB for those. If that is
  not enough, the first knob is a Dist-MoE `activation_slot_capacity_factor`
  below 1.0, which adds some expert recompute.
* Fake-PG tokens/sec are not throughput measurements and are not reported.
* These recipes target GB300 (288 GB). On GB200 (186 GB) they would not fit
  without PP; Megatron's GB200 rows also use PP4 or PP8.

Run a recipe under fake PG (it runs the full 50 steps; wrap it in a module that
sets `config.training.steps` for a shorter check):

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True NGPU=256 COMM_BACKEND=fake \
MODULE=torchtitan_recipes.graph_trainer.megatron_bridge_benchmarks \
CONFIG=graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_256gpu ./run_train.sh
```

Use `NGPU=256` for Qwen3 235B, `NGPU=8` for Qwen3 30B, and `NGPU=64` for
GPT-OSS 120B.

## 8. Requirements

* `dist_moe` for the three Dist-MoE recipes:
  `pip install --no-deps "git+https://github.com/meta-pytorch/dist_moe.git@main"`.
  `--no-deps` avoids downgrading `nvidia-cutlass-dsl`, which `dist_moe` pins
  at `<=4.8.0`.
* A torchao nightly for `MXFP8Linear`:
  `USE_CPP=0 pip install --pre --no-deps torchao --index-url https://download.pytorch.org/whl/nightly/cu132`.

## 9. Next steps

1. Real runs: Qwen3 30B on one 8-GPU GB300 node first, then the 64- and
   256-GPU recipes.
2. Add a data source with full-length documents to remove caveat 6.1.
3. Performance knobs to try after a baseline (all numerics-changing):
   `compile.numerics_changing_optim=True`, `compile.inductor_compilation="full"`,
   and graph_trainer `ep_overlap`.
4. GPT-OSS parity needs expert-bias support in Dist-MoE or a K % 32 tile in
   torchao's MXFP8 grouped quantization.
5. Add the module to `torchtitan_recipes/README.md` once a real run validates
   the recipes.
