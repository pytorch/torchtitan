# Muse Spark Open 816B-A42B: BF16 optimizer state on 64 GPUs

This release measures TorchTitan training throughput for Muse Spark Open
816B-A42B on 64 NVIDIA Vera Rubin GPUs, with BF16 and with MXFP8 expert
compute. In both, Adam moments and FSDP gradient reductions are BF16, while
master weights and gradients stay FP32. The freed memory goes to the Dist-MoE
activation slot, which keeps more MoE activations on the GPU, so fewer MoE
layers recompute in backward.

It is a performance benchmark, not a training recipe. Routing is forced to a
balanced round-robin assignment and weights are randomly initialized.

## Results

### Throughput and memory

TPS is tokens/s/GPU. Each update processes 64 rows of 32,768 token slots, and
TPS counts all slots, including padding. Global TPS is TPS/GPU x 64. TFLOPs/GPU
is TorchTitan's analytical model-FLOP estimate; MFU is not reported. Each
baseline is the 64-GPU recipe of the same precision
(`muse_spark_816b_a42b_dist_moe_{bf16,mxfp8}_spmd_64`) with the same forced
routing; [What changed](#what-changed-from-the-baseline) lists the differences.

| Precision | Configuration | TPS/GPU | Global TPS | vs baseline | TFLOPs/GPU | Step s | Peak active GiB | Peak reserved GiB | Peak device used GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | Candidate, DP32 x CP2 | 6,498.03 | 415,874 | +49.34% | 2,127.55 | 5.043 | 246.12 | 256.78 | 271.84 |
| BF16 | Baseline, DP8 x CP8 | 4,351.26 | 278,481 | -- | 1,424.67 | 7.531 | 209.38 | 257.61 | 266.42 |
| MXFP8 | Candidate, DP64 x CP1 ([stacked](#mxfp8-stacked-recipe)) | 9,754.42 | 624,283 | +103.91% | 3,193.72 | 3.359 | -- | 263.92 | -- |
| MXFP8 | Baseline, DP8 x CP8 | 4,783.77 | 306,161 | -- | 1,566.28 | 6.850 | 209.84 | 258.07 | 266.90 |

Key observations:

- **Throughput:** the candidates are 49.3% (BF16) and 103.9% (MXFP8) faster
  than their baselines. The MXFP8 candidate is 50.1% faster than the BF16
  candidate.
- **Memory:** about 6.7 GiB of device memory stayed free in the BF16 candidate,
  and about 12 GiB in both baselines. The MXFP8 run logged peak reserved memory
  (maximum over all 64 GPUs) only.

The candidates and baselines were measured differently:

| | Candidates | Baselines |
|---|---|---|
| Measured updates | 71..90 of 100 | 13..22 of 22 |
| TPS | Mean of the per-update logged values on rank 0 | Longest synchronized rank-local window, divided by 10 updates |
| Peak active / reserved | PyTorch allocator high-water on rank 0 (MXFP8 reserved: maximum over all 64 GPUs) | Maximum over all 64 GPUs |
| Peak device used | Sampled every 200 ms on all GPUs | Sampled every 100 ms on all GPUs |

Allocation retries and OOM counters stayed zero in both candidate runs. Each
configuration was run once; differences are observed values, not confidence
intervals.

<details>
<summary>Per-update TPS distribution, candidates</summary>

| Precision | Mean TPS/GPU | Median | Min..max | Global TPS | Mean step s |
|---|---:|---:|---:|---:|---:|
| BF16 | 6,498.03 | 6,494.67 | 6,476.82..6,534.83 | 415,874 | 5.0428 |
| MXFP8 | 9,754.42 | 9,797 | 9,585..9,878 | 624,283 | 3.3593 |

Updates 71..90, rank 0.

</details>

### What changed from the baseline

| Area | Baselines | Candidates |
|---|---|---|
| Parallelism | FSDP 8 x CP 8 x EP 64; 8 microbatches per DP rank | BF16: FSDP 32 x CP 2 x EP 64, 2 microbatches per DP rank. MXFP8: FSDP 64 x CP 1 x EP 64, 1 microbatch per DP rank |
| Attention CP | K/V all-gather | K/V all-gather, with selective K/V gather in the sliding-window layers |
| Saved activations | Attention and MoE projection regions kept on the GPU | Attention regions and block inputs offloaded to pinned host memory, except in the last block |
| Dist-MoE | About 17 GiB activation slot (capacity factor 0.25) | 70 GiB slot (BF16); 40 GiB per live slot and the staged expert pipeline (MXFP8) |
| Optimizer state and reductions | FP32 Adam moments; FP32 gradient reductions | BF16 Adam moments; BF16 gradient reductions |
| FSDP | NCCL collectives; default resharding | Symmetric-memory collectives; default resharding |
| Norms and LM head | Eager | MXFP8 only: pre-norms write the branch dtype, the post-expert norm is fused into the Dist-MoE combine, MXFP8 LM head with FP32 output and composed loss, compiled post-gate router, softcap and cross entropy |
| Tokenizer | Test tokenizer | Muse Spark Open tokenizer |

All four use the same model, 32K sequences, global batch 64, `c4_test`, seed
42, forced round-robin routing, Dist-MoE scratch factor 1 and GPU-resident Adam
moments. MXFP8 runs the routed experts and the attention and compressed-MoE
projections in MXFP8; inner attention stays BF16.

### Numerics

| Precision | Loss, update 1 | Loss, update 100 | Mean loss, updates 71..90 |
|---|---:|---:|---:|
| BF16 | 12.2302 | 6.9235 | 7.1078 |
| MXFP8 | 12.2302 | 6.9075 | 7.0970 |

All 64 ranks completed 100 updates with finite loss and gradient norm in both
candidates. The baselines used a different tokenizer, so their losses are not
comparable.

## Workload

| Area | Configuration |
|---|---|
| Hardware | 64 NVIDIA Vera Rubin GPUs: 16 hosts x 4 GPUs, all in one NVLink domain (required by Dist-MoE EP64). Each CP group must share an NVLink domain. |
| Model | Muse Spark Open 816B-A42B, text decoder: 62 layers, hidden 8,192, 256 compressed-MoE experts, top-8 |
| Batch | Sequence length 32,768; global batch 64 sequences (2,097,152 tokens per update); 2 (BF16) or 1 (MXFP8) microbatches per DP rank |
| Data | `c4_test` with concat-then-split packing and at most 64 documents per row; Muse Spark Open tokenizer (revision `a604fa753b672b5dca3304d8648cf756120e29db`) |
| Initial weights | Random, seed 42; no checkpoints saved |
| Parallelism | FSDP 32 x CP 2 x EP 64 (BF16), FSDP 64 x CP 1 x EP 64 (MXFP8); no TP or PP |
| Attention | FA4 varlen; K/V all-gather CP with selective K/V gather in the sliding-window layers |
| Routed experts | Dist-MoE, BF16 or MXFP8 grouped GEMMs; forced balanced round-robin routing; scratch capacity factor 1; one activation slot of 70 GiB (BF16), or 40 GiB per live slot with the staged expert pipeline (MXFP8) |
| Precision | BF16 or MXFP8 compute; FP32 master weights and gradients; BF16 Adam moments (`moment_dtype="bfloat16"`) and FSDP gradient reductions (`reduce_dtype="bfloat16"`) |
| Optimizer | Fused AdamW; lr 5e-5 (router gate 6.25e-6); betas 0.9 / 0.95; eps 1e-8; weight decay 0; gradient clip 1.0; BF16 moments on the GPU |
| LR schedule | 10 warmup steps, cosine decay, minimum factor 0.1 |
| Memory policy | Selective activation remat: attention and Dist-MoE regions saved, everything else recomputed. Attention regions and block inputs offloaded to pinned host memory, except in the last block. |
| FSDP | Symmetric-memory collectives; default resharding |
| CUDA graphs | Training graphs per microbatch group and a CUDA graph for the optimizer step |
| Environment | `NCCL_NVLS_ENABLE=1`; `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`; MXFP8 adds `NCCL_MIN_CTAS=1`, `NCCL_MAX_CTAS=8` and binds local ranks 0,1 to NUMA node 0 and 2,3 to NUMA node 1 |

## Software

| Component | Candidates | Baselines |
|---|---|---|
| TorchTitan | BF16: [#6252](https://github.com/pytorch/torchtitan/pull/6252) at `814df1c9e892f0e45bb2af5523c1968c52e7262b` plus the activation-offload change of [#6020](https://github.com/pytorch/torchtitan/pull/6020). MXFP8: [#6352](https://github.com/pytorch/torchtitan/pull/6352) at `48a3ed8f48950977dd908c9f6ea53626a9851947` plus [#6369](https://github.com/pytorch/torchtitan/pull/6369) and [#6371](https://github.com/pytorch/torchtitan/pull/6371) | [#6136](https://github.com/pytorch/torchtitan/pull/6136) at `bf1558c02ecbe4d9855f623b1a98222e4e36c44e` |
| PyTorch | nightly `2.16.0.dev20261119+cu134` (git `daf1a40a5de981f7cab47fed4dd9883b1581cd63`) | nightly `2.16.0.dev20261116+cu134` |
| Dist-MoE | BF16: `b76e0e042b4e97731fec981e974d375eb8eeed8b`. MXFP8: [dist_moe#18](https://github.com/meta-pytorch/dist_moe/pull/18) at `bea783f9585afbe1aaf087650acf54465a918af6` | `b76e0e042b4e97731fec981e974d375eb8eeed8b` |
| torch_remat | `71905a6e60123a9adadd4d0cd5acae4e3cb8ba88` | `6c05e8d8d6895a60dcc02674af0774b28155b5af` |
| attention-gym | v0.0.20 (`7adfbadb06c30eeea82dc3f050ae25512d44088e`) plus the unmodified commit of [attention-gym#651](https://github.com/meta-pytorch/attention-gym/pull/651) (`d08fdce601ef0ea6c925e970442b3dda26e79752`); the PR head alone lacks modules this TorchTitan revision imports | same |

The BF16 candidate recipe in `torchtitan_recipes/models/muse_spark.py` and the
CP1 recipe of #6352 encode the measured candidate configurations.

## Reproducing

Both setups need `NCCL_NVLS_ENABLE=1` and
`PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`, and launch on 16 hosts x 4
GPUs in one NVLink domain:

```bash
torchrun --nnodes=16 --nproc-per-node=4 \
  --node-rank="$NODE_RANK" \
  --rdzv-backend=static --rdzv-endpoint="$MASTER_ADDR:29500" \
  -m torchtitan.train --module "$MODULE" --config "$CONFIG" \
  --output-dir "/shared/outputs/$CONFIG"
```

Known issue: with FSDP symmetric memory and EP, `destroy_process_group` can hang
after the last update. The measured runs wrapped `Trainer.close` to call
`os._exit(0)` after a successful close.

### Candidates

With the tokenizer files staged at `./assets/hf/Muse-Spark-Open-816B-A42B`, set
`MODULE=torchtitan_recipes.models.muse_spark` and `CONFIG` to:

| Precision | `CONFIG` |
|---|---|
| BF16 | `muse_spark_816b_a42b_dist_moe_bf16_spmd_64_bf16_state_benchmark` |
| MXFP8 | `muse_spark_816b_a42b_dist_moe_mxfp8_spmd_64_benchmark_cp1`, on the [MXFP8 stack](#mxfp8-stacked-recipe) |

Each recipe runs 100 updates; compute TPS as the mean over updates 71..90.

### MXFP8 stacked recipe

The MXFP8 candidate needs open PRs in TorchTitan and Dist-MoE. Build TorchTitan
from #6352 and cherry-pick #6369 and #6371 on top:

```bash
git fetch origin pull/6352/head pull/6369/head pull/6371/head
git checkout 48a3ed8f48950977dd908c9f6ea53626a9851947  # #6352
git cherry-pick 90605bffb  # #6369: pre-norms write the branch dtype
git cherry-pick 62ca7bc10  # #6371: fused post-expert norm
```

Install Dist-MoE from dist_moe#18, which adds
`dist_moe.InputScaledRMSNormPostprocess` used by #6371:

```bash
git fetch origin pull/18/head
git checkout bea783f9585afbe1aaf087650acf54465a918af6
```

Launch with these in addition to the common settings:

```bash
export NCCL_MIN_CTAS=1 NCCL_MAX_CTAS=8
```

and bind local ranks 0 and 1 to NUMA node 0 and local ranks 2 and 3 to NUMA
node 1 (for example with `numactl --cpunodebind=N --membind=N`).

### Baselines

Check out the baseline commit (`git fetch origin pull/6136/head`, then
`git checkout bf1558c02ecbe4d9855f623b1a98222e4e36c44e`). Save this module as
`baseline_configs.py` on `PYTHONPATH`, then set `MODULE=baseline_configs` and
`CONFIG=bf16` or `CONFIG=mxfp8`:

```python
from dataclasses import replace

from torchtitan.components.data import ConcatThenSplitPackingConfig
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan_recipes.models import muse_spark


def _baseline(precision):
    config = getattr(muse_spark, f"muse_spark_816b_a42b_dist_moe_{precision}_spmd_64")()
    config.hf_assets_path = "./tests/assets/tokenizer"
    config.tokenizer = HuggingFaceTokenizer.Config()
    config.dataloader = replace(
        config.dataloader,
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        shuffle=False,
    )
    config.checkpointer = None
    config.debug.seed = 42
    config.dist_moe_runtime = replace(
        config.dist_moe_runtime,
        scratch_capacity_factor=1.0,
        activation_slot_capacity_factor=0.25,
        vmm_capacity_factor=None,
    )
    config.optim.optimizer.optimizers = [
        replace(optimizer, moment_location="device")
        for optimizer in config.optim.optimizer.optimizers
    ]
    config.parallelism.fsdp_enable_cpu_offloading = False
    muse_spark._use_forced_balanced_routing(config.model)
    return config


def bf16():
    return _baseline("bf16")


def mxfp8():
    return _baseline("mxfp8")
```

The baseline recipes keep their 120-update horizon; the measured runs stopped
after update 22 and timed updates 13..22.
