We demonstrate the effectiveness of elastic distributed training using torchtitan, via experiments on Llama 3.1 8B, 70B, and 405B models, from 1D parallelism to 4D parallelism, at the scale from 8 GPUs to 512 GPUs.

We ran our performance benchmarks on the [Grand Teton platform](https://engineering.fb.com/2022/10/18/open-source/ocp-summit-2022-grand-teton/), where
- Each host has 8 NVIDIA H100 GPUs fully connected with NVLink.
- Each H100 GPU is equipped with 96GB HBM2e with 2.4 TB/sec peak memory bandwidth.
- Hosts are inter-connected with backend RDMA network with 400 Gb/s per GPU.
- We used the default 500W power limit, although tuning it up to 700W TDP can potentially provide further speedups.

We note that, throughout our experimentation, memory readings are stable across the whole training process[^1], whereas throughput numbers (TPS/GPU) are calculated and logged every 10 iterations, and always read at the (arbitrarily determined) 90th iteration.

We do not report Model FLOPS Utilization (MFU) because when Float8 is enabled (on `nn.Linear` modules), both BFLOAT16 Tensor Core and FP8 Tensor Core are involved in model training, but they have different peak FLOPS and the definition of MFU under such scenario is not well-defined. We note that the 1D Llama 3.1 8B model training on 8 or 128 H100 GPUs without Float8 achieves 33% to 39% MFU[^2] (with or without torch.compile, respectively).

**Table 1** 1D Parallelism (FSDP). Llama 3.1 8B model. 8 GPUs. Local batch size 2, global batch size 16. Selective activation checkpointing.

| Techniques | TPS/GPU | Memory(GiB) |
| ----- | ----: | ----: |
| FSDP | 5,762 | 82.4 |
| FSDP + torch.compile | 6,667 | 77.0 |
| FSDP + torch.compile + Float8 | 8,532 | 76.8 |

**Table 2** FSDP + CP + torch.compile + Float8. Llama 3.1 8B model. 8 GPUs. Local batch size 1. Full activation checkpointing.

| Parallelism | Sequence Length | TPS/GPU | Memory(GiB) |
| ----- | ----: | ----: | ----: |
| FSDP 8, CP 1 | 32768 | 3,890 | 83.9 |
| FSDP 4, CP 2 | 65536 | 2,540 | 84.2 |
| FSDP 2, CP 4 | 131072 | 1,071 | 84.0 |
| FSDP 1, CP 8 | 262144 | 548 | 84.5 |

**Table 3** 1D Parallelism (FSDP). Llama 3.1 8B model. 128 GPUs. Local batch size 2, global batch size 256. Selective activation checkpointing.

| Techniques | TPS/GPU | Memory(GiB) |
| ----- | ----: | ----: |
| FSDP | 5,605 | 67.0 |
| FSDP + torch.compile | 6,514 | 62.0 |
| FSDP + torch.compile + Float8 | 8,380 | 61.8 |

**Table 4** 2D parallelism (FSDP + TP) + torch.compile + Float8. Llama 3.1 70B model. 256 GPUs (FSDP 32, TP 8). Local batch size 16, global batch size 512. Full activation checkpointing.

| Techniques | TPS/GPU | Memory(GiB) |
| ----- | ----: | ----: |
| 2D | 829 | 71.9 |
| 2D + AsyncTP | 876 | 67.6 |

**Table 5** 3D parallelism (FSDP + TP + PP) + torch.compile + Float8 + AsyncTP. Llama 3.1 405B model. 512 GPUs (FSDP 8, TP 8, PP8). Local batch size 32, global batch size 256. Full activation checkpointing.

| Schedule | TPS/GPU | Memory(GiB) |
| ----- | ----: | ----: |
| 1F1B | 100 | 82.5 |
| Interleaved 1F1B | 128 | 72.7 |

**Table 6** 4D parallelism (FSDP + TP + PP + CP) + torch.compile + Float8 + AsyncTP + 1F1B. Llama 3.1 405B model. 512 GPUs (TP 8, PP8). Local batch size 8. Full activation checkpointing.

| Parallelism | Sequence Length | TPS/GPU | Memory(GiB) |
| ----- | ----: | ----: | ----: |
| FSDP 8, CP 1 | 32768 | 76 | 75.3 |
| FSDP 4, CP 2 | 65536 | 47 | 75.9 |
| FSDP 2, CP 4 | 131072 | 31 | 77.1 |
| FSDP 1, CP 8 | 262144 | 16 | 84.9 |


#### Versions used for performance testing
| repo | commit | date |
| --- | --- | --- |
| torch | [1963fc8](https://github.com/pytorch/pytorch/commit/1963fc83a1c32e162162e2414f78b043f0674bae) | 2024/12/23 |
| torchao | [eab345c](https://github.com/pytorch/ao/commit/eab345c2268a7506355d506ebfc27b5d28e5e7d0) | 2024/12/23 |
| torchtitan | [9dec370](https://github.com/pytorch/torchtitan/commit/9dec370ad26b5f8e9a7333a0e36165018262644b) | 2024/12/26 |


[^1]: Different PP ranks can have different peak memory usages. We take the maximum across all GPUs.

[^2]: In our test we used HBM2e-based SXM H100 with lower TDP, the actual peak TFLOPs number is between SXM and NVL, and we don't know its exact value. So this MFU number is lower than actual MFU because we use the peak number of SXM directly.
