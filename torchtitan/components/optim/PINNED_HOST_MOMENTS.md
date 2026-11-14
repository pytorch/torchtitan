## Adam Moments in Pinned Host Memory

Set an Adam/AdamW config's `moment_location` to `"pinned_host"` to keep `exp_avg` and `exp_avg_sq` in pinned CPU memory instead of accelerator memory. The unchanged fused Adam kernel reads and writes them in place over the CPU-GPU link. There are no copies, side streams, or extra kernels, and the step stays CUDA-graph capturable.

```python
config.optim.optimizer.optimizers[0].moment_location = "pinned_host"
```

Launch with NUMA binding so each rank's moments are allocated on its GPU's NUMA node:

```bash
torchrun --numa-binding=node ...
```

Without it, on GB300 with four GPUs busy, copies in both directions at once dropped about 2-2.5x (86-111 GB/s to 44 GB/s). Host-to-device copies alone dropped 1.0-1.8x.

### When to use it

Use it when the Adam moments decide whether training fits, for example when a run goes out of memory at the first optimizer step. It is designed for hosts with a fast CPU-GPU link, such as GB300. It has not been measured on PCIe hosts, where the step will be much slower.

### Cost

Each optimizer step reads and writes every local moment once over the host link:

```
bytes per step = 2 (read + write) x 2 moments x moment bytes x local parameters
               = 16 bytes per local parameter for fp32 moments
```

The AdamW time is about these bytes divided by the host link bandwidth. Accelerator memory for the moments drops to zero. Pinned host memory grows by `2 x moment bytes` per local parameter.

On a run that fits both ways, the Muse Spark 816B fake-process-group recipe `muse_spark_816b_a42b_dist_moe_bf16_spmd_256_fake` (256 logical GPUs, DP16/TP4/CP4/EP64, 96K sequence length) on one GB300 GPU bound to its NUMA node, steps 4-10:

| | `"device"` | `"pinned_host"` |
| --- | --- | --- |
| peak reserved memory | 253.06 GiB | 228.84 GiB (-24.2 GiB) |
| tps | 3,301 | 3,273 (-0.85%) |
| AdamW per step (mean of steps 4-9) | 26.2 ms | 166.5 ms (about 1.1% of a 15 s step) |

On one GB300 GPU bound to its NUMA node, a 64-GPU fake-process-group Muse Spark run (DP8/CP8/EP64, 32K sequence length) moves 2 x 104 GB per AdamW step. AdamW alone takes about 0.63 s (about 330 GB/s), 5.8% of the step, in one synchronized measurement. The two GPUs of a Grace socket share its host bandwidth, so when both run the step at the same time, AdamW takes about 2x longer.

### How it works

The optimizer creates both moments and `step` at construction. Each moment is a CPU tensor pinned in place with `cudaHostRegister` and exposed to CUDA as a tensor aliasing that memory (`__cuda_array_interface__`). A DTensor parameter gets DTensor moments with the same mesh and placements.

```
GPU: param, grad --fused Adam--> param
                     ^    |
                     |    v  host link, in place
CPU (pinned):  exp_avg, exp_avg_sq
```

DCP saves and loads the moments like device moments, so checkpoints are interchangeable between `"device"` and `"pinned_host"`. Loading writes into the same pinned buffers, without a temporary copy in accelerator memory.

### Requirements

Rejected with `ValueError`:

- `fused=False` (config time). Other Adam implementations pass over the moments several times.
- Parameters not on CUDA (optimizer construction). This includes `parallelism.fsdp_enable_cpu_offloading` and `create_seed_checkpoint` runs, which need `moment_location="device"`. A seed checkpoint is model-only, so the training run can still use `"pinned_host"`.
- Non-contiguous local parameter shards (optimizer construction).
- ROCm (config time). Pinning and aliasing host memory is not verified there.
- TorchFT (config time). Replica recovery is not verified.
