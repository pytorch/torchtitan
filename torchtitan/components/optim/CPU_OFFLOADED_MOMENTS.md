## CPU-Offloaded Adam Moments

Set an Adam/AdamW config's `moment_cpu_offload_fraction` to keep that share of the local `exp_avg` and `exp_avg_sq` bytes in pinned CPU memory instead of accelerator memory. `0.0` (the default) keeps all moments on the device. `1.0` offloads all of them. The unchanged fused Adam kernel reads and writes the offloaded moments in place over the CPU-GPU link. There are no copies, side streams, or extra kernels, and the step stays CUDA-graph capturable.

```python
from dataclasses import replace

optimizers = config.optim.optimizer.optimizers
optimizers[0] = replace(optimizers[0], moment_cpu_offload_fraction=0.5)
```

`replace` runs the config checks below. Assigning the field directly skips them.

The unit is a whole parameter. The optimizer walks its parameters in order and offloads each one's moments until the offloaded bytes reach the fraction of all local moment bytes. The rest stay on the device. FSDP shards evenly and experts have equal sizes, so all ranks usually pick the same parameters. Offloaded and device moments share one param group and one fused AdamW call.

### Allocator setting

By default, the PyTorch pinned allocator rounds each allocation up to a power of two, which can double the host memory of the moments. Turn the rounding off for large allocations when you offload moments:

```bash
PYTORCH_ALLOC_CONF="expandable_segments:True,pinned_max_round_threshold_mb:1" ./run_train.sh
```

If you already set `PYTORCH_ALLOC_CONF`, append `pinned_max_round_threshold_mb:1` to your value. `run_train.sh` keeps your value and uses `expandable_segments:True` only when it is unset. With this setting, allocations above 1 MB are pinned at their exact size instead of rounded up to a power of two. Smaller allocations are still rounded. Allocations above `pinned_max_cached_size_mb`, if it is set, are never rounded. After allocating the moments, the optimizer compares the pinned bytes that `torch.cuda.host_memory_stats()` reports with the requested bytes. If rounding wasted more than 1 GiB, it raises `ValueError`.

### NUMA binding

`TrainingEngine` (`torchtitan/training_engine.py`) calls `maybe_apply_numa_binding` to bind each rank to its GPU's NUMA node before anything allocates pinned memory, so the moments land on the GPU's node. Without binding, on GB300 with four GPUs busy, copies in both directions at once dropped about 2-2.5x (86-111 GB/s to 44 GB/s). Host-to-device copies alone dropped 1.0-1.8x.

### Choosing the fraction

Use the smallest fraction that fits, because every offloaded byte costs step time. For example, raise it until a run that goes out of memory at the first optimizer step fits. The feature is designed for hosts with a fast CPU-GPU link, such as GB300. It has not been measured on PCIe hosts, where the step will be much slower.

Activation offloading (torchtitan#6020) is complementary: it moves activations, not optimizer state. Both use the same pinned allocator and NUMA binding.

### Cost

Each optimizer step reads and writes every offloaded moment once over the host link:

```
bytes per step = 2 (read + write) x 2 moments x moment bytes x offloaded parameters
               = 16 bytes per offloaded parameter for fp32 moments
```

The AdamW time grows by about these bytes divided by the host link bandwidth. Accelerator memory drops, and pinned host memory grows, by `2 x moment bytes` per offloaded parameter. On GB300, expect about 13.5 ms per step per GiB of offloaded moments.

On one GB300 GPU bound to its NUMA node, a 64-GPU fake-process-group Muse Spark run (DP8/CP8/EP64, 32K sequence length, all moments offloaded) moves 2 x 104 GB per AdamW step. AdamW alone takes about 0.63 s (about 330 GB/s), 5.8% of the step, in one synchronized measurement. The two GPUs of a Grace socket share its host bandwidth, so when both run the step at the same time, AdamW takes about 2x longer.

### How it works

The optimizer creates the state of the offloaded parameters at construction. Each moment is a CPU tensor from `torch.zeros(..., pin_memory=True)`, exposed to CUDA as a tensor that aliases that memory (`__cuda_array_interface__`). A DTensor parameter gets DTensor moments with the same mesh and placements. The other parameters get device moments lazily, at the first step.

```
GPU: param, grad --fused Adam--> param
                     ^    |
                     |    v  host link, in place
CPU (pinned):  exp_avg, exp_avg_sq
```

DCP saves and loads offloaded moments like device moments, so checkpoints are interchangeable between any two fractions. Loading writes into the same pinned buffers, without a temporary copy in accelerator memory.

`optimizer.state_dict()` returns each offloaded moment as the CPU tensor over its pinned memory, so checkpointers copy it on the CPU. A CUDA copy between two host buffers is synchronous and much slower. `optimizer.load_state_dict()` copies loaded values into that memory if they are other tensors, and keeps the CUDA moments in the optimizer state.

### Requirements

Rejected with `ValueError` when the fraction is above 0:

- `fused=False` (config time). Other Adam implementations pass over the moments several times.
- An accelerator other than CUDA, including ROCm (config time).
- Offloaded parameters not on CUDA (optimizer construction). This includes `parallelism.fsdp_enable_cpu_offloading` and `create_seed_checkpoint` runs, which need `moment_cpu_offload_fraction=0`. A seed checkpoint is model-only, so the training run can still offload.
- Non-contiguous local shards of offloaded parameters (optimizer construction).
- Pinned memory rounded up by more than 1 GiB (optimizer construction). See [Allocator setting](#allocator-setting).
- TorchFT (config time). Replica recovery would replace the pinned moments with device tensors.

A fraction outside [0, 1] is also rejected at config time.
