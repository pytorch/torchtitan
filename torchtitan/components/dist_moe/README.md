# Distributed MoE

TorchTitan can replace its standard routed experts with the CuTe DSL
[`dist-moe`](https://github.com/meta-pytorch/dist_moe) backend. The router
remains in TorchTitan and produces top-k expert IDs and scores. DistMoE then
owns dispatch, local expert compute, and combine as one ordered operation.

## Requirements

- NVIDIA SM100 or newer.
- `dist-moe`, installed through TorchTitan's normal requirements.
- `training.mixed_precision_param = "bfloat16"`.
- The PyTorch pipeline metadata and liveness APIs when pipeline parallelism is
  enabled.

The integration supports BF16 and asynchronous MXFP8 training. NVFP4 remains
an inference capability of the annex and is not exposed by TorchTitan.

## How To Configure DistMoE

DistMoE is selected through TorchTitan's model-transform interface. Apply
`DistMoeTransform` to replace every compatible `RoutedExperts` module with the
BF16 backend. Apply `MXFP8DistMoeTransform` after it to upgrade only those
routed experts to the asynchronous block-scaled backend. Dense attention,
shared-expert, feed-forward, and language-model-head linears remain independent
model-registry choices.

The DeepSeek V3 registry provides debug, 16B, and 671B reference recipes named
`*_dist_moe_bf16` and `*_dist_moe_mxfp8`. Start from those recipes when
possible. The examples below show the explicit transforms so that the memory
policy is visible.

### Capacity factor

Let `T` be the maximum tokens local to a rank after CP and sequence-parallel
sharding, `K` be top-k, and `P` be the expert-parallel degree. The EP group
creates `P * T * K` routed rows in total, so balanced routing receives `T * K`
rows per rank. `device_scratch_capacity_factor=f` reserves device scratch for
approximately `f * T * K` receive rows. Block-scaled kernels pad each local
expert to their M-tile size, so the exact planned capacity can be slightly
larger.

The largest possible receive count on one rank is:

```text
T * P * min(K, num_local_experts)
```

Therefore, the worst-case factor relative to balanced routing is:

```text
P * min(K, num_local_experts) / K
```

A factor of `1.0` is appropriate for forced-balanced routing. Real routing
needs measured headroom; `4.0` is a common deliberate choice, not a universal
default. Without VMM, exceeding the device factor is an error. With VMM, the
device factor remains the fast-HBM boundary and
`vmm.total_scratch_capacity_factor` is the larger device-plus-host correctness
boundary. Exceeding that total factor is still an error.

### Device-only example

This BF16 configuration supports real-routing imbalance up to factor four in
HBM. `saved_activation_buffer_bytes=None` selects the minimum correct saved
state: the inputs required to recompute every local MoE layer in backward.
Device scratch is planned separately from this value.

```python
from torchtitan.config.transform import DistMoeTransform, apply_transforms
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_16b

config = deepseek_v3_16b()
config = apply_transforms(
    config,
    [
        DistMoeTransform(
            device_scratch_capacity_factor=4.0,
            saved_activation_buffer_bytes=None,
            activation_slot_policy="auto",
            vmm=None,
        )
    ],
)
```

All saved activations and scratch are in local HBM. The symmetric dispatch and
combine buffers are separate peer-addressable HBM allocations. If the actual
receive rows exceed factor four, execution fails rather than allocating hidden
fallback storage.

To use asynchronous MXFP8 routed experts, append the precision transform:

```python
from torchtitan.config.transform import MXFP8DistMoeTransform

config = apply_transforms(
    config,
    [MXFP8DistMoeTransform(pipeline="staged", fast_math=False)],
)
```

`pipeline="staged"` uses separate expert projections. `pipeline="mega"` uses
the fused chunk-pipelined forward and fused DGRAD/WGRAD backward. Both use the
same planner and fixed CUDA-graph topology.

### Host-backed VMM example

This configuration keeps balanced scratch in HBM and covers imbalance through
factor four with pinned host-backed VMM pages. It is useful when reserving all
factor-four scratch in HBM would displace model state or saved activations.

```python
from torchtitan.config.transform import (
    DistMoeTransform,
    MXFP8DistMoeTransform,
    apply_transforms,
)
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_16b
from dist_moe import VmmConfig

config = deepseek_v3_16b()
config = apply_transforms(
    config,
    [
        DistMoeTransform(
            device_scratch_capacity_factor=1.0,
            saved_activation_buffer_bytes=None,
            activation_slot_policy="auto",
            vmm=VmmConfig(
                total_scratch_capacity_factor=4.0,
                prefetch=True,
            ),
        ),
        MXFP8DistMoeTransform(pipeline="staged", fast_math=False),
    ],
)
```

The activation arena remains in HBM. Scratch for receive factors in `(1, 4]`
addresses the host-backed middle section through the same stable CUDA virtual
range; pages do not migrate, and the CPU does not copy data during execution.
This preserves correctness but is slower than keeping the same rows in HBM.
`VmmConfig(prefetch=True)` constructs the exact mapping concurrently with
communication-buffer initialization inside `dist_moe.create_context()`. It is
not an execution-time prefetch and does not change kernel behavior.

### Choosing the saved-activation budget

The complete device allocation contains two rank-local regions:

```text
total device buffer = saved-activation bytes + device scratch bytes
```

Device scratch is fixed by shape, precision, kernel pipeline, and
`device_scratch_capacity_factor`. `saved_activation_buffer_bytes` controls only
the saved-activation region and is split evenly across the selected activation
slots. At the minimum, every slot can save the input of every assigned MoE
layer and backward recomputes the other expert intermediates. Increasing the
budget lets the device planner retain more intermediates and skip corresponding
recompute. At the maximum useful budget, all saveable intermediates fit;
additional bytes do not remove more work.

Start with `saved_activation_buffer_bytes=None`, inspect the logged
`Distributed MoE memory plan`, and use its exact minimum and maximum-useful
values to choose an explicit saved-state budget. The logged total device buffer
adds mandatory hot scratch to this value. VMM adds host overflow scratch and
never moves saved activations out of HBM.

### Transform settings

Common `DistMoeTransform` settings:

| Setting | Meaning and selection guidance |
| --- | --- |
| `device_scratch_capacity_factor` | HBM receive-row and scratch capacity relative to balanced routing. Use `1.0` for forced balance or a measured real-routing bound. |
| `saved_activation_buffer_bytes` | Aggregate saved-activation budget, excluding device scratch. `None` selects the minimum all-recompute plan; choose a logged value up to the maximum useful budget to retain more activations. |
| `activation_slot_policy` | PP lifetime granularity: `microbatch`, `stage_microbatch`, or `auto`. `auto` evaluates both finalized schedule plans and chooses the smaller `slots * layer_depth` allocation. |
| `num_activation_slots` | Optional lower bound on the schedule-derived slot count. Leave unset unless an external lifetime requires additional simultaneously live stacks. |
| `vmm` | Optional annex `VmmConfig`. Its total scratch factor must cover at least the device factor after EP-size clamping; `prefetch` controls initialization overlap, not capacity or placement. |
| `num_sms` | Expert override for SMs assigned to each DistMoE launch. `None` uses the annex default; tune only with shape-specific measurements. |
| `bf16_grouped_gemm_preset` | Expert BF16 CuTe schedule override. `None` uses the annex's shape-aware production configuration. |
| `wgrad_dtype` | Dtype of the W13/W2 gradient output: `"bfloat16"` or `"float32"`. It does not change tensor-core accumulation, which remains FP32. |
| `inplace_wgrad_accum` | Ask the annex WGRAD kernels to accumulate into the owned parameter-gradient destination. Enable only when the optimizer/FSDP lifecycle provides a stable compatible destination. |

MXFP8-only `MXFP8DistMoeTransform` settings:

| Setting | Meaning and selection guidance |
| --- | --- |
| `pipeline` | `"staged"` for separate projections or `"mega"` for fused chunk pipelines. Both support training and dynamic recompute. |
| `fast_math` | Use the approximate sigmoid in fused MXFP8 SwiGLU. This is an explicit numerical/performance choice and defaults to `False`. |
| `kernel_config` | Fully resolved `BlockScaledKernelConfig`. `None` selects the original shape-aware production presets. |

The MXFP8 transform inherits every common memory and WGRAD setting from the
BF16 transform it upgrades. It does not independently resize the arena.

## Module And Checkpoint Contract

`DistMoeTransform` accepts an ordinary `RoutedExperts.Config` containing
structured `GroupedLinear` W13/W2 projections, SwiGLU, and the standard
all-to-all dispatcher config. The transformed runtime module constructs only
the state it owns:

- `w13.weight`: structured `[E, 2, F, D]` gate/up weights.
- `w2.weight`: `[E, D, F]` down-projection weights.
- An optional `output_postprocess` module.

DistMoE does not construct the stock activation or token dispatcher because
its kernels own those operations. The common config still supplies shape,
top-k, sharding, and checkpoint metadata. Stock and DistMoE modules therefore
use the same `w13.weight` and `w2.weight` checkpoint and optimizer keys without
translation.

An expert-output postprocessor runs after W2 and before combine. Because that
boundary is inside DistMoE, the module must expose
`to_dist_moe_postprocess()` and return an annex-native descriptor. The
transform rejects unsupported modules; it does not fall back to a Python
callback or move the operation after combine.

## FSDP And MXFP8 Weights

BF16 parameters use the ordinary FSDP lifecycle. The MXFP8 module variant
reuses TorchTitan's generic prepared-weight lifecycle:

1. FSDP all-gathers the persistent high-precision shard in BF16.
2. The post-all-gather hook prepares grouped 32x32 MXFP8 qdata and FPROP/DGRAD
   scale layouts.
3. FSDP releases the temporary BF16 communication storage.
4. DistMoE consumes the prepared operands for the current unshard lifetime.
5. FSDP releases prepared-operand storage at its normal reshard boundary.

With `reshard_after_forward=False`, pipeline microbatches reuse one prepared
weight through backward. With `reshard_after_forward=True`, forward and
backward perform their normal separate unshards and preparations. Without
FSDP, the module prepares its ordinary parameter dynamically for each call.

## Runtime Setup And Memory Ownership

The trainer, rather than an individual expert layer, owns the rank-local
`DistMoeRuntime`. This ownership matches the resource: its memory plan depends
on every local MoE layer, the EP group, and the finalized PP schedule, and its
symmetric/VMM allocations must be created and released exactly once. Keeping
the runtime on the trainer also makes independent trainer and test lifetimes
explicit instead of relying on implicit process-global state.

The standard trainer performs two explicit setup steps:

1. After model parallelization and PP schedule construction,
   `prepare_dist_moe_runtime()` derives the schedule-aware activation plan,
   resolves one shared runtime configuration, attaches it to every local
   DistMoE layer, and registers PP stage-forward contexts.
2. After parameters and buffers have been materialized, the trainer calls
   `DistMoeRuntime.initialize()` to construct the annex context. The annex
   plans memory and owns any configured VMM prefetch, allocation, and cleanup.

All modes attach the same runtime reference to their local DistMoE modules. PP
additionally registers one forward context on each participating
``PipelineStage`` to select its immutable stage/microbatch activation slot.
Recipe authors configure transforms; they do not call either runtime setup
method.

The annex owns four distinct allocations:

| Allocation | Placement | Lifetime and contents |
| --- | --- | --- |
| Symmetric communication buffers | Peer-addressable GPU HBM | Context lifetime; route metadata, signals, dispatch inputs, combine outputs, and their gradients. |
| Activation arena | Rank-local GPU HBM | Context lifetime, suballocated per live activation slot; mandatory layer inputs and any intermediates selected for saving. |
| Device scratch | Rank-local GPU HBM | Shared by one local layer action at a time; forward temporaries, recompute temporaries, and activation gradients. |
| VMM overflow scratch | Pinned host memory mapped into the same CUDA virtual range | Optional context lifetime; only scratch demand above the device factor and within the total factor. |

The planner logs device activations, device scratch, host overflow, and the
total virtual-address budget. Within a selected activation slot, DistMoE may
save an intermediate or recompute it in backward according to available
capacity; this does not change CUDA-graph topology.

VMM prefetch is resource preparation, not CPU/GPU synchronization or demand
paging. The annex owns the single-use mapping and closes it if context creation
fails; TorchTitan never holds or transfers the private prefetch handle. See the annex
[memory-planner guide](https://github.com/meta-pytorch/dist_moe/blob/main/docs/memory_planner.md)
for the byte-level layout, offset protocol, and dynamic-recompute algorithm.

## Pipeline Parallelism

For eager multi-stage schedules, TorchTitan asks PyTorch to analyze the final
schedule IR before warmup. It colors overlapping resource lifetimes into
deterministic reusable slots. In `auto` mode, DistMoE compares:

- `microbatch`: all local stages for one microbatch share a deeper stack.
- `stage_microbatch`: each live `(stage, microbatch)` pair uses a shallower
  stack sized for the largest local stage.

The smaller `slots * layer_depth` plan wins. PyTorch invokes each participating
stage's registered forward context with a `PipelineStageInfo` containing the
stage and microbatch indices. The context selects the immutable slot before
ordinary model forward. This works with eager PP and whole-step eager-PP CUDA
graphs without reserved model kwargs or a model-specific `PipelineStage`
subclass.

GraphPP does not yet carry schedule-action metadata into reusable stage graphs.
It therefore cannot use schedule-derived DistMoE pipeline slots until that
[graph-runtime contract](https://github.com/pytorch/torchtitan/issues/4655) is
implemented. Non-pipeline GraphTrainer remains supported.

## Lifecycle And Failure Behavior

TorchTitan validates model-known requirements in `Trainer.Config`, including
the BF16 FSDP communication dtype. Setup validates resolved topology, device,
token divisibility, and shared context policy. Unsupported hardware,
postprocessors, schedules, or inconsistent local layer configurations fail
explicitly rather than selecting a slower fallback.

If model-state initialization fails after VMM preparation, the trainer closes
the pending mapping before propagating the exception. Context construction also
closes an unconsumed mapping on failure. Normal trainer teardown calls the
idempotent `DistMoeRuntime.close()` exactly once to remove PP hooks and release
symmetric, activation, scratch, and VMM resources. Expert modules do not own or
recursively close this shared state. Kernel algorithms and allocator internals
are documented in the annex; this guide defines only TorchTitan ownership and
composition.
