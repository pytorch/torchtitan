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

## Configure DistMoE

DistMoE uses the standard model-transform interface. Apply the BF16 transform
after constructing the complete trainer config. Add the MXFP8 transform to
upgrade only the routed experts; dense MXFP8 linears remain an independent
model-registry choice.

```python
from torchtitan.config.transform import (
    apply_transforms,
    DistMoeTransform,
    MXFP8DistMoeTransform,
)

config = deepseek_v3_16b()
config = apply_transforms(
    config,
    [
        DistMoeTransform(max_routing_imbalance_factor=4.0),
        MXFP8DistMoeTransform(fast_math=True),
    ],
)
```

The DeepSeek V3 registry provides debug, 16B, and 671B reference recipes named
`*_dist_moe_bf16` and `*_dist_moe_mxfp8`. They use CUDA-graph-compatible
varlen attention. MXFP8 recipes independently quantize eligible attention,
shared-expert, dense feed-forward, and language-model-head linears.

## Module And Checkpoint Contract

`DistMoeTransform` accepts an ordinary `RoutedExperts.Config` containing
structured `GroupedLinear` W13/W2 projections, SwiGLU, and the standard
all-to-all dispatcher config. The transformed runtime module constructs only
the state it owns:

- `w13.weight`: structured `[E, 2, F, D]` gate/up weights.
- `w2.weight`: `[E, D, F]` down-projection weights.
- An optional expert-output postprocess module.

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

## Runtime And Memory Ownership

`setup_dist_moe()` runs once after model parallelization and before parameter
materialization. Its phases are explicit:

1. Collect each local DistMoE module once.
2. Validate the resolved runtime and hardware invariants.
3. Derive topology, token capacity, and schedule-aware activation ownership.
4. Bind one shared context and its pipeline metadata hooks.

The annex owns the symmetric communication buffer, activation arena, device
scratch, and optional host-backed VMM overflow. TorchTitan exposes these
controls through `DistMoeTransform`:

| Setting | Contract |
| --- | --- |
| `max_routing_imbalance_factor` | Device receive-row and scratch capacity relative to balanced routing. Exceeding total configured capacity is an error. |
| `device_memory_budget_bytes` | Optional total device arena budget. Space above the minimum may retain more activations. |
| `vmm_host_scratch_imbalance_factor` | Total device-plus-host scratch imbalance covered by VMM. `None` disables VMM and host overflow. |
| `prefetch_vmm` | Prepare enabled host-backed VMM mappings during setup before context creation and graph capture. It requires VMM. |
| `activation_slot_policy` | `microbatch`, `stage_microbatch`, or `auto`, which selects the smaller exact schedule-derived allocation. |
| `num_activation_slots` | Optional lower bound on the schedule-derived slot count. |
| `num_sms` | Optional expert override for SMs assigned to DistMoE kernels. |
| `kernel_config` | Optional BF16 or MXFP8 kernel schedule on the corresponding transform. |
| `wgrad_dtype` | BF16 or FP32 WGRAD output. |
| `inplace_wgrad_accum` | Accumulate WGRAD directly into owned parameter gradients. |

The planner logs device activations, device scratch, host overflow, and the
total virtual-address budget. Within a selected activation slot, DistMoE may
save an intermediate or recompute it in backward according to available
capacity; this does not change CUDA-graph topology.

VMM prefetch is resource preparation, not CPU/GPU synchronization or demand
paging. A prefetched mapping is single-use and must exactly match context
creation. Failed, closed, consumed, or mismatched mappings are errors and do
not silently allocate a second arena.

## Pipeline Parallelism

For eager multi-stage schedules, TorchTitan asks PyTorch to analyze the final
schedule IR before warmup. It colors overlapping resource lifetimes into
deterministic reusable slots. In `auto` mode, DistMoE compares:

- `microbatch`: all local stages for one microbatch share a deeper stack.
- `stage_microbatch`: each live `(stage, microbatch)` pair uses a shallower
  stack sized for the largest local stage.

The smaller `slots * layer_depth` plan wins. A participating pipeline stage
requests canonical `pipeline_stage_index` and `pipeline_microbatch_index`
keyword metadata. A stage-root pre-hook selects the immutable slot and removes
the reserved metadata before ordinary model forward. This works with eager PP
and whole-step eager-PP CUDA graphs without a model-specific `PipelineStage`
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

Trainer teardown calls the generic, idempotent `Module.close()` lifecycle.
DistMoE uses it to remove pipeline hooks and release symmetric and VMM storage.
Kernel algorithms and allocator internals are documented in the annex; this
guide defines only TorchTitan ownership and composition.
