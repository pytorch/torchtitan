# Distributed MoE

TorchTitan can replace the stock routed-experts module with the CuTe DSL
`dist_moe` backend. The router stays in TorchTitan: it computes top-k expert
IDs and scores, then Dist-MoE fuses dispatch, local expert compute, and combine
into its distributed kernels.

## Requirements

- NVIDIA SM100 or newer.
- A PyTorch build containing the pipeline schedule resource-planning APIs used
  by pipeline parallelism.
- The `dist_moe` package and its pinned CuTe DSL dependency.
- `training.mixed_precision_param = "bfloat16"`.

The backend supports BF16 expert compute and asynchronous MXFP8 expert compute.
NVFP4 is currently an annex inference capability and is not exposed by the
TorchTitan training integration.

## Enabling the backend

Dist-MoE is a model-config converter. Add it after any converters that target
ordinary dense linear modules:

```python
from torchtitan.components.dist_moe import (
    DistMoeBackendConfig,
    DistMoeConverter,
)

model_spec = model_registry(
    "16B",
    attn_backend="varlen",
    converters=[
        DistMoeConverter.Config(
            backend=DistMoeBackendConfig(
                dtype="mxfp8",
                max_routing_imbalance_factor=4.0,
            )
        )
    ],
)
```

The DeepSeek V3 registry includes debug, 16B, and 671B BF16 and MXFP8
reference recipes named `*_dist_moe_bf16` and `*_dist_moe_mxfp8`. Those
recipes use CUDA-graph-compatible varlen attention. The MXFP8 recipes also use
TorchTitan's `MXFP8LinearConverter` for attention projections, shared experts,
dense feed-forward layers, and the language-model head.

## Model and checkpoint contract

`DistMoeConverter` accepts the stock `RoutedExperts.Config` backed by
`GroupedExperts` and `AllToAllTokenDispatcher`. It replaces the runtime module
with `DistMoeRoutedExperts`, which directly owns:

- `w13_EGFD`: fused gate and up-projection weights.
- `w2_EDF`: down-projection weights.

There is deliberately no runtime `inner_experts` child. Generic TorchTitan
code finds the true parameter owner through `expert_parameters_module()`.
State-dict and optimizer-state hooks translate the fused representation back
to the stock `inner_experts.w{1,2,3}_EFD` keys. A stock checkpoint therefore
loads into Dist-MoE, and a Dist-MoE checkpoint remains consumable by the stock
grouped-experts backend.

## FSDP and MXFP8 weights

BF16 parameters follow the ordinary FSDP lifecycle. MXFP8 reuses TorchTitan's
generic `_ShardedFSDPTensor` post-all-gather contract:

1. FSDP all-gathers the persistent BF16 shard.
2. The post-all-gather hook creates grouped 32x32 MXFP8 qdata plus FPROP and
   DGRAD scale layouts.
3. The temporary BF16 communication storage is released.
4. Dist-MoE consumes the prepared operands while that FSDP unshard is live.
5. FSDP releases the prepared operands at its normal reshard boundary.

With `reshard_after_forward=False`, one prepared weight is reused by every
pipeline microbatch until backward finishes. With
`reshard_after_forward=True`, forward and backward perform their normal
separate unshards and quantizations. Without FSDP, the module dynamically
prepares its ordinary parameter before each invocation.

## Runtime and memory ownership

`setup_dist_moe()` runs after model parallelization and before parameter
materialization. It validates that local layers share one expert-parallel
group and backend policy, derives the local token capacity after CP/SP
sharding, asks the annex for a memory plan, and creates one context shared by
all local Dist-MoE layers.

The annex owns the symmetric communication buffer, activation arena, device
scratch, and optional host-backed VMM overflow. The important controls are:

| Setting | Meaning |
| --- | --- |
| `max_routing_imbalance_factor` | Device receive-row and scratch capacity relative to balanced routing. Exceeding the configured total capacity is an error. |
| `device_memory_budget_bytes` | Optional total device budget. The planner uses any budget above its minimum for saved activations. |
| `vmm_host_scratch_imbalance_factor` | Total device-plus-host scratch imbalance covered by VMM. The default, `None`, uses ordinary device allocation and disables host overflow. |
| `prefetch_vmm` | Allocate an enabled VMM arena asynchronously during model setup. This is an independent opt-in and defaults to `False`; enabling it while VMM is disabled is invalid. |
| `activation_slot_policy` | Pipeline activation ownership: `microbatch`, `stage_microbatch`, or the lower-memory `auto` choice. |
| `num_activation_slots` | Optional lower bound overriding the schedule-derived slot count. |
| `num_sms` and kernel configs | Expert controls for launch resources and tuned schedules. |

The planner logs the resolved device activations, device scratch, host
overflow, and total virtual-address budget. Dist-MoE may dynamically save an
intermediate in the activation arena or recompute it during backward according
to the capacity available in the selected slot; this does not change kernel
topology under CUDA graphs.

## Pipeline parallelism

For a multi-stage pipeline schedule, TorchTitan asks PyTorch to analyze the
final schedule IR before warmup. The analysis computes deterministic resource
lifetimes and colors overlapping lifetimes into reusable slots. Dist-MoE
compares two exact plans in `auto` mode:

- `microbatch`: all local logical stages for a microbatch share one deeper
  stack.
- `stage_microbatch`: each live `(stage, microbatch)` pair uses a shallower
  stack sized for the largest local stage.

The plan with the smaller `slots * layer_depth` allocation wins. PyTorch passes
`pipeline_stage_index` and `pipeline_microbatch_index` as canonical keyword
arguments on every stage forward. A stage-root pre-hook selects the immutable
slot before any activation-checkpointed layer executes. This placement makes
the original forward and backward recomputation observe the same selected
storage and works for eager pipeline schedules without a TorchTitan-specific
`PipelineStage` subclass.

Pipeline CUDA graphs additionally require a multi-stage schedule, static input
shapes, and per-direction P2P communicators. The reference recipes use fixed-
capacity varlen-attention metadata so document packing cannot change captured
input shapes between steps.

## Lifecycle and failure behavior

`cleanup_dist_moe()` closes each distinct context during trainer teardown,
releasing symmetric and VMM storage. Configuration and topology mismatches fail
before training: unsupported hardware, non-BF16 mixed-precision parameters,
incompatible routed-expert implementations, inconsistent local policies,
non-divisible CP/SP token sharding, or pipeline schedules without a static
liveness plan are not silently downgraded.

The annex documentation describes the kernel and memory algorithms in detail;
this document covers only their TorchTitan ownership and composition.
