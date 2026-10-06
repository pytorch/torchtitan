# Distributed MoE

TorchTitan can replace standard routed experts with the CuTe DSL
[`dist-moe`](https://github.com/meta-pytorch/dist_moe) backend. The router stays
in TorchTitan and produces top-k expert IDs and scores. Dist-MoE owns token
dispatch, grouped expert computation, activation storage, scratch storage, and
combine as one ordered operation.

TorchTitan supports BF16 and asynchronous MXFP8 Dist-MoE training on NVIDIA
SM100 or newer. The annex also provides NVFP4 inference, which TorchTitan does
not expose yet. Dist-MoE requires
`training.mixed_precision_param="bfloat16"` because FSDP unshards its persistent
parameters in BF16.

## Installation

Dist-MoE is optional and released independently from TorchTitan. Install its
latest nightly directly from the public Git repository before selecting a
Dist-MoE recipe:

```bash
python -m pip install \
  "git+https://github.com/meta-pytorch/dist_moe.git@main"
```

Ordinary TorchTitan imports and recipes do not import Dist-MoE. A selected
Dist-MoE recipe performs a guarded import and reports a focused error when the
package is unavailable. TorchTitan's base requirements intentionally do not
install this Blackwell-only backend.

## Select A Backend

Choose the expert precision on one transform, and configure the rank-wide
runtime separately on the trainer:

```python
from torchtitan.config.transform import apply_transforms
from torchtitan.config.transform.dist_moe import DistMoeTransform
from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime
from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_16b

config = deepseek_v3_16b()
config.dist_moe = DistMoeRuntime.Config(
    scratch_capacity_factor=4.0,
    activation_slot_capacity_factor=1.0,
    pp_activation_slot_policy="stage_microbatch",
)
config = apply_transforms(config, [DistMoeTransform()])
```

For MXFP8 experts, set `expert_precision="mxfp8"`. Dense attention,
shared-expert, feed-forward, and language-model-head linears remain separate
quantization choices:

```python
import dist_moe

from torchtitan.config.transform import apply_transforms
from torchtitan.config.transform.dist_moe import DistMoeTransform
from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime
from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_16b

config = deepseek_v3_16b()
config.dist_moe = DistMoeRuntime.Config(
    scratch_capacity_factor=4.0,
    activation_slot_capacity_factor=1.0,
)
config = apply_transforms(
    config,
    [
        DistMoeTransform(
            expert_precision="mxfp8",
            block_scaled_config=dist_moe.BlockScaledConfig(
                pipeline="staged",
                fast_math=False,
            ),
        )
    ],
)
```

Verified 671B recipes live in `torchtitan_recipes.models.deepseek_v3`, while
debug and 16B CI recipes live in `torchtitan_recipes.tests.models.deepseek_v3`.
GraphTrainer follows the corresponding `graph_trainer` and
`tests.graph_trainer` ownership. Recipes use `_dist_moe_bf16` and
`_dist_moe_mxfp8` suffixes and select CUDA-graph-compatible varlen attention.
MXFP8 recipes also convert dense projections and the language-model head.

## Configuration Ownership

`DistMoeTransform` replaces each stock `RoutedExperts.Config` directly with the
selected BF16 or MXFP8 implementation. The replacement keeps TorchTitan's
structured `GroupedLinear` parameter layout:

- `w13.weight` has shape `[E, 2, F, D]`.
- `w2.weight` has shape `[E, D, F]`.
- An optional `output_postprocess` remains a normal TorchTitan module.

This preserves parameter, optimizer, FSDP, and checkpoint ownership. The
transformed module does not construct the stock activation or token dispatcher,
because the annex performs those operations internally.

Per-layer settings belong to the transform:

| Setting | Meaning |
| --- | --- |
| `expert_precision` | Select `"bf16"` (default) or the asynchronous `"mxfp8"` expert path. |
| `inplace_wgrad_accum` | Let annex WGRAD kernels accumulate directly into standard `parameter.grad` storage. This is enabled by default for BF16 and MXFP8 experts. GraphTrainer recipes temporarily disable it because tracing functionalizes parameters into graph inputs; a graph pass must bind those inputs to their leaf `parameter.grad` destinations before enabling accumulation. |
| `bf16_grouped_gemm_preset` | Optional BF16 FPROP/DGRAD schedule override for expert users. `None` uses the annex's shape-aware production defaults. BF16 WGRAD has its own production schedule. |
| `block_scaled_config` | MXFP8-only annex policy. `pipeline="staged"` uses separate expert kernels; `pipeline="mega"` uses the fused chunk-pipelined implementation. `fast_math` selects approximate fused-SwiGLU sigmoid math, and `kernel_config` is an expert-only CuTe tuning override. |

One `DistMoeRuntime.Config` owns rank-wide resources shared by every local
Dist-MoE layer:

| Setting | Meaning |
| --- | --- |
| `activation_slot_bytes` | Exact saved-forward-state capacity requested for each live activation slot, excluding scratch. Set `activation_slot_capacity_factor=None` when using this expert-level byte override. |
| `activation_slot_capacity_factor` | Per-slot saved-state capacity relative to balanced routing. `1.0` retains every eligible intermediate when aggregate slot usage is balanced. It is `None` by default and mutually exclusive with `activation_slot_bytes`. |
| `scratch_capacity_factor` | Routing imbalance that must fit entirely in HBM scratch. `1.0` covers balanced `local_tokens * top_k` routing. |
| `vmm_capacity_factor` | Optional total device-plus-host scratch bound. `None` disables VMM; a value larger than `scratch_capacity_factor` provides host-backed overflow capacity. Saved activations remain in HBM. |
| `pp_activation_slot_policy` | `"stage_microbatch"` reuses slots at stage-microbatch lifetime; `"microbatch"` retains one deeper slot across all local stages for a microbatch. The default is `"stage_microbatch"`. |

The recipe assigns this config to the optional `TrainingEngine.Config.dist_moe`
field, whose base type is `Configurable.Config | None`. The concrete config's
`build()` method constructs `DistMoeRuntime` only when the recipe selects it.
Forward/backward initialization supplies materialized model parts, topology,
schedule, and execution-mode registration to the runtime. Model dimensions,
exact local token count, expert metadata, live slots, layers per slot, and kernel
precision are derived rather than duplicated in the recipe. Eager execution
leaves the annex WGrad dtype unset, so the annex resolves it from each live
parameter's declared gradient dtype, existing gradient storage, or parameter
dtype. GraphTrainer uses the BF16 mixed-precision parameter dtype for its
functional WGrad output because prepared MXFP8 graph operands do not expose a
live parameter destination. Current FSDP casts the resulting local BF16 WGrad
to `training.mixed_precision_reduce` for reduction. If FSDP later declares an
FP32 parameter gradient dtype, eager Dist-MoE will use it without a separate
TorchTitan override.

## Scratch Capacity

Let `T` be the exact local input token count after CP and sequence-parallel
sharding, `K` be top-k, and `P` be the expert-parallel degree. Balanced routing
receives `T * K` rows on each rank. The largest possible receive count is:

```text
T * P * min(K, num_local_experts)
```

The corresponding worst-case imbalance relative to balanced routing is:

```text
P * min(K, num_local_experts) / K
```

`scratch_capacity_factor` sets the HBM-resident receive and scratch
bound. It does not truncate or rebalance routing. A factor of `1.0` is suitable
for forced-balanced routing. Real routing needs measured headroom; factor four
is a deliberate recipe choice, not a universal default. Block-scaled kernels
also pad each local expert to their M-tile size, so planned rows can be slightly
larger than the logical bound.

Without VMM, exceeding the device factor is an error. With VMM,
`vmm_capacity_factor` is the larger device-plus-host correctness bound.
Exceeding that bound is still an error.

## Saved Activations And Pipeline Slots

The annex allocates four distinct regions:

| Allocation | Placement | Contents and lifetime |
| --- | --- | --- |
| Symmetric communication buffers | Peer-addressable HBM | Routing metadata, signals, dispatch inputs, combine outputs, and gradients for the context lifetime. |
| Saved-activation region | Rank-local HBM | Mandatory layer inputs and any expert intermediates selected for saving, partitioned into live activation slots. |
| Device scratch | Rank-local HBM | Forward temporaries, recompute temporaries, and activation gradients, reused by one local layer action at a time. |
| VMM overflow scratch | Pinned host memory in the same CUDA virtual range | Optional scratch demand above the device factor and within the total factor. |

Without PP there is one activation slot containing all local MoE layers. With
eager PP, PyTorch analyzes the final schedule before warmup and assigns every
live `(stage, microbatch)` interval to a reusable slot. The default
`"stage_microbatch"` policy sizes each slot for the largest local stage and
reuses it after that stage's backward releases the saved state. The
`"microbatch"` policy keeps one slot across all local stages for a microbatch
and therefore sizes each slot for their combined MoE depth.

Activation capacity is configured independently for each live slot. Set
`activation_slot_bytes` for an exact logical byte budget, or use
`activation_slot_capacity_factor` to scale optional saved state above the
mandatory layer inputs. The controls are mutually exclusive. When both are
`None`, the annex selects its minimum correct plan: every assigned layer input
remains saved and other expert intermediates are dynamically recomputed in
backward. Increasing either policy can retain more intermediates and reduce
recomputation up to the planner's logged maximum useful per-slot budget.

PyTorch calls each eager pipeline stage's registered forward context with its
stage and microbatch indices. `DistMoeRuntime` uses the precomputed assignment
to select the annex activation slot before model execution. Every slot is
planned for `max_moe_layers_per_activation_slot`, so only the slot ID varies.
This does not add model kwargs or require a custom pipeline-stage subclass.

GraphPP reuses the same runtime-owned slot resolver without entering the eager
stage context. During tracing, TorchTitan supplies one explicit
`activation_slot_id_1` input and rewrites the exact Dist-MoE forward operations
to consume it. At execution, the forward-bearing schedule action resolves its
precomputed physical slot and passes the corresponding immutable device-scalar
view. This produces one graph per stage without model kwargs, per-microbatch
graph variants, or a device-side indexing operation. Dist-MoE backward uses its
saved forward state and therefore needs no second slot input.

The annex's functional and accumulating backward operations are both visible to
non-strict FX tracing. In-place accumulation passes its destination explicitly
to the registered backward operation, so eager and graph execution retain the
same standard `parameter.grad` ownership contract.

## VMM Overflow Scratch

VMM keeps the fast scratch prefix in HBM and maps additional pinned host pages
into the same stable CUDA virtual range:

```python
from torchtitan.config.transform import apply_transforms
from torchtitan.config.transform.dist_moe import DistMoeTransform
from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime
from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_16b

config = deepseek_v3_16b()
config.dist_moe = DistMoeRuntime.Config(
    activation_slot_capacity_factor=1.0,
    scratch_capacity_factor=1.0,
    vmm_capacity_factor=4.0,
)
config = apply_transforms(config, [DistMoeTransform()])
```

Scratch above factor one and at most factor four can use host-backed pages. No
saved activation is moved to host memory. The runtime uses the annex's default
prefetch lifecycle, which overlaps physical VMM allocation with communication-
buffer initialization during context creation without changing capacity,
steady-state placement, kernel behavior, or execution-time synchronization.

## FSDP, MXFP8, And Rematerialization

BF16 weights use the ordinary FSDP lifecycle. MXFP8 experts reuse TorchTitan's
prepared-weight tensor lifecycle:

1. FSDP all-gathers the persistent high-precision shard in BF16.
2. The post-all-gather hook prepares grouped 32x32 MXFP8 qdata and FPROP/DGRAD
   scale layouts.
3. FSDP releases the temporary BF16 communication storage.
4. Dist-MoE consumes the prepared operands for that unshard lifetime.
5. FSDP releases prepared storage at the normal reshard boundary.

With `reshard_after_forward=False`, pipeline microbatches reuse one prepared
weight through backward. With `reshard_after_forward=True`, forward and
backward perform separate unshards and preparations. Without FSDP, the module
prepares the live parameter directly for each call.

The complete `dist_moe.routed_experts` call is an ordered operation wrapped in
a non-recomputed remat region. Dist-MoE owns its device-side decision to save or
recompute expert intermediates; an outer activation-checkpoint policy must not
duplicate dispatch, communication, or arena mutation.

## Expert Output Postprocessing

The common routed-expert API can own an `output_postprocess` module that runs
after W2 and before score-weighted top-k combine. TorchTitan keeps that module's
parameter, optimizer, checkpoint, and FSDP ownership. At each forward the
Dist-MoE adapter converts its current parameter to the annex's typed
`RMSNormPostprocess` execution descriptor. Unsupported postprocessors fail
during configuration instead of running after combine or falling back to an
unfused callback.

## Runtime Lifecycle

TorchTitan owns one optional Dist-MoE runtime through forward/backward setup:

1. A recipe assigns `DistMoeRuntime.Config` to `TrainingEngine.Config.dist_moe`.
2. The model transform independently replaces routed-expert modules.
3. At the beginning of forward/backward initialization, the standard or graph
   engine supplies final model parts, topology, liveness schedule, and its
   narrow forward-context registration capability.
4. The runtime creates one annex context, derives slot assignments, installs
   eager-stage hooks or the GraphPP resolver, retains removal ownership, and
   attaches non-owning references to every local Dist-MoE module.
5. GraphPP supplies the resolver's immutable slot views as explicit graph
   inputs; eager PP enters the registered stage context.
6. Partial initialization unwinds runtime-owned resources. Normal teardown
   removes registrations, detaches modules, and closes the Annex context.

The RL vLLM generator owns an inference runtime
(`DistMoeRuntime.Config(inference=True)`, passed as
`VLLMGenerator.Config.dist_moe_runtime`): scratch-only planning, sized for
vLLM's per-step token budget. vLLM pads every step at the end (TP rounding,
CUDA-graph size, DP maximum); the runner publishes the unpadded count and the
model passes a `padding_mask` down to the routed experts.

## Padding Rows

Rows marked in `padding_mask` get expert ID `-1` and score zero, in training and
inference. Dist-MoE never dispatches a `-1` route, so padding costs no
communication or expert GEMM work and produces zero output rows. Every EP rank
must pass the same row count to a call:

- Training passes its microbatch rows unchanged.
- Inference calls with the largest TP-local token count across vLLM DP
  replicas (graph steps already share one size; eager steps do not), padded
  with `-1` rows. In batch-invariant mode calls are raised to more than 128
  rows: at or below 128 rows the annex's post-expert top-k reduction uses
  another, equally correct, accumulation order, so a token's output would
  depend on how many tokens share its step.

The annex context owns symmetric buffers, activation storage, scratch storage,
VMM allocation, and VMM prefetch. TorchTitan never accesses their private
representations. See the annex
[memory planner](https://github.com/meta-pytorch/dist_moe/blob/main/docs/memory_planner.md),
[pipeline slots](https://github.com/meta-pytorch/dist_moe/blob/main/docs/pipeline_activation_slots.md),
and [VMM](https://github.com/meta-pytorch/dist_moe/blob/main/docs/vmm.md)
documentation for byte-level layouts and planner behavior.
