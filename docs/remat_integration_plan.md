# Remat Integration Plan

This is the running inventory and implementation plan for integrating
`torch_remat` across TorchTitan models. Update this document whenever a model
adds, removes, or renames a remat save region.

## Current contract

- `RematAC` checkpoints each child of `model.layers`.
- Save-region names are qualified relative to one transformer block.
- The same `save_regions` policy is applied to every transformer block.
- A module exposes local region names through
  `AVAILABLE_REMAT_SAVE_REGIONS` and wraps the corresponding call sites with
  `maybe_remat_save_region`.
- A module lists correctness-sensitive regions in
  `REQUIRED_REMAT_SAVE_REGIONS`. `RematAC` retains these automatically, in
  addition to regions selected by the caller, and reports them separately.
- Outside required correctness boundaries, models expose available save
  regions rather than a default policy. The caller owns the memory/performance
  policy passed to `RematAC.Config`.
- If a subclass overrides `forward`, it must redeclare and implement its save
  regions. Inherited names are intentionally ignored because they may no longer
  describe the active forward implementation.
- `maybe_remat_recompute_needs` keeps selected-region outputs alive when a bare
  operation outside a remat region consumes them. Each call site still needs a
  consumer-side memory audit.

The common token-choice router requires its narrow `routing_decision` region.
It includes node-limited group selection and final expert `topk`, but not the
router gate or full score tensor. Replaying a nondeterministic tie could
otherwise route backward through different experts than forward.

Relevant implementation files:

- `torchtitan/distributed/activation_checkpoint.py`
- `torchtitan/models/common/remat.py`
- `torchtitan/models/deepseek_v3/remat.py`

The DeepSeek file contains a specifically named policy copied from NVIDIA
Megatron's H100 configuration. It is not a general DeepSeek default, and it
does not select every available model boundary. Its `attention.wq*` pattern
covers either the direct query projection or both Query-LoRA projections.

## Inventory method

Do not deduplicate blocks by Python class alone. A single block class can have
dense and MoE instances with different children and therefore different region
sets. Identify a block variant by:

1. The model and block-container path.
2. The block class.
3. Its direct child-module names and classes.
4. Its exact `available_remat_save_regions(block)` result.

Build representative configurations on the `meta` device. Include optimized
module substitutions, such as DistGEMM, because an overridden `forward` can
change or invalidate inherited regions.

## Current model inventory

Status meanings:

- `complete`: the important attention and FFN/MoE compute boundaries expose
  save regions.
- `partial`: some useful regions exist, but important boundaries are missing.
- `unavailable`: no usable regions are currently exposed.
- `not wrapped`: regions may exist, but `RematAC.apply` does not checkpoint the
  relevant block container.

### Common MoE dispatch

`RoutedExperts` exposes high-level `dispatch` and `combine` save regions for
the exact `LocalTokenDispatcher` and `AllToAllTokenDispatcher` backends. The
AllToAll dispatch adapter retains the routed activation and the tensor metadata
needed by combine; its small host split lists are represented as CPU tensors so
the region has a tensor-only output contract. Stateful and asynchronous
dispatchers do not currently expose communication save regions.

| Dispatcher backend | `dispatch` / `combine` regions | Status |
| --- | --- | --- |
| Local | available | unit tested |
| Standard AllToAll | available | unit and four-GPU EP validation complete |
| TorchAO | unavailable | tabled: confirm production usage, then validate padded permutation metadata |
| DeepEP | unavailable | tabled: dispatch and combine share a handle lifecycle |
| HybridEP | unavailable | tabled: opaque dispatch-handle support needs a design decision |
| MinimalAsyncEP | unavailable | tabled: asynchronous symmetric-buffer state needs validation |

These regions intentionally retain communication results, which can be large.
They are optional policy choices rather than required regions. The router's
`routing_decision` remains the only required common MoE region.

SelectiveAC already marks both dispatch and combine as `MUST_SAVE` for DeepEP
and HybridEP. RematAC needs a deliberate way to prevent users from retaining
only one side of these handle-coupled operations. HybridEP also returns an
opaque `DispatchHandle`, while `torch_remat.region` currently accepts only
Tensor or `None` output leaves. Before integrating either backend, decide how
TorchTitan should represent inseparable policy choices and whether
`torch_remat` should support registered opaque output leaves. Validate both
HybridEP modes after those decisions.

### Common vision transformer

The shared `VisionTransformerBlock` stack exposes `attn.qkv`,
`attn.inner_attention`, `attn.proj`, `mlp.fc1`, and `mlp.fc2`. Q/K/V are one
region so all three projections follow one policy. Norms, GELU, and model-owned
RoPE application remain recomputable. Qwen 3.5, Muse Glimmer, Kimi K2.5, and
Kimi K3 explicitly apply activation checkpointing to their vision encoder's
`layers` container separately from the text decoder. Patch embedding, final
mergers/adapters, and other encoder-level preparation remain outside the
checkpointed block stack.

### DeepSeek V3

Runtime checked with `deepseek_v3_debugmodel`,
`deepseek_v3_debugmodel_q_lora`, and `deepseek_v3_debugmodel_mtp`.

| Block variant | Current available save regions | Status |
| --- | --- | --- |
| Dense `DeepSeekV3TransformerBlock` | `attention.wq` without Query LoRA, or `attention.wq_a`/`attention.wq_b` with Query LoRA; `attention.wkv_a`, `attention.wkv_b`, `attention.inner_attention`, `attention.wo`; dense FFN `w1`/`w3`/`w2` | complete |
| MoE `DeepSeekV3TransformerBlock` | Same attention variants; `moe.router`, required `moe.router.routing_decision`, routed-expert `w1`/`w3`/`w2`, shared-expert `w1`/`w3`/`w2` | complete |
| `MTPTransformerBlock` | Same attention and MoE regions as its underlying DeepSeek block | not wrapped |

`MTPTransformerBlock` instances live in `model.mtp_layers`, while `RematAC`
currently visits only `model.layers`.

### Llama 3

Runtime checked with `llama3_debugmodel` and
`llama3_debugmodel_dist_gemm`.

| Block variant | Current available save regions | Status |
| --- | --- | --- |
| Stock `Llama3TransformerBlock` | `attention.qkv`, `attention.inner_attention`, `attention.wo`, `feed_forward.w1`, `feed_forward.w3`, `feed_forward.w2` | complete |
| DistGEMM `Llama3TransformerBlock` | `attention.qkv`, `attention.inner_attention`, `attention.wo`, `feed_forward.w13`, `feed_forward.w2` | complete |

`DistGEMMFeedForward` exposes its fused gate/up projection as the atomic `w13`
region. Its all-gather and GEMMs stay inside that region. The `w2` region
similarly contains the output GEMM and reduce-scatter. The
`DistGEMMFusedSwiGLU` override exposes the same two regions.

The non-DistGEMM `FusedSwiGLU` override also exposes atomic `w13` and `w2`
regions, with its fused SiLU-and-multiply operation left outside those regions
for recomputation.

### Qwen 3

Runtime checked with `qwen3_debugmodel` and `qwen3_moe_debug`.

| Block variant | Current available save regions | Status |
| --- | --- | --- |
| Dense `Qwen3TransformerBlock` | `attention.qkv`, `attention.inner_attention`, `attention.wo`, `feed_forward.w1`, `feed_forward.w3`, `feed_forward.w2` | complete |
| MoE `Qwen3TransformerBlock` | `attention.qkv`, `attention.inner_attention`, `attention.wo`, `moe.router`, required `moe.router.routing_decision`, routed-expert `w1`/`w3`/`w2` | complete |

The stock attention path is shared with Llama 3 through `GQAttention`.

### GPT-OSS

Runtime checked with `gpt_oss_debugmodel_flex`. The default varlen debug model
requires `flash_attn` in the local environment.

| Block variant | Current available save regions | Status |
| --- | --- | --- |
| `GptOssTransformerBlock` | `moe.router`, required `moe.router.routing_decision` | partial |

The custom attention exposes no regions. `GptOssGroupedExperts` overrides
`GroupedExperts.forward`, so inherited expert regions are ignored. Further
GPT-OSS remat integration is intentionally tabled.

### Qwen 3.5

Source reviewed; local meta construction is blocked by the optional `fla`
dependency.

| Block variant | Current available save regions | Status |
| --- | --- | --- |
| Full-attention + dense FFN | `attn.qkv`, `attn.inner_attention`, `attn.wo`, `feed_forward.w1`, `feed_forward.w3`, `feed_forward.w2` | complete |
| DeltaNet + dense FFN | `feed_forward.w1`, `feed_forward.w3`, `feed_forward.w2` | partial |
| Full-attention + MoE | `attn.qkv`, `attn.inner_attention`, `attn.wo`, plus common MoE router/expert/shared-expert regions, including required `moe.router.routing_decision` | complete |
| DeltaNet + MoE | common MoE router/expert/shared-expert regions, including required `moe.router.routing_decision` | partial |

`Qwen35Attention.qkv` groups its three projection calls; the `wq` result also
contains the output gate. `GatedDeltaNet` still needs an explicit boundary
design. Qwen 3.5's sigmoid-gated shared experts expose `w1`, `w3`, `w2`, and
the additional `gate` projection. Its vision encoder is checkpointed
separately and uses the common vision-transformer regions.

### Kimi K2.5

Runtime checked with `muse_glimmer_debugmodel`. The multimodal configuration's
vision path still depends on optional vision packages.

The text decoder reuses the DeepSeek V3 transformer blocks, so its text-region
surface should match the DeepSeek dense and MoE variants. Its vision encoder is
checkpointed separately and uses the common vision-transformer regions.

### Kimi K3

Source reviewed; local meta construction is blocked by the optional `attn_gym`
dependency.

| Block variant | Current available save regions | Status |
| --- | --- | --- |
| MLA + dense `KimiK3TransformerBlock` | `feed_forward.w1`, `feed_forward.w3`, `feed_forward.w2` | partial |
| KDA + dense `KimiK3TransformerBlock` | `feed_forward.w1`, `feed_forward.w3`, `feed_forward.w2` | partial |
| MLA + MoE `KimiK3TransformerBlock` | routed-expert `w1`/`w3`/`w2`, required `moe.router.routing_decision` | partial |
| KDA + MoE `KimiK3TransformerBlock` | routed-expert `w1`/`w3`/`w2`, required `moe.router.routing_decision` | partial |

`KimiFeedForward` and `KimiGroupedExperts` now preserve the common `w1`, `w3`,
and `w2` region surface around their SiTU activation. `KimiLatentMoE`, Kimi MLA,
KDA, and residual projection paths still need explicit boundary audits. The
vision encoder's transformer blocks use the common vision-region surface. This
remaining Kimi K3 work is intentionally tabled.

### Muse Glimmer

Source reviewed; local meta construction is blocked by the optional
`torchvision` dependency.

| Block variant | Current available save regions | Status |
| --- | --- | --- |
| `MuseGlimmerTransformerBlock` | `attention.qkv`, `attention.inner_attention`, `attention.o_gate`, `attention.wo`, `feed_forward.w1`, `feed_forward.w3`, `feed_forward.w2` | complete |

Muse attention overrides `GQAttention.forward`, so it declares its own region
surface. All current Muse configurations build `o_gate`; its config type remains
optional. The optional vision encoder is checkpointed separately and uses the
common vision-transformer regions.

### Flux

Runtime checked with `flux_debugmodel`.

| Block variant | Current available save regions | Status |
| --- | --- | --- |
| `DoubleStreamBlock` | none | not wrapped |
| `SingleStreamBlock` | none | not wrapped |

Flux has `double_blocks` and `single_blocks` rather than `layers`, so the current
`RematAC.apply` contract cannot be used for it.

## Implementation order

### 1. Add exact inventory tests

- Add one small representative configuration for every architecture variant.
- Build on `meta` and snapshot the block container, child structure, and exact
  available-region list.
- Assert that every entry in `torchtitan.models._supported_models` is represented.
- Skip a model only when an optional dependency is unavailable, and report the
  missing dependency explicitly.

### 2. Audit shared GQA

- [x] Expose `qkv`, `inner_attention`, and `wo` regions.
- [x] Leave Q/K normalization and RoPE outside save regions.
- [x] Validate fused and non-fused QKV region discovery.
- [x] Validate forward/backward numerics and execution counts for each region.
- [x] Validate the DistGEMM attention path with TP enabled.

### 3. Audit specialized implementations

- Qwen 3.5 DeltaNet.
- Tabled: GPT-OSS attention and grouped experts.
- Tabled: Kimi K3 MLA, KDA, latent MoE, and block residual projections.

Each subclass that overrides a region-bearing `forward` must explicitly declare
and implement its own regions. Do not copy a base declaration without checking
that the call boundaries are semantically equivalent.

### 4. Generalize checkpoint-block discovery

Replace the unconditional `model.layers` assumption with an explicit model
contract before enabling:

- DeepSeek MTP `mtp_layers`.
- Flux `double_blocks` and `single_blocks`.
- Multimodal vision-encoder block stacks.

The contract should make checkpoint ownership visible on the model rather than
embedding more model-specific paths in `RematAC`.

### 5. Audit tensor persistence

For every `maybe_remat_recompute_needs` call:

- Identify the first consumer during replay.
- Keep the call only if that consumer is a bare operation outside another remat
  region and needs the selected-region output.
- Prefer placing persistence near the consumer when ownership is clear.
- Measure memory to ensure an unnecessary call is not retaining an activation.

Known follow-ups from the initial DeepSeek audit:

- Router `scores_TE` may not need persistence when its only consumer is
  metadata-only `zeros_like`.
- Move `attention.wo` persistence toward the transformer-block residual
  consumer if the ownership remains clear.
- Audit every consumer of `FeedForward.w2` before moving its persistence out of
  the shared module.

### 6. Resolve module-boundary redistribution replay

Status: tabled until source-to-destination redistributions move from the generic
module forward wrapper into model code.

This is a performance problem, not a correctness problem. In the standard
non-DistGEMM TP/SP path, the attention and FFN input all-gathers run in the
outer module wrapper before execution enters the fine-grained save regions.
Consequently, checkpoint replay repeats those collectives even when the
downstream projection is saved and skipped.

| Path | Collective placement today | Replay behavior |
| --- | --- | --- |
| Standard attention input -> `qkv` | Outside `qkv`, in the attention wrapper | Input all-gather is replayed |
| Standard FFN input -> `w1`/`w3` | Outside `w1`/`w3`, in the FFN wrapper | Input all-gather is replayed |
| Standard `wo`/`w2` output | Inside the selected module call | Reduce-scatter is skipped when the region is saved |
| DistGEMM `w13`/`w2` | Inside the fused operation | Collective is skipped when the region is saved |

Once redistributions are explicit in model code, place each input all-gather
inside the corresponding projection region. Add a distributed profiler or
collective-counting test that distinguishes original forward execution from
checkpoint replay; numerical equality alone does not prove that a collective
was skipped.

### 7. Validate each completed model policy

- Unit test exact region discovery and selection.
- Verify selected regions execute once while recomputed work executes twice.
- Compare loss and `grad_norm` bit-for-bit against the non-remat baseline with
  `--debug.seed=42 --debug.deterministic`.
- Measure peak memory for the complete policy and for individual regions.
- Exercise relevant combinations of FSDP reshard-after-forward, PP, TP/SP, CP,
  and EP.

## Completed validation

- Integration-test policies cover stock Llama 3, DistGEMM Llama 3, and MoE
  Qwen 3. Unit tests assert that each configured policy exactly matches the
  regions exposed by its model variant.
- Shared GQA unit tests cover fused and non-fused QKV projections. Each of
  `qkv`, `inner_attention`, and `wo` is selected independently to verify exact
  forward/backward numerics and that only the selected region skips replay.
- Llama 3 FSDP2+TP2+CP2 matched its non-remat baseline bit-for-bit for loss and
  `grad_norm` across 10 deterministic steps with the fake process-group
  backend.
- Qwen 3 MoE FSDP2+TP2+CP2+EP8 matched its non-remat baseline bit-for-bit for
  loss and `grad_norm` across 10 deterministic steps with the fake
  process-group backend.
- DistGEMM Llama 3 FSDP2+TP2 matched its non-remat baseline bit-for-bit for loss
  and `grad_norm` across 10 deterministic steps on four GPUs while retaining
  both the attention regions and the atomic `feed_forward.w13` and
  `feed_forward.w2` regions.
- Muse Glimmer FSDP8 matched its non-remat baseline bit-for-bit for loss and
  `grad_norm` across 10 deterministic steps with the fake process-group
  backend while retaining all four attention regions and the shared FFN
  regions.
- DeepSeek V3 Query-LoRA TP2+SP+CP2 matched its non-remat baseline bit-for-bit
  for loss and `grad_norm` across 10 deterministic steps on four GPUs while
  retaining `wq_a`, `wq_b`, `wkv_a`, `wkv_b`, `inner_attention`, and `wo`.
- DeepSeek V3 FSDP4+EP2 matched its non-remat baseline bit-for-bit for loss and
  `grad_norm` across 10 deterministic steps on four GPUs while retaining the
  standard AllToAll `dispatch` and `combine` regions.

## Progress log

- 2026-09-01: Added the generic `RematAC` integration and DeepSeek V3 region
  annotations.
- 2026-09-01: Moved the DeepSeek policy out of generic `RematAC.Config`; callers
  must now provide `save_regions` explicitly.
- 2026-09-01: Completed the first runtime/source inventory. Identified the
  structural-variant, MTP-container, optimized-subclass, vision-stack, and Flux
  gaps documented above.
- 2026-09-01: Added `qkv`, `inner_attention`, and `wo` save regions to shared
  `GQAttention`, covering stock Llama 3 and Qwen 3 attention implementations.
- 2026-09-01: Added Llama 3 and Qwen 3 integration-test policies for FSDP,
  TP/SP, CP, EP, and DistGEMM combinations. These are validation choices, not
  model defaults.
- 2026-09-01: Verified 10-step bit-for-bit loss and `grad_norm` equality for
  Llama 3, Qwen 3 MoE, and four-GPU DistGEMM baseline/remat comparisons.
- 2026-09-02: Added atomic `w13` and `w2` regions to DistGEMM FFNs, including
  the fused-SwiGLU override, and revalidated exact four-GPU TP2 numerics.
- 2026-09-02: Added the same atomic `w13` and `w2` region surface to the
  non-DistGEMM fused-SwiGLU override.
- 2026-09-02: Added `qkv`, `inner_attention`, `o_gate`, and `wo` save regions to
  Muse Glimmer attention.
- 2026-09-02: Verified 10-step bit-for-bit loss and `grad_norm` equality for the
  Muse Glimmer text-model FSDP8 baseline/remat comparison.
- 2026-09-02: Added grouped `qkv`, `inner_attention`, and `wo` save regions to
  Qwen 3.5 full attention. DeltaNet region design remains open.
- 2026-09-02: Restored the common `w1`, `w3`, and `w2` save-region surface on
  Kimi K3 dense FFNs and grouped experts while keeping SiTU recomputed.
- 2026-09-02: Added automatically retained required save regions and marked the
  common token-choice router's expert-selection decision as required.
- 2026-09-02: Added `attention.wkv_b` as a separate optional DeepSeek V3 MLA
  save boundary while keeping `kv_norm` outside the region.
- 2026-09-02: Split DeepSeek V3 query regions into `wq` for the direct path and
  `wq_a`/`wq_b` for Query LoRA, with instance-specific region discovery.
- 2026-09-02: Added a small Query-LoRA DeepSeek V3 configuration and validated
  exact 10-step TP2+SP+CP2 baseline/remat loss and `grad_norm` on four GPUs.
- 2026-09-02: Added the `gate` save region to the common
  `SigmoidGatedFeedForward` implementation.
- 2026-09-02: Added optional `dispatch` and `combine` regions for the exact
  Local and standard AllToAll MoE dispatchers. Stateful, padded, and
  asynchronous backends remain tabled pending backend-specific lifetime
  validation.
- 2026-09-02: Audited DeepEP and HybridEP, then tabled their RematAC
  communication regions. Their dispatch and combine operations share handle
  lifecycles and should not be independently configurable. HybridEP also
  returns an opaque `DispatchHandle` that cannot cross the current tensor-only
  `torch_remat.region` output boundary.
- 2026-09-02: Added `qkv`, `inner_attention`, and `proj` regions to shared
  vision attention and `fc1`/`fc2` regions to the shared vision MLP. Corrected
  the inventory to reflect that multimodal models apply activation
  checkpointing separately to their vision encoder block stacks.
