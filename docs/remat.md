# `torch_remat` activation checkpointing

TorchTitan's selective and configurable region activation-checkpointing policies
use `torch_remat` to checkpoint each transformer block. Operations outside saved
regions are recomputed during backward. `FullAC` continues to use PyTorch's
native selective checkpointing so registered ordered effects are saved rather
than replayed.

## Motivation

Model code identifies semantic compute and communication regions, while the
activation-checkpointing policy chooses which region outputs to retain.

This requires small, explicit annotations in model code. In return, the policy
surface is visible next to the operations it controls, and configurations refer
to stable model concepts such as attention projections instead of individual
ATen operators.

The policies are:

- `FullAC` recomputes pure block operations while preserving registered effects.
- `SelectiveAC` retains every model-declared region except the routed-expert
  grouped projections (`*routed_experts.w13`, `*routed_experts.w2`), and
  recomputes operations outside those regions. Routed-expert grouped matmul
  activations scale with top-k and dominate MoE activation memory; the former
  operator-level policy also recomputed them. EP token-dispatcher
  communication regions stay retained, as the former policy saved all-to-all
  outputs. A model that declares no regions gets full recomputation
  under `SelectiveAC`.
- `RegionAC` uses an explicit `save_regions` pattern list.

The former operator-level SelectiveAC policy and its
`force_recompute_mm_shapes_by_fqns` option have been removed. Use `RegionAC`
when a policy needs finer control than saving all declared regions.

## Configuring saved regions

`RegionAC.Config.save_regions` contains shell-style patterns relative to a
transformer block. For example:

```python
RegionAC.Config(
    save_regions=[
        "attention.qkv_linear.wqkv.linear",
        "attention.wo.linear",
    ]
)
```

The same policy currently applies to every transformer block. Wildcards such
as `attention.*` are supported. Unmatched patterns are currently ignored;
validation must eventually account for regions across all pipeline stages.

## Diagnosing the effective policy

After applying an activation-checkpointing policy, use `torch_remat`'s trace
collector around the forward that you want to inspect:

```python
import torch_remat as remat

with remat.collect_trace() as trace:
    output = model(inputs, **model_kwargs)

print(trace.format())
```

For example, a trace may look like:

```text
torch_remat trace
attention.qkv_linear.wqkv.tp_gather: recompute
attention.qkv_linear.wqkv.linear: save
attention.inner_attention: recompute
attention.wo.linear: save
attention.wo.tp_reduce: save
feed_forward.w13.tp_gather: recompute
feed_forward.w13.linear: recompute
feed_forward.w2.linear: save
feed_forward.w2.tp_reduce: save
```

The trace lists the regions actually exercised, in execution order, and
whether each region was saved or recomputed. A full-model forward may contain
repeated region names from different transformer blocks. This diagnostic is
explicitly controlled by the caller, so it can be scoped to the model input,
batch, or block under investigation without changing the training config.

## Adding regions to model code

Model code defines a region at the operation being controlled:

```python
out = remat.region(
    self.inner_attention,
    self.remat_region_name("inner_attention"),
    recompute=self.remat_should_recompute("inner_attention"),
)(q, k, v)
```

The activation-checkpointing policy configures each module with its name
relative to the transformer block and its save patterns. The helpers above
therefore resolve `inner_attention` to a qualified name such as
`attention.inner_attention` and select whether it is saved or recomputed. Without an enclosing
`remat.checkpoint`, `remat.region` does not change execution.

Every `Linear` declares its own regions, so model code calls it directly:

- `<fqn>.linear` is the local projection. It covers every `Linear` subclass,
  including quantized and LoRA linears, which override only the local compute.
- `ColumnParallelLinear` adds `<fqn>.tp_gather` before the projection: an
  input all-gather under sequence parallelism, and otherwise a forward no-op
  whose backward all-reduces. Saving the projection while recomputing the
  gather keeps only the sequence shard: backward replays the all-gather for
  the weight gradient instead of retaining the gathered input.
- `RowParallelLinear` adds `<fqn>.tp_reduce` after the projection. It follows
  the `linear` save policy, since saving only one of the two either retains
  the TP-times larger partial output or saves nothing.

For example, the fused attention projection is
`attention.qkv_linear.wqkv.linear`. Do not wrap a `Linear` call in another
region: a saved outer region cannot contain a recomputed inner region.

A `Linear` also pins the tensor it returns with
`remat.recompute_needs_tensor` (for `RowParallelLinear`, the reduced output),
since bare operations such as activations, norms and residual adds read it.
Model code therefore never pins a `Linear` output.

When several plain `Linear` projections share one TP input, the module gathers
it once at their common boundary with `maybe_gather_tp_input(self, x)`, which
declares `<module fqn>.tp_gather` with the same semantics as the
`ColumnParallelLinear` gather. For example, `attention.tp_gather` in DeepSeek V3
and Kimi K3 MLA, `attn.tp_gather` in the Qwen3.5-family attention and DeltaNet,
and `delta_attention.tp_gather` in Kimi K3 KDA. Every attention module wraps its
kernel in `<module fqn>.inner_attention`.

## Declaring recomputation dependencies

During the original forward, `torch_remat` determines whether an output from a
saved region will be needed during recomputation. It can infer this dependency
when the output is consumed by an explicit
`remat.region(..., recompute=True)`.

If the consumer is not inside such a region, call
`remat.recompute_needs_tensor(...)` immediately before the output is consumed:

```python
out = remat.region(
    self.inner_attention,
    self.remat_region_name("inner_attention"),
    recompute=self.remat_should_recompute("inner_attention"),
)(q, k, v)
remat.recompute_needs_tensor(out)
out = out.contiguous() * torch.sigmoid(gate)
```

Without this marker, a tensor required by ordinary recomputed operations may
not be retained. Place the marker on the consumer side, immediately before the
bare operation that reads the tensor, rather than immediately after the region
that produced it. This ensures the output is retained only when that consumer
actually runs. Views may be passed because `torch_remat` resolves them to their
producing region by storage.

When one bare operation consumes multiple region outputs, pass all of them to
one call. Keep separate calls for separate consumers.
Do not add a marker when the output is consumed only by another `remat.region`;
that dependency is inferred automatically.

The marker can be omitted when a region's output is returned directly from the
checkpointed transformer block and no operation inside the block reads its
data. Being the last region in a submodule is not sufficient: for example, an
attention output may still be consumed by a residual addition in the enclosing
transformer block. If the consumer lives outside the helper that owns the
region, place the marker as close to that call-site consumer as the module
boundary permits.

## Random state

`SelectiveAC` and `RegionAC` require `preserve_rng_state=False`. Random state
that can advance inside a saved region must instead be managed with an explicit
`torch_remat.RecomputeStateHook`. `FullAC` retains PyTorch's native RNG-state
handling.

## Forward side effects

State accumulated for logging or optimizer-step updates must advance only on
the original forward. The always-retained MoE `routing_decision` region owns
expert selection and quantile-histogram observation, while token-count
accumulation explicitly ignores checkpoint replay. Both reuse the routing map
built for dispatch and auxiliary loss. Kimi K2.7 QK-clipping statistics also
ignore replay. Auxiliary-loss accumulation uses an always-retained region.

The currently supported transformer blocks do not advance RNG state
inside their forwards, so they do not require a `RecomputeStateHook`. Any future
dropout, stochastic rounding counter, or other external RNG state must add a
hook before it can be used safely with these policies.

## Saving expensive MoE work

Avoiding replay of expensive MoE work requires retaining both its compute and
communication regions:

- Routed-expert `w13` and `w2` grouped projections.
- Token-dispatcher `dispatch` and `combine`. With the all-to-all dispatcher
  (EP > 1), each is one region: `dispatch` covers expert sorting, the
  token-count exchange and its device-to-host sync, the dispatch all-to-all
  and the expert-major permute; `combine` covers the unpermute, the combine
  all-to-all and the score-weighted scatter-add. The DeepEP and HybridEP
  dispatchers instead declare `ep_communication.dispatch` and
  `ep_communication.combine` around their kernels.
- Shared-expert linear regions. The shared `w2.tp_reduce` region is the
  `Partial -> Shard(0)` reduce-scatter when sequence parallelism is enabled,
  and follows the `w2.linear` policy.
- `tp_output_reduction`, which controls the final TP all-reduce when sequence
  parallelism is disabled.

For a common MoE module named `moe`, the corresponding policy is:

```python
RegionAC.Config(
    save_regions=[
        "moe.routed_experts.w13",
        "moe.routed_experts.w2",
        "moe.routed_experts.token_dispatcher.dispatch",
        "moe.routed_experts.token_dispatcher.combine",
        "moe.shared_experts.*",
        "moe.tp_output_reduction",
    ]
)
```

At EP=1 the dispatcher declares finer regions: `sort` and `gather` (local
expert ordering) and `combine` (the score-weighted scatter-add back to token
order), plus `permute` and `unpermute` for the padded TorchAO dispatcher. Save
them individually when their replay cost matters more than their memory. Operations outside all
regions, such as token-shard zero-fill and branch addition, are recomputed. Routing decisions are retained
separately to keep expert selection identical during replay.
