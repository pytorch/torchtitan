# `torch_remat` activation checkpointing

TorchTitan's full, selective, and configurable region activation-checkpointing
policies use `torch_remat` to checkpoint each transformer block. Operations
outside saved regions are recomputed during backward.

## Motivation

Model code identifies semantic compute and communication regions, while the
activation-checkpointing policy chooses which region outputs to retain.

This requires small, explicit annotations in model code. In return, the policy
surface is visible next to the operations it controls, and configurations refer
to stable model concepts such as attention projections instead of individual
ATen operators.

The policies differ only in which optional regions they retain:

- `FullAC` retains none and recomputes the full block except mandatory
  correctness regions.
- `SelectiveAC` retains every model-declared region and recomputes operations
  outside those regions.
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
        "attention.qkv",
        "attention.wo",
    ]
)
```

The same policy currently applies to every transformer block. Wildcards such
as `attention.*` are supported. Unmatched patterns are currently ignored;
validation must eventually account for regions across all pipeline stages.

Set `report_effective_policy=True` on any of the three policies to log the
unique regions exercised by the first model forward. The report classifies
each region as `SAVE`, `RECOMPUTE`, or `ALWAYS_SAVE`; the last category
identifies correctness regions whose retention is independent of the policy.
Reported configurable names are the logical policy keys accepted by
`RegionAC.save_regions`, so a grouped implementation may report one policy
region even when it contains several physical regions.

The main attention region families are:

| Transformer block | Input projections | Inner compute | Output projection |
| --- | --- | --- | --- |
| Common, DeepSeek V3, Muse Glimmer | `attention.qkv` | `attention.inner_attention` | `attention.wo` |
| Qwen3.5 full attention | `attn.qkv` | `attn.inner_attention` | `attn.wo` |
| Qwen3.5 DeltaNet | `attn.input_projections` | `attn.inner_compute` | `attn.output_projection` |
| Kimi K3 MLA | `attention.qkv` | `attention.inner_attention` | `attention.wo` |
| Kimi K3 KDA | `delta_attention.input_projections` | `delta_attention.inner_compute` | `delta_attention.output_projection` |

Qwen3.6 and Qwen3.8 reuse the Qwen3.5 implementations. Kimi K2.7 reuses
DeepSeek V3 attention. Kimi K3 latent MoE additionally exposes
`moe.routed_down` and `moe.routed_up`.

## Adding regions to model code

Model code defines a region at the operation being controlled:

```python
q, k, v = remat.region(
    self.qkv_linear,
    self.remat_region_name("qkv"),
    recompute=self.remat_should_recompute("qkv"),
)(x)
```

The activation-checkpointing policy configures each module with its name
relative to the transformer block and its save patterns. The helpers above
therefore resolve `qkv` to a qualified name such as `attention.qkv` and select
whether it is saved or recomputed. Without an enclosing `remat.checkpoint`,
`remat.region` does not change execution.

## Declaring recomputation dependencies

During the original forward, `torch_remat` determines whether an output from a
saved region will be needed during recomputation. It can infer this dependency
when the output is consumed by an explicit
`remat.region(..., recompute=True)`.

If the consumer is not inside such a region, call
`remat.recompute_needs_tensor(...)` immediately before the output is consumed:

```python
gate_up = remat.region(
    self.w13,
    self.remat_region_name("w13"),
    recompute=self.remat_should_recompute("w13"),
)(x)
gate, up = gate_up.unflatten(-1, (-1, 2)).unbind(-1)
remat.recompute_needs_tensor(gate, up)
hidden = F.silu(gate) * up
```

Without this marker, a tensor required by ordinary recomputed operations may
not be retained. Place the marker on the consumer side, immediately before the
bare operation that reads the tensor, rather than immediately after the region
that produced it. This ensures the output is retained only when that consumer
actually runs. Views may be passed because `torch_remat` resolves them to their
producing region by storage.

When one bare operation consumes multiple region outputs, pass all of them to
one call, as in the example above. Keep separate calls for separate consumers.
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

All `torch_remat` policies require `preserve_rng_state=False`. Random state
that can advance inside a saved region must instead be managed with an explicit
`torch_remat.RecomputeStateHook`.

## Forward side effects

State accumulated for logging or optimizer-step updates must advance only on
the original forward. The MoE `routing_decision` region therefore owns expert
selection, token-count accumulation, and quantile-histogram observation. It is
always retained and stores only the selected expert IDs, not the full router
scores or routing map. Auxiliary-loss accumulation uses its own always-retained
region. Kimi K2.7 QK-clipping statistics explicitly ignore checkpoint replay.

The currently supported transformer blocks do not advance RNG state
inside their forwards, so they do not require a `RecomputeStateHook`. Any future
dropout, stochastic rounding counter, or other external RNG state must add a
hook before it can be used safely with these policies.

## Saving expensive MoE work

Avoiding replay of expensive MoE work requires retaining both its compute and
communication regions:

- Routed-expert `w13` and `w2` grouped projections.
- Token-dispatcher `ep_communication`, which controls the token-count exchange,
  dispatch, and combine collectives together.
- Shared-expert projection regions. The shared `w2` region includes its
  `Partial -> Shard(0)` reduce-scatter when sequence parallelism is enabled.
- `tp_output_reduction`, which controls the final TP all-reduce when sequence
  parallelism is disabled.

For a common MoE module named `moe`, the corresponding policy is:

```python
RegionAC.Config(
    save_regions=[
        "moe.routed_experts.w13",
        "moe.routed_experts.w2",
        "moe.routed_experts.token_dispatcher.ep_communication",
        "moe.shared_experts.*",
        "moe.tp_output_reduction",
    ]
)
```

Operations outside these regions, including local permutation, token-shard
zero-fill, and branch addition, are recomputed. Routing decisions are retained
separately to keep expert selection identical during replay.
