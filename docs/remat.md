# `torch_remat` activation checkpointing

`RegionAC` uses `torch_remat` to checkpoint each transformer block while
allowing selected regions inside the block to retain their outputs. Operations
outside saved regions are recomputed during backward.

## Motivation

TorchTitan currently provides full and selective activation checkpointing.
Selective activation checkpointing makes save decisions at the operator level.
`torch_remat` provides a model-aware alternative: model code identifies
semantic regions, while training configuration chooses which region outputs to
retain.

This requires small, explicit annotations in model code. In return, the policy
surface is visible next to the operations it controls, and configurations refer
to stable model concepts such as attention projections instead of individual
ATen operators.

`RegionAC` is the initial integration name. The long-term plan is to migrate
the existing `FullAC` and `SelectiveAC` implementations to `torch_remat` and
converge on one activation-checkpointing implementation. The `RegionAC` name is
therefore provisional and may change as that migration progresses.

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
attention.qkv: save
attention.inner_attention: recompute
attention.wo: save
feed_forward.w13: recompute
feed_forward.w2: save
```

The trace lists the regions actually exercised, in execution order, and
whether each region was saved or recomputed. A full-model forward may contain
repeated region names from different transformer blocks. This diagnostic is
explicitly controlled by the caller, so it can be scoped to the model input,
batch, or block under investigation without changing the training config.

## Regions by model

Attention region names use the same meanings across models: `qkv` covers the
query, key, and value input projections; `latent_projections` covers the MLA
down-projections; `gate` covers a separate output-gate projection;
`inner_attention` covers the attention kernel; and `wo` covers the output
projection. `input_redistribution` is the TP gather shared by multiple input
branches and is exercised only when a TP group exists. `routed_down` and
`routed_up` cover the Kimi K3 latent-MoE projections around its routed experts.

Each row below lists the complete region set for that model component. Names
are relative to a transformer block and can be used directly in
`RegionAC.save_regions`.

| Model component | Regions |
| --- | --- |
| Llama 3 attention | `attention.qkv`, `attention.inner_attention`, `attention.wo` |
| Qwen 3 attention | `attention.qkv`, `attention.inner_attention`, `attention.wo` |
| DeepSeek V3 attention | `attention.input_redistribution`, `attention.latent_projections`, `attention.inner_attention`, `attention.wo` |
| Kimi K2.7 attention | `attention.input_redistribution`, `attention.latent_projections`, `attention.inner_attention`, `attention.wo` |
| Muse Glimmer attention | `attention.input_redistribution`, `attention.qkv`, `attention.gate` (when configured), `attention.inner_attention`, `attention.wo` |
| Qwen3.5 full attention | `attn.input_redistribution`, `attn.qkv`, `attn.inner_attention`, `attn.wo` |
| Qwen3.5 DeltaNet | `attn.input_redistribution`, `attn.qkv`, `attn.gate`, `attn.inner_attention`, `attn.wo` |
| Qwen3.6 full attention | `attn.input_redistribution`, `attn.qkv`, `attn.inner_attention`, `attn.wo` |
| Qwen3.6 DeltaNet | `attn.input_redistribution`, `attn.qkv`, `attn.gate`, `attn.inner_attention`, `attn.wo` |
| Qwen3.8 full attention | `attn.input_redistribution`, `attn.qkv`, `attn.inner_attention`, `attn.wo` |
| Qwen3.8 DeltaNet | `attn.input_redistribution`, `attn.qkv`, `attn.gate`, `attn.inner_attention`, `attn.wo` |
| Kimi K3 MLA | `attention.input_redistribution`, `attention.latent_projections`, `attention.gate`, `attention.inner_attention`, `attention.wo` |
| Kimi K3 KDA | `delta_attention.input_redistribution`, `delta_attention.qkv`, `delta_attention.gate`, `delta_attention.inner_attention`, `delta_attention.wo` |
| Kimi K3 latent MoE | `moe.routed_down`, `moe.routed_up` |

The DeepSeek V3 and Kimi K3 MLA QKV up-projections are intentionally outside a
region and are therefore recomputed. The Qwen3.5-family full-attention output
gate is fused into the query projection and is covered by `attn.qkv`.

## Adding regions to model code

Model code defines a region at the operation being controlled:

```python
q, k, v = remat.region(
    self.qkv_linear,
    self.remat_region_name("qkv"),
    recompute=self.remat_should_recompute("qkv"),
)(x)
```

`RegionAC` configures each module with its name relative to the transformer
block and the user's save patterns. The helpers above therefore resolve `qkv`
to a qualified name such as `attention.qkv` and select whether it is saved or
recomputed. Without an enclosing `remat.checkpoint`, `remat.region` does not
change execution.

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
remat.recompute_needs_tensor(gate_up)
gate, up = gate_up.unflatten(-1, (-1, 2)).unbind(-1)
hidden = F.silu(gate) * up
```

Without this marker, a tensor required by ordinary recomputed operations may
not be retained. Place the marker on the consumer side, immediately before the
first bare operation that reads the region output. If that operation is a view,
split, or unbind that will itself be recomputed, mark the region output before
the operation rather than marking its derived tensors.

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

`RegionAC` requires `preserve_rng_state=False`. Random state that can advance
inside a saved region must instead be managed with an explicit
`torch_remat.RecomputeStateHook`.

## Forward side effects

State accumulated for logging or optimizer-step updates must advance only on
the original forward. The always-retained MoE `routing_decision` region owns
expert selection and quantile-histogram observation, while token-count
accumulation explicitly ignores checkpoint replay. Both reuse the routing map
built for dispatch and auxiliary loss. Kimi K2.7 QK-clipping statistics also
ignore replay. Auxiliary-loss accumulation uses an always-retained region.

The currently supported RegionAC transformer blocks do not advance RNG state
inside their forwards, so they do not require a `RecomputeStateHook`. Any future
dropout, stochastic rounding counter, or other external RNG state must add a
hook before it can be used safely with RegionAC.

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
