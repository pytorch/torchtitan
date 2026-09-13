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

`RegionAC` requires `preserve_rng_state=False`. Random state that can advance
inside a saved region must instead be managed with an explicit
`torch_remat.RecomputeStateHook`.
