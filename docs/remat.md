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
    ],
    log_available_regions=True,
)
```

The same policy currently applies to every transformer block. Wildcards such
as `attention.*` are supported. Unmatched patterns are currently ignored;
validation must eventually account for regions across all pipeline stages.

With `log_available_regions=True`, TorchTitan groups transformer blocks with
the same type and region catalog, then logs their layer IDs and available
save-region names. Regions can also be inspected programmatically before
applying RegionAC:

```python
model.layers["0"].available_remat_save_regions()
# ["attention.qkv", "attention.inner_attention", "attention.wo"]
```

## Adding regions to model code

Model code registers a region handle as a class attribute on the class that
implements `forward`, then uses that same handle at the operation being
controlled:

```python
class Attention(Module):
    qkv_region = Module.register_remat_region("qkv")

    def forward(self, x):
        q, k, v = remat.region(
            self.qkv_linear,
            self.remat_region_name(self.qkv_region),
            recompute=self.remat_should_recompute(self.qkv_region),
        )(x)
```

The handle is the single source of truth for the local name. Assigning it as a
class attribute performs the registration; creating one inside `__init__` or
`forward` is invalid. RegionAC discovers available names from these handles,
and `forward` uses the same objects when it calls `remat.region`. A subclass
that inherits `forward` also inherits its regions. A subclass that replaces
`forward` must declare the regions used by its replacement implementation.

`RegionAC` configures each module with its name relative to the transformer
block and the user's save patterns. The helpers above therefore resolve `qkv`
to a qualified name such as `attention.qkv` and select whether it is saved or
recomputed. Without an enclosing `remat.checkpoint`, `remat.region` does not
change execution.

A complete call site with an explicit recomputation dependency looks like:

```python
q, k, v = remat.region(
    self.qkv_linear,
    self.remat_region_name(self.qkv_region),
    recompute=self.remat_should_recompute(self.qkv_region),
)(x)
remat.recompute_needs_tensor(q, k, v)
q, k = self.rope(q, k, positions)
```

## Declaring recomputation dependencies

During the original forward, `torch_remat` determines whether an output from a
saved region will be needed during recomputation. It can infer this dependency
when the output is consumed by an explicit
`remat.region(..., recompute=True)`.

If the consumer is not inside such a region, call
`remat.recompute_needs_tensor(...)` immediately before the output is consumed,
as shown above.

Without this marker, a tensor required by ordinary recomputed operations may
not be retained. For the initial TorchTitan integration, model code
conservatively calls `recompute_needs_tensor` immediately after each
configurable region because region outputs are usually consumed by bare
operations soon afterward. This prioritizes correct replay, but a saved region
may retain an output that recomputation does not actually need.

The marker can be omitted when a region's output is returned directly from the
checkpointed transformer block and no operation inside the block reads its
data. Being the last region in a submodule is not sufficient: for example, an
attention output may still be consumed by a residual addition in the enclosing
transformer block. We plan to audit these call sites and place markers at the
actual consumers in a later change.

## Random state

`RegionAC` requires `preserve_rng_state=False`. Random state that can advance
inside a saved region must instead be managed with an explicit
`torch_remat.RecomputeStateHook`.
