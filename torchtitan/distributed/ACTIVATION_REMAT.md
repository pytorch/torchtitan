# `torch_remat` activation rematerialization

Activation rematerialization (remat), also known as activation checkpointing,
trades compute for memory: a transformer block keeps only some activations from
its forward and recomputes the rest during backward. TorchTitan's remat
policies use `torch_remat` to checkpoint each transformer block. Operations
outside saved regions are recomputed during backward.

## Motivation

Model code identifies semantic compute and communication regions, while the
remat policy chooses which region outputs to retain.

This requires small, explicit annotations in model code. In return, the policy
surface is visible next to the operations it controls, and configurations refer
to stable model concepts such as attention projections instead of individual
ATen operators.

The policies are configs, set as the trainer's `activation_remat` field, and
applied to every transformer block with `config.apply(model)`:

- `ActivationRematConfig` saves the regions matching `save_regions`, except
  those matching `recompute_regions`, and recomputes everything else.
- `FullActivationRematConfig` is a fixed policy with `save_regions=[]`. It
  recomputes every block operation except the regions model code always
  retains with `recompute=False`, such as routing decisions and trailing adds.
- `DefaultActivationRematConfig` is the trainer's default policy:
  `save_regions=["*"]` and `recompute_regions=["*routed_experts.w13.*"]`.
  Its regions may change as measurements improve; use `ActivationRematConfig`
  with explicit regions for a fixed policy. It retains every
  model-declared region except the routed-expert `w13` grouped projection, and
  recomputes operations outside those regions. Routed-expert grouped matmul
  activations scale with top-k and dominate MoE activation memory. `w2` stays
  saved: its saved input is the activation output, which replay rebuilds
  anyway, so recomputing it would cost time without freeing memory. EP
  token-dispatcher communication regions stay retained, as the former policy
  saved all-to-all outputs. A model that declares no regions gets full
  recomputation under this policy.

The former operator-level SelectiveAC policy and its
`force_recompute_mm_shapes_by_fqns` option have been removed. Use
`ActivationRematConfig` when a policy needs finer control than the fixed
policies.

## Configuring saved regions

`ActivationRematConfig.save_regions` contains shell-style patterns relative to a
transformer block. For example:

```python
ActivationRematConfig(
    save_regions=[
        "attention.qkv_linear.wqkv.linear",
        "attention.wo.linear",
    ]
)
```

`ActivationRematConfig.recompute_regions` uses the same pattern syntax and wins
over `save_regions`. Starting from no AC, save everything and list the few
regions to recompute until the model fits its memory budget:

```python
ActivationRematConfig(
    save_regions=["*"],
    recompute_regions=["moe.routed_experts.w13.grouped_mm"],
)
```

Wildcards such as `attention.*` are supported. Unmatched patterns are
currently ignored; validation must eventually account for regions across all
pipeline stages.

### Per-block policies

The policy applies to every transformer block unless `block_overrides`
changes it for selected blocks. A uniform policy pays the same price in every
block: if recomputing one expensive region in two blocks is enough to fit,
the uniform choice is to recompute it in all of them. An override names
blocks and replaces any of `save_regions`, `recompute_regions`,
`offload_block_input` and `offload_regions` for them; a field it leaves `None`
keeps the base value:

```python
ActivationRematConfig(
    save_regions=["*"],
    block_overrides=[
        # Recompute the attention core only in the first two blocks.
        BlockRematOverride(
            blocks=["layers.0", "layers.1"],
            recompute_regions=["attention.inner_attention"],
        ),
    ],
)
```

Block names are exact FQNs in the original model, `<container fqn>.<id>`:
`layers.<id>` for decoder blocks, `mtp_layers.<id>` for multi-token
prediction blocks and `<encoder>.layers.<id>` (e.g. `vision_encoder.layers.3`)
for encoder blocks. Each model lists its block containers in
`BaseModel._remat_block_container_fqns`, and `parallelize` passes them to
`apply(model, block_container_fqns=...)`. Pipeline splitting keeps original
layer ids, so an override follows its block to whichever stage holds it.
Empty overrides and a block named by two overrides raise at config time; a
name the model lacks is not detected, since each pipeline stage holds only
some blocks. Within a block's result, `recompute_regions` still wins over
`save_regions`, so an override that sets only `save_regions` cannot retain a
region the base recomputes; override `recompute_regions` too. The fixed
presets accept overrides that set only the offload fields; use
`ActivationRematConfig` to change what they retain.

Patterns match each region name independently. Saving a parent region does not
save its child regions. Use a wildcard to save both when their names share a
prefix:

```python
ActivationRematConfig(save_regions=["attention.inner_attention*"])
```

For example, selective-gather attention has a nested `assemble_kv` region.
The wildcard saves both regions.
Check the remat trace for the actual names. Region names may differ from module
names.

A saved parent cannot contain a recomputed child. `torch_remat` rejects this
during forward. Do not force a child into `recompute_regions` while saving its
parent. Recompute both, or save both.

## Saved-tensor hooks

A checkpointed block keeps a few tensors alive from its forward to its
backward: its inputs, which recompute replays from, and the tensors its save
regions keep. torch_remat's saved-tensor hooks see exactly those: `pack`
replaces each with a payload and `unpack` rebuilds it when recompute or
backward needs it. Inside `pack`, `torch_remat.current_saved_tensor_info().kind`
says why a tensor is retained: `CHECKPOINT_INPUT` for a block input,
`SAVE_OUTPUT` and `BACKWARD` for tensors kept by save regions. `pack` runs at
block entry, before the forward reads its inputs, so it must not free or
overwrite their storage; unpacks need not run in reverse pack order; and
torch_remat rejects the hooks when the checkpoint runs under `torch.compile`.
See torch_remat's
[offloading guide](https://github.com/meta-pytorch/remat/blob/main/docs/offloading.md).

TorchTitan uses them for CPU offloading, driven by the remat policy:
`offload_block_input` and `offload_regions` choose which retained tensors go
to CPU, and `apply` then installs the offloader, which runs the copies
overlapped with compute. See
[ACTIVATION_OFFLOAD.md](ACTIVATION_OFFLOAD.md).

## Diagnosing the effective policy

After applying a remat policy, use `torch_remat`'s trace
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

The remat policy configures each module with its name
relative to the transformer block and its save patterns. The helpers above
therefore resolve `inner_attention` to a qualified name such as
`attention.inner_attention` and select whether it is saved or recomputed. Without an enclosing
`remat.checkpoint`, `remat.region` does not change execution.

Every `Linear` declares its own regions, so model code calls it directly:

- `<fqn>.linear` is the local projection. It covers every `Linear` subclass,
  including quantized and LoRA linears, which override only the local compute.
  LoRA adapters run inside the base projection's region rather than declaring
  their own, since a region nested in a saved region cannot be recomputed.
- `GroupedLinear` declares `<fqn>.grouped_mm` around its grouped matmul the
  same way, e.g. `moe.routed_experts.w13.grouped_mm`.
- `ColumnParallelLinear` adds `<fqn>.tp_gather` before the projection: an
  input all-gather under sequence parallelism, and otherwise a forward no-op
  whose backward all-reduces. Saving the projection while recomputing the
  gather keeps only the sequence shard: backward replays the all-gather for
  the weight gradient instead of retaining the gathered input.
- `RowParallelLinear` adds `<fqn>.tp_reduce` after the projection, controlled
  separately. Saving `linear` while recomputing `tp_reduce` keeps the TP-times
  larger partial output for the replayed reduction; saving both avoids it.

For example, the fused attention projection is
`attention.qkv_linear.wqkv.linear`. Do not wrap a `Linear` call in another
region: a saved outer region cannot contain a recomputed inner region.

When several plain `Linear` projections share one TP input, the module gathers
it once at their common boundary with `maybe_gather_tp_input(self, x)`, which
declares `<module fqn>.tp_gather` with the same semantics as the
`ColumnParallelLinear` gather. Every attention module wraps its kernel in
`<module fqn>.inner_attention`.

## Selective-gather attention

When selective-gather attention recomputes assembly, it declares two regions:

- `attention.inner_attention.remote_kv` always keeps the received halo.
  Backward does not repeat the exchange, even under Full remat.
- `attention.inner_attention.assemble_kv` follows the remat policy.
  Full remat rebuilds compact K/V from local K/V and the retained halo.

When assembly is saved, fused exchange declares only `assemble_kv` and keeps
the assembled K/V. Neither receive nor fused exchange is recomputed.

Use `FullActivationRematConfig()` to recompute attention and assembly while
keeping only halos. To retain attention and assembled K/V, use
`ActivationRematConfig(save_regions=["attention.inner_attention*"])`.

Direct callers must configure the policy on a containing `Module` and wrap
the forward in `remat.checkpoint`. Without block remat configuration, exchange
uses the fused path.

## Declaring recomputation dependencies

During the original forward, `torch_remat` determines whether an output from a
saved region will be needed during recomputation. It can infer this dependency
when the output is consumed by an explicit
`remat.region(..., recompute=True)`.

If the consumer is not inside such a region, call
`remat.recompute_needs_tensor(...)` immediately before the output is consumed:

```python
gate_up = self.w13(x)  # declares feed_forward.w13.linear
gate, up = gate_up.unbind(-2)
remat.recompute_needs_tensor(gate, up)
hidden = F.silu(gate) * up
```

Without this marker, a tensor required by ordinary recomputed operations may
not be retained. Place the marker on the consumer side, immediately before the
bare operation that reads the tensor, rather than immediately after the region
that produced it. This ensures the output is retained only when that consumer
actually runs. Views may be passed because `torch_remat` resolves them to their
producing region by storage. View operations themselves (`view`, `unbind`,
`split`, `transpose`, a `contiguous` or `reshape` that does not copy) need no
marker, even on a saved region's output: replaying a view is metadata-only.

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

### Trailing adds

`torch.utils.checkpoint` stops replay once every tensor saved under the
checkpoint has been recomputed, so it never reruns trailing operations that
save nothing, such as the residual add at the end of a block. `remat.checkpoint`
replays the whole block, so a bare trailing add would need a marker that keeps
its inputs (e.g. the MoE combine output) alive from the forward until replay.
Instead, the end-of-block residual add (`ffn_residual`) and the shared-expert
add (`moe.shared_add`) are fixed `recompute=False` regions: they save nothing
for backward, so a saved add keeps nothing resident, its inputs need no
marker, and replay skips it. Only the adds are wrapped; a trailing norm, such
as Muse Glimmer's post-norm, stays a bare operation with a marker on its input.

## Random state

All policies require `preserve_rng_state=False`. Random state that can advance
inside a saved region must instead be managed with an explicit
`torch_remat.RecomputeStateHook`.

## Determinism check

`determinism_check` selects how recomputed tensors are checked against the
original forward; see the
[PyTorch checkpoint documentation](https://docs.pytorch.org/docs/stable/checkpoint.html).
The policies do not support `debug=True`; use the trace collector described in
[Diagnosing the effective policy](#diagnosing-the-effective-policy) to inspect
which regions are saved or recomputed.

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

- Routed-expert `w13` and `w2` grouped projections (`w13.grouped_mm`,
  `w2.grouped_mm`).
- Token-dispatcher `dispatch` and `combine`. With the all-to-all dispatcher
  (EP > 1), each is one region: `dispatch` covers expert sorting, the
  token-count exchange and its device-to-host sync, the dispatch all-to-all
  and the expert-major permute; `combine` covers the unpermute, the combine
  all-to-all and the score-weighted scatter-add. The DeepEP and HybridEP
  dispatchers instead declare `ep_communication.dispatch` and
  `ep_communication.combine` around their kernels. Both save or replay
  dispatch and combine together, following `ep_communication`. A replay must
  receive tokens in the forward's order, so DeepEP uses a deterministic buffer, which
  adds a sort to every dispatch. HybridEP derives token positions from prefix
  scans. `FullActivationRematConfig` replays both backends' communication.
- Shared-expert linear regions. The shared `w2.tp_reduce` region is the
  `Partial -> Shard(0)` reduce-scatter when sequence parallelism is enabled;
  save it together with `w2.linear`.
- `tp_output_reduction`, which controls the final TP all-reduce when sequence
  parallelism is disabled. In batch-invariant mode with shared experts, the routed
  output has its own TP all-reduce, `tp_routed_reduction`.

For a common MoE module named `moe`, the corresponding policy is:

```python
ActivationRematConfig(
    save_regions=[
        "moe.routed_experts.w13.grouped_mm",
        "moe.routed_experts.w2.grouped_mm",
        "moe.routed_experts.token_dispatcher.dispatch",
        "moe.routed_experts.token_dispatcher.combine",
        "moe.shared_experts.*",
        "moe.tp_output_reduction",
        "moe.tp_routed_reduction",
    ]
)
```

Without EP there is no communication to save, so the local expert ordering
and the score-weighted scatter-add are ordinary operations. Operations outside
these regions, including token-shard zero-fill and branch addition, are
recomputed. Routing decisions are retained separately to keep expert selection
identical during replay.
