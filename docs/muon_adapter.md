# Muon Adapter

`MuonAdapter` runs PyTorch Muon on temporary plain local tensors while retaining
DTensor parameters and momentum as the persistent optimizer and checkpoint
objects. It requires the batched-matrix behavior from PyTorch PR #190597.
It is an internal execution adapter, not a separate optimizer name; users
continue to configure `optimizer_name="Muon"`.

The implementation has two independent layers:

1. `spmd.dtensor_compute_view()` performs generic physical
   storage-to-compute redistribution and writeback.
2. `MuonAdapter` optionally views the resulting physical tensor as a logical
   matrix batch such as `[H, Dh, D]`.

The first layer contains no optimizer or head/expert semantics and can also be
used by storage-level operations such as gradient clipping.

Select parameters explicitly with ordered regular expressions:

```python
OptimizersContainer.Config(
    param_groups=[
        ParamGroupConfig(
            pattern=r"routed_experts\.(w1_EFD|w2_EDF|w3_EFD)$",
            optimizer_name="Muon",
            optimizer_kwargs={"lr": 0.02, "momentum": 0.95},
        ),
        ParamGroupConfig(
            pattern=r".*",
            optimizer_name="AdamW",
            optimizer_kwargs={"lr": 3e-4, "weight_decay": 0.1},
        ),
    ]
)
```

Every selected tensor must logically have shape `[..., M, N]`. `MuonAdapter`
chooses a physical compute placement containing complete matrices. Leading
matrix-batch shards are retained; shards that split `M` or `N` are temporarily
redistributed to `Replicate` and written back to storage after the update.
Residual `Partial` gradients fail before the update.

Grouped expert weights such as `[E, F, D]` work directly when only `E` is
sharded. Physically structured head weights such as `[H, Dh, D]` follow the
same rule.

For a replicated flattened projection `[H * Dh, D]`, declare the logical
matrix shape explicitly:

```python
ParamGroupConfig(
    pattern=r"attention\.wq\.weight$",
    optimizer_name="Muon",
    optimizer_kwargs={
        "lr": 0.02,
        "matrix_shape": (head_dim, model_dim),
    },
)
```

`matrix_shape` is applied only after the generic storage-to-compute transition.
For a physically sharded flattened tensor, `MuonAdapter` first requests a
replicated physical compute tensor and then applies the storage-sharing logical
view. The reshape itself never owns communication or persistent storage.
