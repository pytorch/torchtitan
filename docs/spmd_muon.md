# SPMD-Typed Muon

`SPMDMuon` runs PyTorch Muon on temporary plain local tensors while retaining
DTensor parameters and momentum as the persistent optimizer and checkpoint
objects. It requires the batched-matrix behavior from PyTorch PR #190597.

Select parameters explicitly with ordered regular expressions:

```python
OptimizersContainer.Config(
    param_groups=[
        ParamGroupConfig(
            pattern=r"routed_experts\.(w1_EFD|w2_EDF|w3_EFD)$",
            optimizer_name="SPMDMuon",
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

Every selected tensor must logically have shape `[..., M, N]`. Sharding is
allowed only on dimensions before `M` and `N`; `Partial`, unqualified
`Varying`, and matrix-dimension shards fail before the update.

Grouped expert weights such as `[E, F, D]` work directly when only `E` is
sharded. Physically structured head weights such as `[H, Dh, D]` follow the
same rule.

For a replicated flattened projection `[H * Dh, D]`, declare the logical
matrix shape explicitly:

```python
ParamGroupConfig(
    pattern=r"attention\.wq\.weight$",
    optimizer_name="SPMDMuon",
    optimizer_kwargs={
        "lr": 0.02,
        "matrix_shape": (head_dim, model_dim),
    },
)
```

`matrix_shape` is a storage-sharing view only. It is rejected for physically
sharded flattened tensors because the current metadata cannot prove that shard
boundaries coincide with head boundaries. Store those parameters as
`[H, Dh, D]` and shard `H` instead.
