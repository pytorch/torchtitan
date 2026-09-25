# MoE Sharding

Config-based sharding for MoE submodules, implemented in
[`moe_sharding.py`](moe_sharding.py). Mesh axis names (`ep`, `efsdp`) and how
EP reuses ranks from the dense world mesh are in
[`torchtitan/distributed/PARALLEL_DIMS.md`](../../distributed/PARALLEL_DIMS.md).

## Overview

The diagram below shows the MoE layer's SPMD layout flow using `spmd_types` for
the two supported expert-parallel configurations (SP on/off). For MoE models,
`expert_parallel_degree` must be at least `tensor_parallel_degree`.

![MoE Sharding](../../../assets/images/moe_sharding.png)

[Excalidraw source](https://excalidraw.com/#json=fZ1o2BuwXSVIGQhXM5gbr,KHKKckUbQXm68uiS3i_L7w)

## Configurations

"MoE input src → dst" shows the input redistribution at the MoE boundary.
"Routed input src → dst" shows the redistribution before routed-expert
dispatch and computation.
"Routed expert weights" describes the routed expert weight placement.

| Config | Routed expert mesh | Routed expert weights | MoE input src → dst | Routed input src → dst | MoE output |
|--------|-------------------|----------------------|---------------------|------------------------|------------|
| EP on, SP on | sparse (EP/EFSDP) | `Shard(0)` on EP | `Shard(0)` → `Shard(0)` | `Shard(0)` → `Shard(0)` | `Partial` → `Shard(0)` |
| EP on, SP off | sparse (EP/EFSDP) | `Shard(0)` on EP | `Replicate` → `Replicate` | `Replicate` → `Shard(0)` | `Partial` → `Replicate` |

## Submodule sharding

- **MoE wrapper**: input/output redistribution between `sp_layout` and
  `desired_input_layouts`. With EP, output is sequence-sharded or `Partial`
  and redistributed to `sp_layout` at the boundary. Without EP, TP/SP are
  disabled and routed-expert activations remain replicated.
- **Router gate**: weights `Replicate`, output stays DTensor.
- **Shared experts** (w13/w2): dense-family TP plan. `ColumnParallelLinear`
  gathers the w13 input. The local w2 projection produces `Partial`. With EP
  and SP off, it stays `Partial` until the shared and routed paths are added,
  then the MoE boundary performs the single all-reduce. With SP on, the w2
  boundary reduces `Partial` to `Shard(0)` before the paths are added.
- **Routed experts** (`RoutedExperts`): the local SPMD region runs
  dispatch/compute/combine on local tensors while checking its input and
  output layout contracts. Expert-weight `state_shardings` live on its `w13`
  and `w2` grouped linears and are unsharded when EP is disabled.
