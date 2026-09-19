# MoE Sharding

Config-based sharding for MoE submodules, implemented in
[`moe_sharding.py`](moe_sharding.py).

## Overview

The diagram below shows the DTensor placement flow through the MoE layer for
the two supported expert-parallel configurations (SP on/off). Without EP, the
routed experts are replicated across the dense TP axis instead of using ETP.

![MoE Sharding](../../../assets/images/moe_sharding.png)

## Configurations

"MoE input src → dst" shows the input redistribution at the MoE boundary.
"Routed expert weights" describes the routed expert weight placement.

| Config | Routed expert mesh | Routed expert weights | MoE input src → dst | MoE output |
|--------|-------------------|----------------------|---------------------|------------|
| EP on, SP on | sparse (EP/EFSDP) | `Shard(0)` on EP | `Shard(1)` → `Shard(1)` | `Partial` → `Shard(1)` |
| EP on, SP off | sparse (EP/EFSDP) | `Shard(0)` on EP | `Replicate` → `Replicate` | `Partial` → `Replicate` |

## Submodule sharding

- **MoE wrapper**: input/output redistribution between `sp_layout` and
  `desired_input_layouts`. With EP, output is sequence-sharded or `Partial`
  and redistributed to `sp_layout` at the boundary. Without EP, the replicated
  output is redistributed to `sp_layout`.
- **Router gate**: weights `Replicate`, output stays DTensor.
- **Shared experts** (w13/w2): dense-family TP plan. Colwise for w13 and
  rowwise for w2. When EP is disabled, the `Partial` output is reduced to
  `Replicate` before it is added to the replicated routed-expert output.
- **Routed experts** (`RoutedExperts`): the local SPMD region runs
  dispatch/compute/combine on local tensors while checking its input and
  output layout contracts. The expert-weight `state_shardings` live on its
  `GroupedExperts` child and are replicated across TP when EP is disabled.
