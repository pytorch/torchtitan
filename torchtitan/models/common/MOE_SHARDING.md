# MoE Sharding

Config-based sharding for MoE submodules, implemented in
[`moe_sharding.py`](moe_sharding.py).

## Overview

The diagram below shows the DTensor placement flow through the MoE layer
for all four parallelism configurations (EP on/off × SP on/off).

![MoE Sharding](../../../assets/images/moe_sharding.png)

([Excalidraw source](https://excalidraw.com/#json=2abKr0m2s26fc6lyoF9Qq,MqMzUIoXWYJIfckHNOB7Sw))

## Configurations

"MoE input src → dst" shows the input redistribution at the MoE boundary.
"Routed expert weights" describes the routed expert weight placement.

| Config | Routed expert mesh | Routed expert weights | MoE input src → dst | MoE output |
|--------|-------------------|----------------------|---------------------|------------|
| EP on, SP on | sparse (EP/EFSDP) | `Shard(0)` on EP | `Shard(1)` → `Shard(1)` | `Partial` → `Shard(1)` |
| EP on, SP off | sparse (EP/EFSDP) | `Shard(0)` on EP | `Replicate` → `Replicate` | `Partial` → `Replicate` |
| EP off, SP on | dense (TP) | TP-sharded (colwise/rowwise) | `Shard(1)` → `Replicate` | `Partial` → `Shard(1)` |
| EP off, SP off | dense (TP) | TP-sharded (colwise/rowwise) | `Replicate` → `Replicate` | `Partial` → `Replicate` |

## Submodule sharding

- **MoE wrapper**: input/output redistribution between `sp_layout` and
  `desired_input_layouts`. Output is `Partial`, reduced to `sp_layout`
  at the boundary.
- **Router gate**: weights `Replicate`, output stays DTensor.
- **Shared experts** (w13/w2): dense-family TP plan. The standard w13 is a
  `ColumnParallelLinear` that owns its input redistribution, and w2 is a
  `RowParallelLinear` that owns its output redistribution. Without SP, the
  output stays `Partial` so reduction happens once at the MoE boundary. A
  model with multiple projections consuming the same input may instead gather
  once at its shared-expert boundary; Qwen3.5 uses this for w13 and its sigmoid
  gate.
- **Routed experts** (`RoutedExperts`): the local SPMD region runs
  dispatch/compute/combine on local tensors while checking its input and
  output layout contracts. The expert-weight `state_shardings` live on its
  `GroupedExperts` child.

When EP is disabled, routed-expert weights may still use expert tensor
parallelism (ETP): the dense TP axis shards each expert's input/output tensor
dimensions according to `expert_param_layout`. Enabling or disabling the
shared-expert projection boundaries does not remove this routed-expert ETP
path.
