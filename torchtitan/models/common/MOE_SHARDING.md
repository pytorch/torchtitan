# MoE Sharding

Config-based sharding for MoE submodules, implemented in
[`moe_sharding.py`](moe_sharding.py).

## Overview

The diagram below shows the DTensor placement flow through the MoE layer
for all four parallelism configurations (EP on/off × SP on/off).

![MoE Sharding](../../../assets/images/moe_sharding.png)

([Excalidraw source](https://excalidraw.com/#json=2abKr0m2s26fc6lyoF9Qq,MqMzUIoXWYJIfckHNOB7Sw))

## Configurations

"MoE input src -> dst" shows the explicit input redistribution in `MoE.forward`.
"Routed expert weights" describes the routed expert weight placement.

| Config | Routed expert mesh | Routed expert weights | MoE input src -> dst | MoE output |
|--------|-------------------|----------------------|---------------------|------------|
| EP on, SP on | sparse (EP/EFSDP) | `Shard(0)` on EP | `Shard(1)` -> `Shard(1)` | `Shard(1)` |
| EP on, SP off | sparse (EP/EFSDP) | `Shard(0)` on EP | `Invariant` -> `Invariant` | `Partial` -> `Invariant` |
| EP off, SP on | dense (TP) | TP-sharded (colwise/rowwise) | `Shard(1)` -> `Replicate` | `Partial` -> `Shard(1)` |
| EP off, SP off | dense (TP) | TP-sharded (colwise/rowwise) | `Invariant` -> `Replicate` | `Partial` -> `Invariant` |

## Submodule sharding

- **MoE boundary**: without EP, `MoE.forward` gathers its input once for the
  routed and shared branches. It adds their `Partial` outputs before one final
  reduction. With EP, the routed path retains sequence-sharded tokens and the
  shared path owns its required communication.
- **Router gate**: weights `Replicate`, output stays DTensor.
- **Shared experts** (w13/w2): dense-family TP plan. With EP, the standard w13
  is a `ColumnParallelLinear` that gathers only the shared branch, and w2 is a
  `RowParallelLinear` that restores the routed branch's token layout. Without
  EP, the MoE boundary gathers once for both branches, adds their partial
  outputs, and reduces once. A model with multiple shared-expert projections
  may instead gather once at its shared-expert boundary; Qwen3.5 uses this for
  w13 and its sigmoid gate.
- **Routed experts** (`RoutedExperts`): the local SPMD region runs
  dispatch/compute/combine on local tensors while checking its input and
  output layout contracts. The expert-weight `state_shardings` live on its
  `GroupedExperts` child.

When EP is disabled, routed-expert weights may still use expert tensor
parallelism (ETP): the dense TP axis shards each expert's input/output tensor
dimensions according to `expert_param_layout`. The MoE boundary gathers the
shared activation before both routed and shared computation, then reduces their
combined partial result.
