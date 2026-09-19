# MoE Sharding

Config-based sharding for MoE submodules, implemented in
[`moe_sharding.py`](moe_sharding.py).

## Overview

The diagram below shows the placement flow through the MoE layer for all four
EP/SP configurations. With SP, the routed and shared branches independently
align their outputs before addition. Without SP, their partial outputs are
added before one final MoE all-reduce.

![MoE Sharding](../../../assets/images/moe_sharding.png)

([Excalidraw source](https://excalidraw.com/#json=2abKr0m2s26fc6lyoF9Qq,MqMzUIoXWYJIfckHNOB7Sw))

## Configurations

`S(0)` means the token dimension is sharded. `R`, `I`, and `P` mean
`Replicate`, `Invariant`, and `Partial`, respectively.
The diagram predates the `Invariant` type and labels physically replicated
no-SP activations as `R`; the table records their current `I` contract.

| EP | SP | Common/routed input | Shared input | Layouts at branch sum | Final MoE output |
|----|----|---------------------|--------------|-----------------------|------------------|
| on | on | `S(0)` unchanged | `S(0) -> R` | routed `S(0)` + shared `P -> S(0)` | `S(0)` |
| on | off | `I` unchanged | `I -> R` | routed `P` + shared `P` | `P -> I` |
| off | on | `S(0) -> R`, shared by both branches | reuse `R` | routed `P -> S(0)` + shared `P -> S(0)` | `S(0)` |
| off | off | `I -> R`, shared by both branches | reuse `R` | routed `P` + shared `P` | `P -> I` |

With EP enabled, routed-expert weights use the sparse EP/EFSDP mesh and are
`Shard(0)` on the expert dimension. Without EP, routed-expert weights use the
dense TP mesh and shard their input/output dimensions according to
`expert_param_layout`.

## Submodule sharding

- **MoE boundary**: without EP, `MoE.forward` gathers its input once for the
  routed and shared branches. With EP, the routed path retains its dispatcher
  layout while `MoE.forward` prepares the shared input independently. With SP
  disabled, it adds the two `Partial` outputs before one final all-reduce.
- **Router gate**: weights `Replicate`, output stays DTensor.
- **Shared experts** (w13/w2): `w13` is a compute-only `Linear` because the MoE
  prepares its shared input. `w2` is a `RowParallelLinear`: with SP it
  reduce-scatters `Partial -> S(0)` before the branch sum; without SP it leaves
  the output `Partial` for the final MoE all-reduce. With EP and no SP, the
  shared input's `Invariant -> Replicate` conversion is a forward no-op whose
  backward performs the required reduction.
- **Routed experts** (`RoutedExperts`): the local SPMD region runs
  dispatch/compute/combine on local tensors while checking its input and
  output layout contracts. The expert-weight `state_shardings` live on its
  `GroupedExperts` child.

When EP is disabled, routed-expert weights may still use expert tensor
parallelism (ETP): the dense TP axis shards each expert's input/output tensor
dimensions according to `expert_param_layout`. The MoE boundary gathers the
input activation once before both routed and shared computation, then reduces
their combined partial result when SP is disabled. With SP enabled, each branch
reduce-scatters independently before their sum.
