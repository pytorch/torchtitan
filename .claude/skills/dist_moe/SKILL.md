---
name: dist_moe
description: Configure, validate, and debug TorchTitan training with the standalone DistMoE BF16 or MXFP8 routed-expert backend.
---

# DistMoE In TorchTitan

Use the authoritative integration contract in
`torchtitan/components/dist_moe/README.md` before changing recipes or runtime
setup.

## Configuration Workflow

1. Start from a model whose routed experts use structured `GroupedLinear` W13
   and W2 projections, SwiGLU, and the standard all-to-all dispatcher config.
2. Apply `DistMoeTransform` to select BF16 DistMoE.
3. Optionally apply `MXFP8DistMoeTransform` after it. Configure dense MXFP8
   linears separately.
4. Keep `training.mixed_precision_param="bfloat16"` and use SM100-or-newer
   hardware.
5. Leave VMM disabled unless the run intentionally needs host-backed scratch.
6. For pipeline runs, use a schedule that exposes static liveness metadata.

Do not instantiate or invoke the stock token dispatcher from the transformed
module. Routing remains in TorchTitan, while DistMoE owns dispatch through
combine.

## Verification Ladder

1. Run focused CPU config and ownership tests.
2. Run local BF16 and MXFP8 non-VMM GPU tests for the exact FSDP and activation
   checkpointing policies under review.
3. Run internal-versus-annex byte-parity tests for outputs and gradients.
4. Run real distributed eager and CUDA-graph tests for the target EP/PP shape.
5. Run VMM tests only on an isolated MAST host; never use a shared devgpu for
   VMM debugging.
6. Inspect memory summaries and traces before making performance or overlap
   claims.

Fake and real-PP/fake-SPMD runs establish model ownership, shapes, and allocator
lifetimes. They do not establish communication performance or distributed
numerical parity. Use the `fake_distributed_backend` skill for that workflow.

## Failure Triage

- A transform-time failure means the stock routed-expert contract cannot be
  preserved; do not add a fallback that silently drops behavior.
- An initialization failure belongs to topology, capacity, hardware, or shared
  runtime validation; inspect the resolved memory plan before changing policy.
- An MXFP8 failure should be isolated across dynamic preparation, FSDP first
  unshard, refill, reshard, and checkpoint load.
- A pipeline slot failure should be reproduced from the final schedule IR and
  canonical `(stage, microbatch)` metadata.
- Never modify or duplicate annex kernels to compensate for an integration
  failure. Bitwise parity with the internal oracle is a release requirement.
