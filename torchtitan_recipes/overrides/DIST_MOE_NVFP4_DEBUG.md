# NVFP4 Dist-MoE debugging override

This opt-in adapter is **for debugging only**, not a production precision option.
It requires a separately installed `dist_moe` build that supports NVFP4 training
with native BF16 weights. Select that build on `PYTHONPATH` before launching.

```bash
python -m torchtitan.rl.train \
  --module torchtitan_recipes.tests.rl.alphabet_sort \
  --config rl_grpo_moe_debug_dist_moe_tp2_ep4 \
  --override torchtitan_recipes.overrides.dist_moe_nvfp4_debug.nvfp4_experts \
  --output-dir outputs/rl/nvfp4-debug
```

The existing random-weight toy uses four trainer GPUs and four rollout GPUs on
SM100+ hardware. The override selects staged NVFP4 expert execution in both roles
while preserving ordinary parameter storage, sharding, optimizer ownership, and
dense layers. Dist-MoE performs real NVFP4 tensor-core forward and BF16
straight-through-estimator DGRAD/WGRAD; this is not FP4 backward GEMM support.

The CI recipe disables W&B. Enable it separately through a metrics config
override if needed; this precision adapter does not change logging.

A successful random-weight toy run verifies integration, not convergence or
reference numerical parity. The supported production `DistMoeTransform` API is
unchanged. Remove or replace this debugging adapter when production NVFP4
integration is implemented.
