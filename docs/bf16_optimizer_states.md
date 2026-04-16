## BF16 Optimizer States

In the default fp32 training configuration (`training.dtype="float32"`), Adam/AdamW keep momentum (`exp_avg`) and variance (`exp_avg_sq`) in float32, which roughly doubles optimizer-state memory versus storing those buffers in bfloat16.

Set `optimizer.implementation` to **`fused_opt_states_bf16`** to use the fused Adam/AdamW CUDA kernel with **bf16 optimizer states** ,**fp32 parameters** and **fp32 grads** (mixed precision). That is the main scenario this option targets: lower optimizer memory while keeping params and grads in full precision.

If you use **`training.dtype="bfloat16"`** (params and grads in bf16), you typically keep **`implementation="fused"`** (default). PyTorch then aligns optimizer state dtypes with training; you do not need `fused_opt_states_bf16` unless you explicitly want the pre-hook initialization path (behavior should match fused training in practice).

This is useful for memory-constrained training where slightly lower precision in moment estimates is acceptable.

### Background

This technique was notably used by [DeepSeek-V3](https://arxiv.org/abs/2412.19437) to train their 671B-parameter MoE model on 14.8 trillion tokens with reduced memory overhead. Their approach demonstrated that both momentum and variance buffers can be stored in bfloat16 without convergence issues, particularly for MoE architectures where expert gradients are smaller in magnitude. The effort to add native bf16 AdamW support to PyTorch is tracked in [pytorch/pytorch#146542](https://github.com/pytorch/pytorch/issues/146542).

### Usage

In your config registry function:

```python
from torchtitan.components.optimizer import OptimizersContainer

optimizer=OptimizersContainer.Config(
    name="AdamW",
    implementation="fused_opt_states_bf16",
),
```

Or via CLI override:

```bash
--optimizer.name AdamW --optimizer.implementation fused_opt_states_bf16
```

### Requirements

- **Optimizer**: Must be `Adam` or `AdamW`.
- **Implementation**: Must be `fused_opt_states_bf16`. The fused CUDA kernel (`FusedAdamMathFunctorMP`) handles mixed-precision updates (fp32 parameters + bf16 states).

These constraints are validated at config time.

### How it works

A step pre-hook is registered on each optimizer instance. Before Adam's lazy state initialization runs on the first step, the hook pre-populates `exp_avg` and `exp_avg_sq` as bfloat16 tensors. When `_init_group` finds non-empty state, it skips its own fp32 allocation. The fused kernel detects the dtype mismatch between fp32 parameters and bf16 states and dispatches to the mixed-precision code path.

### Interaction with other features

- **`training.dtype`**: Primary use case is `float32` training with `fused_opt_states_bf16` for optimizer-state memory savings. With `bfloat16` training, default `implementation="fused"` is usually enough; see the introduction above.
- **Checkpointing**: Optimizer states are saved in bfloat16 when this option is enabled. On resume, use the same `implementation="fused_opt_states_bf16"` so checkpoint state matches. The pre-hook only creates bf16 tensors for parameters with empty state; if a checkpoint already populated state, those dtypes are preserved. Mixing checkpoint dtype with a different implementation across save/load is unsupported and can result in dtype-mismatch.
- **FSDP**: Compatible with FSDP2. The optimizer sees DTensor parameters; the bf16 state hook operates on the local shards.

### Limitations

- Only supported with `OptimizersContainer` (standard forward/backward training). Not supported with `OptimizersInBackwardContainer` (optimizer-step-in-backward); that combination is rejected in `OptimizersInBackwardContainer.Config.__post_init__`.
- Only `Adam` and `AdamW` with `fused_opt_states_bf16` are supported.
- Lower precision in moment estimates may affect convergence for some models or hyperparameter settings. Users should verify loss convergence for their specific use case.
