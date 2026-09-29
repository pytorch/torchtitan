## BF16 Optimizer States

In the default fp32 training configuration (`training.dtype="float32"`), Adam/AdamW keep momentum (`exp_avg`) and variance (`exp_avg_sq`) in float32, which roughly doubles optimizer-state memory versus storing those buffers in bfloat16.

Set an Adam/AdamW optimizer config's `moment_dtype` to **`"bfloat16"`** to use the fused CUDA kernel with **bf16 optimizer moments** and **fp32 parameters**. This lowers optimizer-state memory while keeping parameter updates in fp32. FSDP may still reduce gradients in bf16. It casts the reduced gradient shards to fp32 before the optimizer step.

If you use **`training.dtype="bfloat16"`** (params and grads in bf16), leave `moment_dtype="parameter"` (the default). PyTorch then aligns optimizer state dtypes with training.

This is useful for memory-constrained training where slightly lower precision in moment estimates is acceptable.

### Background

This technique was notably used by [DeepSeek-V3](https://arxiv.org/abs/2412.19437) to train their 671B-parameter MoE model on 14.8 trillion tokens with reduced memory overhead. Their approach demonstrated that both momentum and variance buffers can be stored in bfloat16 without convergence issues, particularly for MoE architectures where expert gradients are smaller in magnitude. The effort to add native bf16 AdamW support to PyTorch is tracked in [pytorch/pytorch#146542](https://github.com/pytorch/pytorch/issues/146542).

### Usage

To use BF16 optimizer states with FP32 persistent parameters, use:

```python
config.training.dtype = "float32"
config.training.mixed_precision_param = "bfloat16"
config.training.mixed_precision_reduce = "float32"
# config.training.mixed_precision_reduce = "bfloat16"  # Optional BF16 reduction.
config.optimizer.optimizers[0].moment_dtype = "bfloat16"
```

The reduction dtype is independent of the optimizer-state dtype. Keep
`mixed_precision_reduce="float32"` for FP32 gradient reduction. Set it to
`"bfloat16"` to reduce FSDP communication bandwidth. With BF16 reduction, FSDP
casts each reduced gradient shard back to the FP32 parameter dtype before the
optimizer step.

### Requirements

- **Optimizer**: Must be `Adam` or `AdamW`.
- **Implementation**: The optimizer config must set `fused=True`. The fused CUDA kernel (`FusedAdamMathFunctorMP`) handles mixed-precision updates (fp32 parameters + bf16 moments). Configuration rejects bf16 moments with the for-loop or foreach implementation.

These constraints are validated at config time.

### How it works

A step pre-hook is registered on each optimizer instance. Before Adam's lazy state initialization runs on the first step, the hook pre-populates `exp_avg` and `exp_avg_sq` as bfloat16 tensors. When `_init_group` finds non-empty state, it skips its own fp32 allocation. The fused kernel detects the dtype mismatch between fp32 parameters and bf16 states and dispatches to the mixed-precision code path.

### Interaction with other features

- **`training.dtype`**: The primary use case is `float32` training with `moment_dtype="bfloat16"` for optimizer-state memory savings. With `bfloat16` training, the default `moment_dtype="parameter"` is usually enough; see the introduction above.
- **Checkpointing**: Optimizer moments are saved in bfloat16 when this option is enabled. On resume, use the same `moment_dtype`. A load post-hook restores `exp_avg`, `exp_avg_sq`, and optional AMSGrad `max_exp_avg_sq` to bfloat16 after PyTorch's native loader casts them to the parameter dtype. The `step` counter retains PyTorch's dtype and device policy. This preserves bf16 moments for subsequent updates, but the native loader can still temporarily allocate fp32 states during loading. Mixing moment dtypes across save/load remains unsupported.
- **FSDP**: Compatible with FSDP2. The optimizer sees DTensor parameters; the bf16 state hook operates on the local shards.

### Limitations

- Only the TorchTitan `Adam` and `AdamW` wrappers support `moment_dtype="bfloat16"`.
- The optimizer must use `fused=True`; for-loop and foreach Adam kernels do not support FP32 parameters with BF16 moments.
- Lower precision in moment estimates may affect convergence for some models or hyperparameter settings. Users should verify loss convergence for their specific use case.
