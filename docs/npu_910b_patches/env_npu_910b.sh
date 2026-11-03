# Source after activating the training venv / CANN env.
# 910B4: multi-process inductor / torch_npu precompile pools have segfaulted
# during flex-attention autotune; keep both at 1.
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHNPU_PRECOMPILE_THREADS=1
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-1}"
