# Low-Precision Training

TorchTitan provides the following low-precision training formats:

- [Float8](float8/README.md) for dense and MoE models on Hopper and newer GPUs.
- [MXFP8](mxfp8/README.md) for dense and MoE models on Blackwell GPUs.
- [NVFP4](nvfp4/README.md) for experimental dense training on Blackwell GPUs.

## TorchAO and TorchTitan Boundary

TorchTitan owns the training integration for these formats: modules, autograd,
recipe selection, distributed composition, and the lifecycle of quantized
weight operands. Persistent model parameters and checkpoint state remain in
high precision.

TorchAO supplies the low-level quantization, layout-transformation, and matrix
multiplication kernels. This boundary lets TorchTitan integrate low-precision
training with FSDP and other parallelisms while TorchAO evolves and optimizes
the underlying kernels independently.
