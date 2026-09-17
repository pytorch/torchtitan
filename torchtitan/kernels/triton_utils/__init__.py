# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

# NOTE: vendored from lm_engine/kernels/triton_utils/, trimmed to only what
# the swiglu kernels need. lm_engine's cu_seqlens.py, matmul.py, and norm.py
# were not copied since nothing under torchtitan/kernels/ imports them.

from .activations import (
    clamp,
    leaky_relu,
    leaky_relu_backward,
    sigmoid,
    sigmoid_backward,
    silu,
    silu_backward,
    tanh,
    tanh_backward,
)
from .elementwise import elementwise_2d_kernel, get_elementwise_2d_configs
