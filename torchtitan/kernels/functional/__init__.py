# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

# NOTE: vendored from lm_engine/kernels/functional/, trimmed to only the
# swiglu kernel. lm_engine's other kernels (cross_entropy, rmsnorm,
# fused_linear_cross_entropy, p_norm, softmax, sequence_packing,
# continuous_count, ...) were not copied.

from .swiglu import swiglu, swiglu_packed
