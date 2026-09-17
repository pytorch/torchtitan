# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from .math import get_powers_of_2


LIBRARY_NAME = "xma"

WARP_SIZE = 32
LOG_WARP_SIZE = 5

MAX_CUDA_BLOCK_SIZE = 1024
COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2 = get_powers_of_2(32, MAX_CUDA_BLOCK_SIZE)

MAX_TRITON_BLOCK_SIZE = 65536
COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2 = get_powers_of_2(64, MAX_TRITON_BLOCK_SIZE)

THREAD_BLOCK_CLUSTER_SIZES = get_powers_of_2(1, 8)
