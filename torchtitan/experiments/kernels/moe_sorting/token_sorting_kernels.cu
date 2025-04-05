// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.

/*
 *  Token sorting kernels
 *  sequential and parallel scans
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector.h>

#include "moe_kernel_utils.h"

// our utility namespace
using namespace moe_kernel_utils;

//
// kernels for sorting tokens by expert assignment
//

__global__ void sort_tokens_by_expert_kernel(

    )

    // gather kernel - move tokens to sorted indices
    template <template scalar_t>
    __global__ void gather_sorted_tokens_kernel(
        scalar_t *sorted_tokens, // output: sorted token features
    )
