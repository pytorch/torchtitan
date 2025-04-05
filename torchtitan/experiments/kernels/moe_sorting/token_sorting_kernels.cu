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

// Sequential exclusive prefix sum
// for small num experts .. < 64?
__device__ void
sequential_prefix_sum(int *offsets,      // output: exclusive prefix sum
                      const int *counts, // input: values to sum
                      int num_elements) {
  int running_sum = 0;
  for (int i = 0; i < num_elements; i++) {
    offsets[i] = running_sum;
    running_sum += counts[i];
  }
}

// Parallel exclusive prefix sum  - uses up and down sweep (binary tree style)
// should be faster for larger num experts ala maybe > 64
// TODO - time this and determine threshold to switch between
__device__ void
parallel_prefix_sum(int *offsets, // output and input from shared memory
                    int num_elements, int thread_id) {
  // up sweep (reduction)
  for (int stride = 1; stride < num_elements; stride *= 2) {
    int index = (thread_id + 1) * 2 * stride - 1; // 2i-1
    if (index < num_elements) {
      offsets[index] += offsets[index - stride];
    }
  }
  __syncthreads();

  // set last elem to zero
  if (thread_id == 0) {
    offsets[num_elements - 1] = 0;
  }
  __syncthreads();

  // Down sweep (distribute values)
  // int temp = 0;
  for (int stride = num_elements / 2; stride > 0; stride /= 2) {
    int index = (thread_id + 1) * 2 * stride - 1;
    if (index < num_elements) {
      int temp = offsets[index];
      offsets[index] += offsets[index - stride];
      offsets[index - stride] = temp;
    }
    __syncthreads();
  }
}

__global__ void sort_tokens_by_expert_kernel(

    )

    // gather kernel - move tokens to sorted indices
    template <template scalar_t>
    __global__ void gather_sorted_tokens_kernel(
        scalar_t *sorted_tokens, // output: sorted token features
    )
