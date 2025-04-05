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

//
// reorder (gather) tokens using sorted indices
//
template <template scalar_t>
__global__ void gather_sorted_tokens_kernel(
    scalar_t *sorted_tokens,      // output: sorted tokens
    const scalar_t *input_tokens, // input: original token features
    cnost int *sorted_indices,    // input:
    int seq_len,                  // M
    int hidden_dim                // N

) {
  // get global position
  int token_index = blockIdx.x * blockDim.x + threadIdx.x;
  int feat_index = blockIdx.y * blockDim.y + threadIdx.y;

  // update to new location
  if (token_index < seq_len && feat_index < hidden_dim) {
    int src_index = sorted_indices[token_index];
    // column_offset = hidden
    sorted_tokens[token_index * hidden_dim + feat_index] =
        input_tokens[src_index * hidden_dim + feat_index];
  }
}

//
// Wrapper functions
//

// main function
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sort_tokens_by_expert_cuda(torch::Tensor topk_ids, torch::Tensor x,
                           int n_experts, bool use_parallel_scan = false) {
  auto device = topk_ids.device();
  int seq_len = topk_ids.size(0);
  int k = topk_ids.size(1);
  int hidden_dim = x.size(1);

  // TODO...
}
