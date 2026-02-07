// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.

/*
 * Token sorting for MoE Models
 *
 */

#include "moe_kernel_utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

// Our utility namespace
using namespace moe_kernel_utils;

//
//  CUDA Kernels
//

// count tokens per expert
__global__ void optimized_count_tokens_kernel(
    int *tokens_per_expert,  // output: count of tokens per expert [n_experts]
    const int64_t *topk_ids, // input: expert assignments [seq_len, k]
    int seq_len,             // sequence length
    int n_experts,
    int k, // top-k experts per token
    int hidden_dim) {

  // for local counters
  extern __shared__ unsigned int s_expert_counts[];

  // Initialize shared memory to zero
  for (int i = threadIdx.x; i < n_experts; i += blockDim.x) {
    s_expert_counts[i] = 0;
  }
  __syncthreads();

  // Adjust tokens per thread based on hidden dimension size to reduce register
  // pressure for larger feature dimensions
  const int tokens_per_thread =
      (hidden_dim <= 1024) ? 4 : ((hidden_dim <= 4096) ? 2 : 1);
  const int token_stride = blockDim.x * gridDim.x;

  // Registers to track seen experts for each token
  // Use 32-bit words to track up to 32 experts per word
  // This handles up to 512 experts with 16 words
  unsigned int seen_experts[16] = {0};

  for (int base_idx = blockIdx.x * blockDim.x + threadIdx.x; base_idx < seq_len;
       base_idx += token_stride) {

#pragma unroll
    for (int t = 0; t < tokens_per_thread; t++) {
      const int token_idx = base_idx + t * token_stride;
      if (token_idx >= seq_len)
        break;

// Reset the seen experts tracker for this token
#pragma unroll
      for (int w = 0; w < 16; w++) {
        seen_experts[w] = 0;
      }

// Process all expert assignments for this token
#pragma unroll
      for (int j = 0; j < k; j++) {
        const int expert_id = static_cast<int>(topk_ids[token_idx * k + j]);
        if (expert_id >= 0 && expert_id < n_experts) {
          // Mark this expert as seen
          const int word_idx = expert_id >> 5; // expert_id / 32
          const int bit_idx = expert_id & 31;  // expert_id % 32
          seen_experts[word_idx] |= (1U << bit_idx);
        }
      }

// Update shared memory counters with seen experts
#pragma unroll
      for (int w = 0; w < 16; w++) {
        unsigned int word = seen_experts[w];
        while (word) {
          // Find the least significant set bit
          unsigned int bit_pos = __ffs(word) - 1;

          // Calculate expert id
          int expert_id = (w << 5) | bit_pos; // (w * 32) + bit_pos
          if (expert_id < n_experts) {
            // Increment counter for this expert
            // Use atomic since multiple threads may update same
            // counter
            atomicAdd(&s_expert_counts[expert_id], 1);
          }

          // Clear the processed bit and continue with next set bit
          word &= ~(1U << bit_pos);
        }
      }
    }
  }

  // Make sure all threads in the block have updated shared memory
  __syncthreads();

  // Contribute local counts to global counts with coalesced memory access
  for (int i = threadIdx.x; i < n_experts; i += blockDim.x) {
    if (s_expert_counts[i] > 0) {
      atomicAdd(&tokens_per_expert[i], s_expert_counts[i]);
    }
  }
}

// Optimized gather kernel for large feature dimensions
template <typename scalar_t>
__global__ void gather_sorted_tokens_kernel_large(
    scalar_t *sorted_tokens,      // output: sorted token features
    const scalar_t *input_tokens, // input: original token features
    const int64_t *sort_indices,  // input: indices from argsort
    int total_elements,           // total number of elements
    int hidden_dim,               // hidden dimension size
    int k                         // k value for integer division
) {
  // Calculate global thread ID
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate token and feature indices
  int token_idx = idx / hidden_dim;
  int feat_idx = idx % hidden_dim;

  if (token_idx < total_elements) {
    // Integer division to get token index from flattened index
    int src_idx = static_cast<int>(sort_indices[token_idx]) / k;
    sorted_tokens[token_idx * hidden_dim + feat_idx] =
        input_tokens[src_idx * hidden_dim + feat_idx];
  }
}

// reorder (gather)
template <typename scalar_t>
__global__ void gather_sorted_tokens_kernel(
    scalar_t *sorted_tokens,      // output: sorted token features
    const scalar_t *input_tokens, // input: original token features
    const int64_t *sort_indices,  // input: indices from argsort
    int total_elements,
    int hidden_dim, // hidden dimension size
    int k           // k value for integer division
) {
  // Calculate global thread indices
  int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int feat_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (token_idx < total_elements && feat_idx < hidden_dim) {
    // Integer division to get token index from flattened index
    int src_idx = static_cast<int>(sort_indices[token_idx]) / k;
    sorted_tokens[token_idx * hidden_dim + feat_idx] =
        input_tokens[src_idx * hidden_dim + feat_idx];
  }
}

//////////////////////////////////////////////////////////////////////////////
// C++/CUDA wrapper functions
//////////////////////////////////////////////////////////////////////////////

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sort_tokens_by_expert_cuda(torch::Tensor topk_ids, torch::Tensor x,
                           int n_experts) {

  auto device = topk_ids.device();
  int seq_len = topk_ids.size(0);
  int k = topk_ids.size(1);
  int hidden_dim = x.size(1);
  int total_elements = seq_len * k;

  // Validate inputs
  TORCH_CHECK(topk_ids.device().is_cuda(), "topk_ids must be a CUDA tensor");
  TORCH_CHECK(x.device().is_cuda(), "input tensor must be a CUDA tensor");
  TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be a 2D tensor");
  TORCH_CHECK(x.dim() == 2, "input tensor must be a 2D tensor");
  TORCH_CHECK(n_experts <= 512, "Maximum number of experts supported is 512");

  // Always use int64 for topk_ids to match PyTorch
  auto topk_ids_int64 = topk_ids;
  if (topk_ids.scalar_type() != torch::kInt64) {
    topk_ids_int64 = topk_ids.to(torch::kInt64);
  }
  topk_ids_int64 = topk_ids_int64.contiguous();

  // Step 1: Count tokens per expert using specialized CUDA kernel
  // Use int32 for token counts to avoid atomicAdd issues
  auto tokens_per_expert =
      torch::zeros({n_experts}, torch::dtype(torch::kInt32).device(device));

  // Optimize kernel launch parameters
  const int count_threads = 256;

  const int count_blocks =
      std::min(256, (seq_len + count_threads - 1) / count_threads);

  int shared_mem_size = n_experts * sizeof(int);

  // Make sure shared memory size is reasonable
  int max_shared_mem;
  cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlock,
                         device.index());
  if (shared_mem_size > max_shared_mem) {
    // Fall back to a smaller value
    shared_mem_size = max_shared_mem;
  }

  // Launch optimized counting kernel
  optimized_count_tokens_kernel<<<count_blocks, count_threads,
                                  shared_mem_size>>>(
      tokens_per_expert.data_ptr<int>(), topk_ids_int64.data_ptr<int64_t>(),
      seq_len, n_experts, k,
      hidden_dim // Pass hidden_dim for dynamic optimization
  );

  // Check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error in counting kernel: %s\n", cudaGetErrorString(error));
    throw std::runtime_error("CUDA kernel execution failed");
  }

  // Step 2: Use PyTorch's argsort for now...
  auto flattened_topk = topk_ids_int64.reshape({-1});
  auto sort_indices = flattened_topk.argsort();

  // Step 3: Gather the token features
  auto sorted_tokens = torch::empty({total_elements, hidden_dim}, x.options());

  // kernel strategy based on hidden dimension size
  if (hidden_dim > 2048) {
    // For large feature dimensions, use a 1D kernel with better memory
    // coalescing
    int block_size = 256;
    int num_blocks =
        (total_elements * hidden_dim + block_size - 1) / block_size;

    // Launch gather kernel for large dimensions
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x.scalar_type(), "gather_sorted_tokens_cuda_large", ([&] {
          gather_sorted_tokens_kernel_large<scalar_t>
              <<<num_blocks, block_size>>>(sorted_tokens.data_ptr<scalar_t>(),
                                           x.data_ptr<scalar_t>(),
                                           sort_indices.data_ptr<int64_t>(),
                                           total_elements, hidden_dim, k);
        }));
  } else {
    // For smaller dimensions, use the 2D kernel
    const int gather_token_threads = 16;
    const int gather_feature_threads = 16;

    dim3 gather_threads(gather_token_threads, gather_feature_threads);
    dim3 gather_blocks((total_elements + gather_threads.x - 1) /
                           gather_threads.x,
                       (hidden_dim + gather_threads.y - 1) / gather_threads.y);

    // Launch standard gather kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x.scalar_type(), "gather_sorted_tokens_cuda", ([&] {
          gather_sorted_tokens_kernel<scalar_t>
              <<<gather_blocks, gather_threads>>>(
                  sorted_tokens.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
                  sort_indices.data_ptr<int64_t>(), total_elements, hidden_dim,
                  k);
        }));
  }

  // Check for errors
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error in gather_sorted_tokens_kernel: %s\n",
           cudaGetErrorString(error));
    throw std::runtime_error("CUDA kernel execution failed");
  }

  // Convert token counts back to match input type
  torch::Tensor tokens_per_expert_out;
  if (topk_ids.scalar_type() == torch::kInt64) {
    tokens_per_expert_out = tokens_per_expert.to(torch::kInt64);
  } else {
    tokens_per_expert_out = tokens_per_expert;
  }

  return std::make_tuple(sorted_tokens, sort_indices, tokens_per_expert_out);
}

//////////////////////////////////////////////////////////////////////////////
// Python bindings
//////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sort_tokens_by_expert", &sort_tokens_by_expert_cuda,
        "Sort tokens by expert assignment (CUDA)", py::arg("topk_ids"),
        py::arg("x"), py::arg("n_experts"));
}
