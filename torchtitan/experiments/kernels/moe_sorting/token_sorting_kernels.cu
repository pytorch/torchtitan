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

// CUDA kernel to count tokens per expert (PyTorch-compatible)
__global__ void count_tokens_per_expert_kernel(
    int *tokens_per_expert, // output:  [n_experts]
    const int *topk_ids,    // input: expert assignments [seq_len, k]
    int seq_len, int n_experts,
    int k // top-k experts per token
) {
  extern __shared__ int s_expert_presence[];

  // Process tokens in batches to avoid excessive shared memory usage
  const int tokens_per_batch = 128; // Process 128 tokens at a time

  for (int batch_start = 0; batch_start < seq_len;
       batch_start += tokens_per_batch) {
    int batch_end = min(batch_start + tokens_per_batch, seq_len);
    int batch_size = batch_end - batch_start;

    // For each token, we track which experts are assigned to it
    for (int i = threadIdx.x; i < batch_size * n_experts; i += blockDim.x) {
      s_expert_presence[i] = 0;
    }
    __syncthreads();

    // Mark which experts are assigned to each token
    for (int token_offset = threadIdx.x; token_offset < batch_size;
         token_offset += blockDim.x) {
      int token_idx = batch_start + token_offset;

      for (int j = 0; j < k; j++) {
        int expert_id = topk_ids[token_idx * k + j];
        if (expert_id >= 0 && expert_id < n_experts) {
          // Mark this expert as assigned to this token (once)
          s_expert_presence[token_offset * n_experts + expert_id] = 1;
        }
      }
    }
    __syncthreads();

    // Sum up experts across this batch
    for (int expert_id = threadIdx.x; expert_id < n_experts;
         expert_id += blockDim.x) {
      int count = 0;
      for (int token_offset = 0; token_offset < batch_size; token_offset++) {
        count += s_expert_presence[token_offset * n_experts + expert_id];
      }
      if (count > 0) {
        // note that atomicAdd does not work for int64...
        atomicAdd(&tokens_per_expert[expert_id], count);
      }
    }
    __syncthreads();
  }
}

// prepare flatten-and-argsort inputs
__global__ void prepare_flattened_experts_kernel(
    int *flattened_experts, // output: flattened expert assignments
    const int *topk_ids,    // input: expert assignments [seq_len, k]
    int total_elements      // total number of elements (seq_len * k)
) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements;
       idx += blockDim.x * gridDim.x) {
    flattened_experts[idx] = topk_ids[idx];
  }
}

// generate token indices based on sort order
__global__ void generate_sorted_token_indices_kernel(
    int *sorted_token_indices,   // output: token indices for gathering features
    const int64_t *sort_indices, // input: indices from argsort (int64)
    int total_elements,
    int k // k value for integer division
) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements;
       idx += blockDim.x * gridDim.x) {
    // get token index from flattened index
    sorted_token_indices[idx] = static_cast<int>(sort_indices[idx]) / k;
  }
}

// gather (reorder really) token features
template <typename scalar_t>
__global__ void gather_sorted_tokens_kernel(
    scalar_t *sorted_tokens,         // output: sorted token features
    const scalar_t *input_tokens,    // input: original token features
    const int *sorted_token_indices, // input: token indices for gathering
    int total_elements,
    int hidden_dim // hidden dimension size
) {
  // Calculate global thread indices
  int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int feat_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (token_idx < total_elements && feat_idx < hidden_dim) {
    int src_idx = sorted_token_indices[token_idx];
    sorted_tokens[token_idx * hidden_dim + feat_idx] =
        input_tokens[src_idx * hidden_dim + feat_idx];
  }
}

//
// wrapper functions
//

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sort_tokens_by_expert_cuda(torch::Tensor topk_ids, torch::Tensor x,
                           int n_experts, bool use_parallel_scan = false) {

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

  // Convert input to int32 if needed - this is an atomicAdd limitation...
  torch::Tensor topk_ids_int;
  if (topk_ids.scalar_type() == torch::kInt64) {
    topk_ids_int = topk_ids.to(torch::kInt32);
  } else {
    topk_ids_int = topk_ids;
  }

  // Ensure tensor is contiguous
  if (!topk_ids_int.is_contiguous()) {
    topk_ids_int = topk_ids_int.contiguous();
  }

  // 1: Count tokens per expert using CUDA kernel
  auto tokens_per_expert =
      torch::zeros({n_experts}, torch::dtype(torch::kInt32).device(device));

  const int threads = 256;
  const int blocks = 1; // Single block is sufficient for counting
  int shared_mem_size =
      128 * n_experts * sizeof(int); // For 128 tokens at a time

  // Check if we have enough shared memory
  int max_shared_mem;
  cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlock,
                         device.index());

  /*if (shared_mem_size > max_shared_mem) {
    // Fall back to CPU implementation for counting if shared memory is
    // insufficient This is a rare case for very large expert counts
    auto cnts = torch::zeros({seq_len, n_experts},
                             torch::dtype(torch::kInt32).device(device));

    for (int i = 0; i < seq_len; i++) {
      for (int j = 0; j < k; j++) {
        int expert_id = topk_ids_int[i][j].item<int>();
        if (expert_id >= 0 && expert_id < n_experts) {
          cnts[i][expert_id] = 1;
        }
      }
    }
    tokens_per_expert = cnts.sum(0);
  } else {
  */
  // Use CUDA kernel for counting
  count_tokens_per_expert_kernel<<<blocks, threads, shared_mem_size>>>(
      tokens_per_expert.data_ptr<int>(), topk_ids_int.data_ptr<int>(), seq_len,
      n_experts, k);

  // Check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error in count_tokens_per_expert_kernel: %s\n",
           cudaGetErrorString(error));
    throw std::runtime_error("CUDA kernel execution failed");
  }

  // Step 2: Create flattened experts tensor
  auto flattened_experts = torch::empty(
      {total_elements}, torch::dtype(torch::kInt32).device(device));

  const int flatten_blocks = (total_elements + threads - 1) / threads;
  prepare_flattened_experts_kernel<<<flatten_blocks, threads>>>(
      flattened_experts.data_ptr<int>(), topk_ids_int.data_ptr<int>(),
      total_elements);

  // Check for errors
  // cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error in prepare_flattened_experts_kernel: %s\n",
           cudaGetErrorString(error));
    throw std::runtime_error("CUDA kernel execution failed");
  }

  // Step 3: Use PyTorch's built-in sort to get argsort indices
  // This is the most reliable way to match PyTorch's behavior but need to
  // verify no cpu sync..
  auto sort_indices = flattened_experts.argsort();

  // Step 4: Generate token indices for feature gathering
  auto sorted_token_indices = torch::empty(
      {total_elements}, torch::dtype(torch::kInt32).device(device));

  generate_sorted_token_indices_kernel<<<flatten_blocks, threads>>>(
      sorted_token_indices.data_ptr<int>(),
      sort_indices.data_ptr<int64_t>(), // Note: argsort returns int64 indices
      total_elements, k);

  // Check for errors
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error in generate_sorted_token_indices_kernel: %s\n",
           cudaGetErrorString(error));
    throw std::runtime_error("CUDA kernel execution failed");
  }

  // Step 5: Gather token features
  auto sorted_tokens = torch::empty({total_elements, hidden_dim},
                                    torch::dtype(x.dtype()).device(device));

  dim3 gather_threads(16, 16); // 16x16 = 256 threads per block
  dim3 gather_blocks((total_elements + gather_threads.x - 1) / gather_threads.x,
                     (hidden_dim + gather_threads.y - 1) / gather_threads.y);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x.scalar_type(), "gather_sorted_tokens_cuda", ([&] {
        gather_sorted_tokens_kernel<scalar_t>
            <<<gather_blocks, gather_threads>>>(
                sorted_tokens.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
                sorted_token_indices.data_ptr<int>(), total_elements,
                hidden_dim);
      }));

  // Check for errors
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error in gather_sorted_tokens_kernel: %s\n",
           cudaGetErrorString(error));
    throw std::runtime_error("CUDA kernel execution failed");
  }

  // Convert token counts to match input type
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
        py::arg("x"), py::arg("n_experts"),
        py::arg("use_parallel_scan") = false);
}
