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
#include <vector>

#include "moe_kernel_utils.h"

// our utility namespace
using namespace moe_kernel_utils;

//
// kernels for sorting tokens by expert assignment
//

// Sequential exclusive prefix sum
// for small num experts .. < 64?
__device__ void
sequential_prefix_sum(long *offsets,     // output: exclusive prefix sum
                      const int *counts, // input: values to sum
                      int num_elements) {
  long running_sum = 0;
  for (int i = 0; i < num_elements; i++) {
    offsets[i] = running_sum;
    running_sum += counts[i];
  }
}

// Parallel exclusive prefix sum  - uses up and down sweep (binary tree style)
// should be faster for larger num experts ala maybe > 64
// TODO - time this and determine threshold to switch between
__device__ void
parallel_prefix_sum(long *offsets, // output and input from shared memory
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
// Main sorting
//

__global__ void sort_tokens_by_expert_kernel(
    long *sorted_indices,    // output: sorted token indices [seq_len]
    long *tokens_per_expert, // output: number of tokens per expert [n_experts]
    long *expert_offsets, // output: starting offset for each expert [n_experts]
    const long *topk_ids, // input: expert assignments [seq_len, k]
    int seq_len,          // sequence length
    int n_experts,        // number of experts
    int k                 // top-k experts per token
) {
  // Use shared memory to track local counts
  extern __shared__ long s_expert_counts[];

  // Initialize shared memory counters
  for (int i = threadIdx.x; i < n_experts; i += blockDim.x) {
    s_expert_counts[i] = 0;
  }
  __syncthreads();

  // First pass: count tokens per expert in this block
  for (int token_idx = threadIdx.x + blockIdx.x * blockDim.x;
       token_idx < seq_len; token_idx += blockDim.x * gridDim.x) {

    // We only process the first expert assignment for each token
    long expert_id = topk_ids[token_idx * k];
    if (expert_id >= 0 && expert_id < n_experts) {
      atomicAdd(&s_expert_counts[expert_id], 1);
    }
  }
  __syncthreads();

  // Contribute this block's counts to global counts
  for (int i = threadIdx.x; i < n_experts; i += blockDim.x) {
    if (s_expert_counts[i] > 0) {
      atomicAdd(&tokens_per_expert[i], s_expert_counts[i]);
    }
  }
  __syncthreads();

  // Wait for all blocks to finish counting
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    // Compute exclusive prefix sum for offsets (sequential implementation)
    sequential_prefix_sum(expert_offsets, tokens_per_expert, n_experts);
  }
  __syncwarp();
  __threadfence(); // Ensure offsets are visible to all blocks

  // Second pass: place token indices in the sorted array
  for (int token_idx = threadIdx.x + blockIdx.x * blockDim.x;
       token_idx < seq_len; token_idx += blockDim.x * gridDim.x) {

    // We only process the first expert assignment for each token
    long expert_id = topk_ids[token_idx * k];
    if (expert_id >= 0 && expert_id < n_experts) {
      // Get position by atomically incrementing the expert offset
      long position = atomicAdd(&expert_offsets[expert_id], 1);
      sorted_indices[position] = token_idx;
    }
  }
}

// CUDA kernel to sort token indices by expert assignment using parallel scan
__global__ void sort_tokens_parallel_scan_kernel(
    long *sorted_indices,    // output: sorted token indices [seq_len]
    long *tokens_per_expert, // output: number of tokens per expert [n_experts]
    long *expert_offsets, // output: starting offset for each expert [n_experts]
    const long *topk_ids, // input: expert assignments [seq_len, k]
    int seq_len,          // sequence length
    int n_experts,        // number of experts
    int k,                // top-k experts per token
    bool use_parallel_scan // whether to use parallel scan
) {
  // Use shared memory to track local counts and for scan operation
  extern __shared__ long shared_mem[];
  long *s_expert_counts = shared_mem;
  long *s_scan_workspace = &shared_mem[n_experts];

  // Initialize shared memory counters
  for (int i = threadIdx.x; i < n_experts; i += blockDim.x) {
    s_expert_counts[i] = 0;
  }
  __syncthreads();

  // First pass: count tokens per expert in this block
  for (int token_idx = threadIdx.x + blockIdx.x * blockDim.x;
       token_idx < seq_len; token_idx += blockDim.x * gridDim.x) {

    // We only process the first expert assignment for each token
    long expert_id = topk_ids[token_idx * k];
    if (expert_id >= 0 && expert_id < n_experts) {
      atomicAdd(&s_expert_counts[expert_id], 1);
    }
  }
  __syncthreads();

  // Contribute this block's counts to global counts
  for (int i = threadIdx.x; i < n_experts; i += blockDim.x) {
    if (s_expert_counts[i] > 0) {
      atomicAdd(&tokens_per_expert[i], s_expert_counts[i]);
    }
  }
  __syncthreads();

  // Wait for all blocks to finish counting
  if (blockIdx.x == 0) {
    // Copy token counts to scan workspace
    if (threadIdx.x < n_experts) {
      s_scan_workspace[threadIdx.x] = tokens_per_expert[threadIdx.x];
    }
    __syncthreads();

    if (use_parallel_scan && n_experts >= 64) {
      // Use parallel scan for larger expert counts
      parallel_prefix_sum(s_scan_workspace, n_experts, threadIdx.x);
    } else {
      // Use sequential scan for smaller expert counts
      if (threadIdx.x == 0) {
        sequential_prefix_sum(s_scan_workspace, tokens_per_expert, n_experts);
      }
      __syncthreads();
    }

    // Copy scan results back to expert offsets
    if (threadIdx.x < n_experts) {
      expert_offsets[threadIdx.x] = s_scan_workspace[threadIdx.x];
    }
  }
  __syncthreads();
  __threadfence(); // Ensure offsets are visible to all blocks

  // Second pass: place token indices in the sorted array
  for (int token_idx = threadIdx.x + blockIdx.x * blockDim.x;
       token_idx < seq_len; token_idx += blockDim.x * gridDim.x) {

    // We only process the first expert assignment for each token
    long expert_id = topk_ids[token_idx * k];
    if (expert_id >= 0 && expert_id < n_experts) {
      // Get position by atomically incrementing the expert offset
      long position = atomicAdd(&expert_offsets[expert_id], 1);
      sorted_indices[position] = token_idx;
    }
  }
}

//
// reorder (gather) tokens using sorted indices
//
template <typename scalar_t>
__global__ void gather_sorted_tokens_kernel(
    scalar_t *sorted_tokens,      // output: sorted tokens
    const scalar_t *input_tokens, // input: original token features
    const int *sorted_indices,    // input:
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

  // Validate inputs
  TORCH_CHECK(topk_ids.device().is_cuda(), "topk_ids must be a CUDA tensor");
  TORCH_CHECK(x.device().is_cuda(), "input tensor must be a CUDA tensor");
  TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be a 2D tensor");
  TORCH_CHECK(x.dim() == 2, "input tensor must be a 2D tensor");

  // Output tensors
  auto sorted_indices =
      torch::empty({seq_len}, torch::dtype(torch::kInt64).device(device));
  auto tokens_per_expert =
      torch::zeros({n_experts}, torch::dtype(torch::kInt64).device(device));
  auto expert_offsets =
      torch::zeros({n_experts}, torch::dtype(torch::kInt64).device(device));

  // Launch appropriate kernel based on whether to use parallel scan
  const int threads = 256;
  const int blocks = grid_1d(seq_len, threads);

  if (use_parallel_scan) {
    // For parallel scan, we need additional shared memory for the scan
    // workspace
    int shared_mem_size = calc_shared_memory_size<long>(n_experts * 2);

    // Make sure we have enough threads to perform the scan
    int scan_threads = std::max(threads, n_experts);

    sort_tokens_parallel_scan_kernel<<<blocks, scan_threads, shared_mem_size>>>(
        sorted_indices.data_ptr<long>(), tokens_per_expert.data_ptr<long>(),
        expert_offsets.data_ptr<long>(), topk_ids.data_ptr<long>(), seq_len,
        n_experts, k,
        true // use parallel scan
    );
  } else {
    // Use the original kernel with sequential scan
    int shared_mem_size = calc_shared_memory_size<long>(n_experts);

    sort_tokens_by_expert_kernel<<<blocks, threads, shared_mem_size>>>(
        sorted_indices.data_ptr<long>(), tokens_per_expert.data_ptr<long>(),
        expert_offsets.data_ptr<long>(), topk_ids.data_ptr<long>(), seq_len,
        n_experts, k);
  }

  // Check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    throw std::runtime_error("CUDA kernel execution failed");
  }

  // Create output tensor for sorted tokens
  auto sorted_tokens = torch::empty_like(x);

  // Launch kernel to gather sorted tokens
  dim3 gather_threads(16, 16); // 16x16 = 256 threads per block
  dim3 gather_blocks =
      grid_2d(seq_len, hidden_dim, gather_threads.x, gather_threads.y);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x.scalar_type(), "gather_sorted_tokens_cuda", ([&] {
        gather_sorted_tokens_kernel<scalar_t>
            <<<gather_blocks, gather_threads>>>(
                sorted_tokens.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
                sorted_indices.data_ptr<int>(), seq_len, hidden_dim);
      }));

  // Check for errors
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    throw std::runtime_error("CUDA kernel execution failed");
  }

  return std::make_tuple(sorted_tokens, sorted_indices, tokens_per_expert);
}

//
// Python Bindings
//

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sort_tokens_by_expert", &sort_tokens_by_expert_cuda,
        "Sort tokens by expert assignment with CUDA", py::arg("topk_ids"),
        py::arg("x"), py::arg("n_experts"),
        py::arg("use_parallel_scan") = false);
}
