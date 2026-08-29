/*
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

 This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
*/

// moe_permutation_kernels.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

//////////////////////////////////////////////////////////////////////////////
// CUDA kernels for MoE token permutation
//////////////////////////////////////////////////////////////////////////////

// CUDA kernel to compute prefix sum offsets (scan operation)
__global__ void compute_offsets_kernel(
    int *offsets,                 // output: offsets [n_routed_experts + 1]
    const int *tokens_per_expert, // input: tokens per expert [n_routed_experts]
    int n_routed_experts          // total number of experts
) {
  // This is a simple implementation - for large arrays, a parallel scan would
  // be better
  offsets[0] = 0;
  for (int i = 0; i < n_routed_experts; i++) {
    offsets[i + 1] = offsets[i] + tokens_per_expert[i];
  }
}

// CUDA kernel to calculate expert sizes
__global__ void calculate_expert_sizes_kernel(
    int *m_sizes,                 // output: expert sizes [experts_per_rank]
    int *total_size,              // output: total size (single value)
    const int *tokens_per_expert, // input: tokens per expert [n_routed_experts]
    int n_routed_experts,         // total number of experts
    int experts_per_rank,         // number of experts per rank
    int ep_size,                  // number of expert parallel ranks
    int alignment                 // alignment requirement
) {
  int local_expert_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (local_expert_idx < experts_per_rank) {
    int e_tokens_total = 0;

    // Calculate total tokens for this expert (across all ranks)
    for (int r = 0; r < ep_size; r++) {
      int e_idx = r * experts_per_rank + local_expert_idx;
      if (e_idx < n_routed_experts) {
        e_tokens_total += tokens_per_expert[e_idx];
      }
    }

    // Add padding for alignment
    int padded_size =
        e_tokens_total + (alignment - e_tokens_total % alignment) % alignment;
    m_sizes[local_expert_idx] = padded_size;

    // Atomically add to total size
    atomicAdd(total_size, padded_size);
  }
}

// CUDA kernel to compute permutation indices
__global__ void compute_permutation_indices_kernel(
    int *permuted_indices, // output: permuted indices [total_tokens]
    int *m_sizes,          // output: expert token sizes [experts_per_rank]
    const int *tokens_per_expert, // input: tokens per expert [n_routed_experts]
    const int *offsets, // input: offsets for each expert [n_routed_experts + 1]
    int n_routed_experts, // total number of experts across all ranks
    int experts_per_rank, // number of experts per rank
    int ep_size,          // number of expert parallel ranks
    int alignment,        // alignment requirement (e.g., 128)
    int pad_value         // value to use for padding (-1)
) {
  // Get the local expert ID from thread idx
  int local_expert_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Process only if within valid range
  if (local_expert_idx >= experts_per_rank) {
    return;
  }

  // Step 1: Calculate the output offset for this local expert
  int output_offset = 0;

  // Calculate offsets based on previous experts
  for (int e = 0; e < local_expert_idx; e++) {
    int e_tokens_total = 0;

    // Calculate total tokens for this expert (across all ranks)
    for (int r = 0; r < ep_size; r++) {
      int e_idx = r * experts_per_rank + e;

      // Skip invalid expert indices
      if (e_idx < n_routed_experts) {
        e_tokens_total += tokens_per_expert[e_idx];
      }
    }

    // Add padding for alignment
    int padded_size =
        e_tokens_total + (alignment - e_tokens_total % alignment) % alignment;
    output_offset += padded_size;
  }

  // Step 2: Process tokens for this expert across all remote ranks
  int total_expert_tokens = 0;
  int cur_pos = output_offset;

  // For each remote rank, gather indices
  for (int r = 0; r < ep_size; r++) {
    int expert_idx = r * experts_per_rank + local_expert_idx;

    // Skip if expert index is out of bounds
    if (expert_idx < n_routed_experts) {
      // Get number of tokens for this expert on this rank
      int num_tokens = tokens_per_expert[expert_idx];

      // Get the starting offset for this expert
      int start_offset = offsets[expert_idx];

      // Generate the indices for this expert and write to output
      for (int i = 0; i < num_tokens; i++) {
        permuted_indices[cur_pos + i] = start_offset + i;
      }

      // Update position and count
      cur_pos += num_tokens;
      total_expert_tokens += num_tokens;
    }
  }

  // Step 3: Add padding for alignment
  int padding_needed =
      (alignment - total_expert_tokens % alignment) % alignment;

  // Write padding values (-1) at the end of this expert's section
  for (int i = 0; i < padding_needed; i++) {
    permuted_indices[cur_pos + i] = pad_value;
  }

  // Step 4: Store the padded size for this expert
  m_sizes[local_expert_idx] = total_expert_tokens + padding_needed;
}

// CUDA kernel to apply permutation to tokens
template <typename scalar_t>
__global__ void apply_permutation_kernel(
    scalar_t *output,      // output tensor [total_tokens, hidden_dim]
    const scalar_t *input, // input tensor [total_tokens, hidden_dim]
    const int *indices,    // permutation indices [total_tokens]
    int total_tokens,      // total number of tokens
    int hidden_dim         // hidden dimension size
) {
  // Calculate global thread indices
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // Token index
  int dim = blockIdx.y * blockDim.y + threadIdx.y; // Hidden dimension index

  // Check if we're within valid range
  if (idx < total_tokens && dim < hidden_dim) {
    // Get the permutation index for this token
    int perm_idx = indices[idx];

    // Skip padding tokens (marked with negative indices) or out-of-bounds
    // indices
    if (perm_idx >= 0 && perm_idx < total_tokens) {
      // Copy data from source to destination
      output[idx * hidden_dim + dim] = input[perm_idx * hidden_dim + dim];
    } else {
      // For padding or invalid indices, set output to zero
      output[idx * hidden_dim + dim] = scalar_t(0);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////
// C++/CUDA wrapper functions to launch kernels
//////////////////////////////////////////////////////////////////////////////

// Compute permutation indices for MoE token routing
std::tuple<torch::Tensor, torch::Tensor>
compute_permutation_indices_cuda(torch::Tensor tokens_per_expert_group,
                                 int experts_per_rank, int ep_size,
                                 int alignment = 128, int pad_value = -1) {

  // Ensure input is on CUDA
  TORCH_CHECK(tokens_per_expert_group.device().is_cuda(),
              "tokens_per_expert_group must be a CUDA tensor");
  TORCH_CHECK(tokens_per_expert_group.dim() == 1,
              "tokens_per_expert_group must be a 1D tensor");

  auto device = tokens_per_expert_group.device();
  auto n_routed_experts = tokens_per_expert_group.size(0);

  // 1. Compute offsets (using a separate CUDA kernel)
  auto offsets = torch::zeros({n_routed_experts + 1},
                              torch::dtype(torch::kInt32).device(device));

  // Launch kernel to compute offsets
  compute_offsets_kernel<<<1, 1>>>(offsets.data_ptr<int>(),
                                   tokens_per_expert_group.data_ptr<int>(),
                                   n_routed_experts);

  // 2. Calculate expert sizes on GPU
  auto m_sizes = torch::zeros({experts_per_rank},
                              torch::dtype(torch::kInt32).device(device));

  // Launch a kernel to calculate expert sizes
  const int threads_per_block = 128;
  const int blocks =
      (experts_per_rank + threads_per_block - 1) / threads_per_block;

  // Calculate total size needed
  int total_size = 0;

  // Create a temporary buffer on GPU to hold the total size
  auto d_total_size =
      torch::zeros({1}, torch::dtype(torch::kInt32).device(device));

  // Launch kernel to calculate sizes and total size
  calculate_expert_sizes_kernel<<<blocks, threads_per_block>>>(
      m_sizes.data_ptr<int>(), d_total_size.data_ptr<int>(),
      tokens_per_expert_group.data_ptr<int>(), n_routed_experts,
      experts_per_rank, ep_size, alignment);

  // Copy total size back to host (single value)
  total_size = d_total_size.item<int>();

  // Allocate output tensors with exact size
  auto permuted_indices = torch::full(
      {total_size}, pad_value, torch::dtype(torch::kInt32).device(device));

  // 3. put it all together
  compute_permutation_indices_kernel<<<blocks, threads_per_block>>>(
      permuted_indices.data_ptr<int>(), m_sizes.data_ptr<int>(),
      tokens_per_expert_group.data_ptr<int>(), offsets.data_ptr<int>(),
      n_routed_experts, experts_per_rank, ep_size, alignment, pad_value);

  // Check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    throw std::runtime_error("CUDA kernel execution failed");
  }

  return std::make_tuple(permuted_indices, m_sizes);
}

// Apply permutation to tokens
torch::Tensor apply_permutation_cuda(torch::Tensor token_gather_buf,
                                     torch::Tensor permuted_indices,
                                     torch::IntArrayRef output_shape) {
  // Ensure inputs are on CUDA
  TORCH_CHECK(token_gather_buf.device().is_cuda(),
              "token_gather_buf must be a CUDA tensor");
  TORCH_CHECK(permuted_indices.device().is_cuda(),
              "permuted_indices must be a CUDA tensor");

  auto device = token_gather_buf.device();
  auto total_tokens = token_gather_buf.size(0);
  auto hidden_dim = token_gather_buf.size(1);

  // Create output tensor with the same shape and dtype
  auto permuted_tokens = torch::zeros(
      output_shape, torch::dtype(token_gather_buf.dtype()).device(device));

  // Determine grid and block sizes for optimal performance
  dim3 threads_per_block(16, 16); // 16x16 = 256 threads per block
  dim3 blocks((total_tokens + threads_per_block.x - 1) / threads_per_block.x,
              (hidden_dim + threads_per_block.y - 1) / threads_per_block.y);

  // Launch kernel with the appropriate scalar type
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      token_gather_buf.scalar_type(), "apply_permutation_cuda", ([&] {
        apply_permutation_kernel<scalar_t><<<blocks, threads_per_block>>>(
            permuted_tokens.data_ptr<scalar_t>(),
            token_gather_buf.data_ptr<scalar_t>(),
            permuted_indices.data_ptr<int>(), total_tokens, hidden_dim);
      }));

  // Check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    throw std::runtime_error("CUDA kernel execution failed");
  }

  return permuted_tokens;
}

// Optimized token permutation for MoE routing using CUDA
std::tuple<torch::Tensor, torch::Tensor> optimized_token_permutation_cuda(
    torch::Tensor token_gather_buf, torch::Tensor tokens_per_expert_group,
    int experts_per_rank, int ep_size, int alignment = 128) {
  // Step 1: Compute permutation indices
  auto result = compute_permutation_indices_cuda(
      tokens_per_expert_group, experts_per_rank, ep_size, alignment, -1);

  torch::Tensor permuted_indices = std::get<0>(result);
  torch::Tensor m_sizes = std::get<1>(result);

  // Step 2: Apply permutation
  torch::Tensor permuted_tokens = apply_permutation_cuda(
      token_gather_buf, permuted_indices, token_gather_buf.sizes());

  return std::make_tuple(permuted_tokens, m_sizes);
}

//////////////////////////////////////////////////////////////////////////////
// Python bindings
//////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_permutation_indices", &compute_permutation_indices_cuda,
        "Compute permutation indices for MoE token routing (CUDA)",
        py::arg("tokens_per_expert_group"), py::arg("experts_per_rank"),
        py::arg("ep_size"), py::arg("alignment") = 128,
        py::arg("pad_value") = -1);

  m.def("apply_permutation", &apply_permutation_cuda,
        "Apply permutation to tokens (CUDA)", py::arg("token_gather_buf"),
        py::arg("permuted_indices"), py::arg("output_shape"));

  m.def("optimized_token_permutation", &optimized_token_permutation_cuda,
        "Optimized token permutation for MoE routing (CUDA)",
        py::arg("token_gather_buf"), py::arg("tokens_per_expert_group"),
        py::arg("experts_per_rank"), py::arg("ep_size"),
        py::arg("alignment") = 128);
}
