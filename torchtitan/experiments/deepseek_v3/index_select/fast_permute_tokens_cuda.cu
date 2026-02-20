// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

// Smallest problems - use shared memory as bridge
template <typename scalar_t, int BLOCK_SIZE, int FEATURES_PER_THREAD>
__global__ void
fast_permute_small_kernel(const scalar_t *__restrict__ input,
                          const int64_t *__restrict__ permute_indices,
                          scalar_t *__restrict__ output, int64_t num_indices,
                          int64_t feature_size) {
  // smem buffer for features of a single token
  extern __shared__ char shared_mem[];
  scalar_t *shared_features = reinterpret_cast<scalar_t *>(shared_mem);

  // each block will handle a single token
  const int token_idx = blockIdx.x;
  const int thread_idx = threadIdx.x;

  // if in bounds (effectively M)
  if (token_idx < num_indices) {
    // load source token
    const int64_t src_idx = permute_indices[token_idx];

    // each thread loads features into smem
    for (int i = thread_idx; i < feature_size; i += BLOCK_SIZE) {
      shared_features[i] = input[src_idx * feature_size + i];
    }

    // wait for everyone to load...
    __syncthreads();

    // each thread writes features to output
    for (int i = thread_idx; i < feature_size; i += BLOCK_SIZE) {
      output[token_idx * feature_size + i] = shared_features[i];
    }
  }
}

// medium kernel - multiple tokens per block
template <typename scalar_t, int TOKENS_PER_BLOCK, int THREADS_PER_BLOCK>
__global__ void
fast_permute_medium_kernel(const scalar_t *__restrict__ input,
                           const int64_t *__restrict__ permute_indices,
                           scalar_t *__restrict__ output, int64_t num_indices,
                           int64_t feature_size) {
  // Here each block handles multiple tokens
  const int token_start = blockIdx.x * TOKENS_PER_BLOCK;
  const int thread_idx = threadIdx.x;

  // threads per token
  const int threads_per_token = THREADS_PER_BLOCK / TOKENS_PER_BLOCK;
  // token and thread offset
  const int local_token_idx = thread_idx / threads_per_token;
  const int token_thread_idx = thread_idx % threads_per_token;

  // Global token index
  const int token_idx = token_start + local_token_idx;

  if (token_idx < num_indices) {
    // load source index
    const int64_t src_idx = permute_indices[token_idx];

    // just like in small, each thread copies subset of features
    for (int i = token_thread_idx; i < feature_size; i += threads_per_token) {
      output[token_idx * feature_size + i] = input[src_idx * feature_size + i];
    }
  }
}

// large kernel - each block processes multiple rows and columns
template <typename scalar_t>
__global__ void
fast_permute_large_kernel(const scalar_t *__restrict__ input,
                          const int64_t *__restrict__ permute_indices,
                          scalar_t *__restrict__ output, int64_t num_indices,
                          int64_t feature_size, int batch_size) {

  // setup 2D grid of blocks
  // each handles a portion of indiees and features
  const int block_row = blockIdx.y;
  const int block_col = blockIdx.x;
  const int thread_idx = threadIdx.x;

  const int BLOCK_SIZE = 256; // 512?
  const int FEATURES_PER_BLOCK = 1024;
  const int INDICES_PER_BLOCK = 16;

  // stread index for this block
  const int start_idx = block_row * INDICES_PER_BLOCK;

  const int start_feature = block_col * FEATURES_PER_BLOCK;

  // calc num features to process for this block
  const int num_features =
      min(FEATURES_PER_BLOCK, static_cast<int>(feature_size - start_feature));

  // each thread handles multiple features (cdiv...)
  const int features_per_thread = (num_features + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // process things
  for (int i = 0; i < INDICES_PER_BLOCK; ++i) {
    const int token_idx = start_idx + i;
    // load source
    const int64_t src_idx = permute_indices[token_idx];

    // each thread processes features
    for (int j = 0; j < features_per_thread; ++j) {
      const int feature_idx = start_feature + thread_idx + j * BLOCK_SIZE;
      if (feature_idx < feature_size) {
        output[token_idx * feature_size + feature_idx] =
            input[src_idx * feature_size + feature_idx];
      }
    }
  }
}

// XL kernel - adds vectorized loads - not sure where the tradeoff is for this
template <typename scalar_t>
__global__ void
fast_permute_vectorized_kernel(const scalar_t *__restrict__ input,
                               const int64_t *__restrict__ permute_indices,
                               scalar_t *__restrict__ output,
                               int64_t num_indices, int64_t feature_size

) {
  // templatized vec4

  using vec4_t = typename std::aligned_storage<4 * sizeof(scalar_t),
                                               4 * sizeof(scalar_t)>::type;

  // for loop vs tail
  const int vec_width = 4;

  const int token_idx = blockIdx.x;
  const int thread_idx = threadIdx.x;

  if (token_idx < num_indices) {
    // load soarce
    const int64_t src_idx = permute_indices[token_idx];

    // base pointers
    const scalar_t *src_ptr = input + src_idx * feature_size;
    scalar_t *dst_ptr = output + token_idx * feature_size;

    // calc number of aligned vec4 elements

    const int num_vectors = feature_size / vec_width;

    // process aligned elements
    for (int i = thread_idx; i < num_vectors; i += blockDim.x) {
      reinterpret_cast<vec4_t *>(dst_ptr)[i] =
          reinterpret_cast<const vec4_t *>(src_ptr)[i];

      // remainder (tail)
      const int remaining_start = num_vectors * vec_width;
      for (int i = remaining_start + thread_idx; i < feature_size;
           i += blockDim.x) {
        dst_ptr[i] = src_ptr[i];
      }
    }

    //
  }
}

// Launch the best kernel based on problem size (effectively adaptive tuning)
torch::Tensor adaptive_fast_permute(torch::Tensor input,
                                    torch::Tensor permute_indices) {

  // INput checks
  TORCH_CHECK(input.dim() == 2,
              "Input tensor must be 2D [num_tokens, hidden_dim]");
  TORCH_CHECK(permute_indices.dim() == 1, "Permute indices tensor must be 1D");
  TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(permute_indices.device().is_cuda(),
              "Permute indices must be a CUDA tensor");

  const int64_t batch_size = input.size(0);
  const int64_t feature_size = input.size(1);
  const int64_t num_indices = permute_indices.size(0);

  auto output = torch::empty({num_indices, feature_size}, input.options());

  const int threads_count = 512;

  // auto select kernel - criterion is not finalized
  if (num_indices <= 128 && feature_size <= 4096) {
    // small, use shared mem kernel
    const int threads = threads_count;
    const int features_per_thread = 4;
    const int blocks = num_indices;
    const int shared_mem_size = feature_size * sizeof(float); // half?

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "fast_permute_msall", ([&] {
          fast_permute_small_kernel<scalar_t, threads, features_per_thread>
              <<<blocks, threads, shared_mem_size>>>(
                  input.data_ptr<scalar_t>(),
                  permute_indices.data_ptr<int64_t>(),
                  output.data_ptr<scalar_t>(), num_indices, feature_size);
        }));
  } else if (num_indices <= 2048 && feature_size <= 4096) {
    // call this medium, process multiple tokens per block
    const int tokens_per_block = 4;
    const int threads_per_block = threads_count;
    const int blocks = (num_indices + tokens_per_block - 1) / tokens_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "fast_permute_medium", ([&] {
          fast_permute_medium_kernel<scalar_t, tokens_per_block,
                                     threads_per_block>
              <<<blocks, threads_per_block>>>(
                  input.data_ptr<scalar_t>(),
                  permute_indices.data_ptr<int64_t>(),
                  output.data_ptr<scalar_t>(), num_indices, feature_size);
        }));
  } else if (feature_size >= 8192 || num_indices >= 8192) {
    // XL problem - use vectorized loads
    const int threads = threads_count;
    const int blocks = num_indices;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "fast_permute_vectorized", ([&] {
          fast_permute_vectorized_kernel<scalar_t><<<blocks, threads>>>(
              input.data_ptr<scalar_t>(), permute_indices.data_ptr<int64_t>(),
              output.data_ptr<scalar_t>(), num_indices, feature_size);
        }));

  } else {
    // Large - use 2D grid
    const int features_per_block = 1024;
    const int indices_per_block = 16;
    const int threads = threads_count;

    const int grid_x =
        (feature_size + features_per_block - 1) / features_per_block;
    const int grid_y =
        (num_indices + indices_per_block - 1) / indices_per_block;

    dim3 blocks(grid_x, grid_y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "fast_permute_large", ([&] {
          fast_permute_large_kernel<scalar_t><<<blocks, threads>>>(
              input.data_ptr<scalar_t>(), permute_indices.data_ptr<int64_t>(),
              output.data_ptr<scalar_t>(), num_indices, feature_size,
              batch_size);
        }));
  }
  // finished
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fast_permute_tokens", &adaptive_fast_permute,
        "Adaptive Fast Permute Implementation for MoE using CUDA",
        py::arg("input"), py::arg("permute_indices"));
}
