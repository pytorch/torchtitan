// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

// Smallest problems - use shared memory
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
                          int64_t feature_size, int batch_size) {}
