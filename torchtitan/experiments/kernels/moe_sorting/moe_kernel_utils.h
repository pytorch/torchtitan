// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.

/*
 * Cuda kernel utils file for MoE related kernels
 * basically let's not reinvent the wheel for core functions...
 * ======================
 * cdiv
 * grid_1d
 * grid_2d
 * calc_shared_memory_size
 * =======================

 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace moe_kernel_utils {
/**
 * cdiv - Ceiling division - grid and block size calc support
 *
 * @param numerator Number of elements to process
 * @param denominator Number of elements per thread/block
 * @return Ceiling of the division (usually number of blocks needed)
 */
inline int cdiv(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

/**
 * grid_1d - calculate 1D grid size with upper limit
 *
 * @param elements Number of elements to process
 * @param threads_per_block Number of threads per block
 * @param max_blocks Upper limit of blocks (default to 256 for now)
 * @return optimal number of blocks for the 1d grid
 */
inline int grid_1d(int elements, int threads_per_block, int max_blocks = 256) {
  return std::min(max_blocks, cdiv(elements, threads_per_block));
}

/**
 * grid_2d - calcuate 2d grid based on input dimensions (x,y)
 * @param dim_x  1st dimension size - usually rows
 * @param dim_y  2nd dimension (usually features/columns)
 * @param block_dim_x Number of threads per block in x dimension
 * @param block_dim_y Number of threads per block in y dimension
 * @return dim3 with grid dimensions
 */
inline dim3 grid_2d(int dim_x, int dim_y, int block_dim_x, int block_dim_y) {
  return dim3(cdiv(dim_x, block_dim_x), cdiv(dim_y, block_dim_y));
}

/**
* calc_shared_memory_size - calculate shared memory size needed for given type
and count
*
* @param T Type to allocate for
* @param count Num elements
* @return Size in bytes for shared memory allocation

 */
template <typename T> inline size_t calc_shared_memory_size(int count) {
  return count * sizeof(T);
}
} // namespace moe_kernel_utils
