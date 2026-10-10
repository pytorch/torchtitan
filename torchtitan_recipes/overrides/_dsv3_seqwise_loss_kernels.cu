// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// DSv3 microbatch-wise load-balance loss for FP32 scores[4096,256].
// Routing assignments are supplied by the router. Floating-point operations
// follow MicrobatchWiseLoadBalanceLoss and its autograd graph in this order:
//
//   row L1 norm / sum256   reduce_kernel<512,1>: one warp per row, vec4 loads,
//                          four accumulators each adding x[e] and x[e + 128],
//                          sequential accumulator combine, shuffle-down tree.
//   token sum4096          reduce_kernel<128,4>: block (32,4), 64 CTAs per
//                          output, vt0=4 accumulators over a 256-token stride,
//                          y-tree (0+2, 1+3) and a 64-CTA global combine.
//   scalar sum256          one CTA of 64 threads, vec4, then a 64->1 tree.
//
// The forward is one launch of 64 eight-CTA clusters, one per token partition.
// Each cluster combines its partition through distributed shared memory; the
// last cluster to finish (an integer arrival ticket) performs the global
// combine. No floating-point atomics, scratch memset, or second kernel.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <tuple>

namespace {
namespace cg = cooperative_groups;
constexpr unsigned kWarpMask = 0xffffffffu;
constexpr int kExperts = 256;
constexpr int kTokens = 4096;
constexpr int kPartitions = 64;   // ATen CTAs per output in the token reduction.
constexpr int kSlots = 4;         // ATen vt0 accumulators.
constexpr int kRows = 4;          // ATen blockDim.y in the token reduction.
constexpr int kSteps = 4;         // Tokens per accumulator slot.
constexpr int kClusterCtas = 8;   // CTAs per token partition (one cluster).
constexpr int kWarpsPerCta = 8;   // Two accumulator slots of one ATen row.
constexpr int kSliceExperts = kExperts / kClusterCtas;
constexpr int kForwardThreads = 32 * kWarpsPerCta;
constexpr int kBackwardThreads = 256;
constexpr float kNormEpsilon = 1.0e-12f;

__device__ __forceinline__ float warp_sum(float value) {
  #pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    value = __fadd_rn(value, __shfl_down_sync(kWarpMask, value, offset));
  }
  return __shfl_sync(kWarpMask, value, 0);
}

// Lane l owns experts 4l..4l+3 (items 0-3) and 128+4l..128+4l+3 (items 4-7).
// Expert indices increase with the item index inside a lane.
__device__ __forceinline__ int expert_of(int lane, int item) {
  return lane * 4 + (item & 3) + (item >= 4 ? 128 : 0);
}

// Native reduce_kernel<512,1> order for a 256-wide row held as items[8].
__device__ __forceinline__ float row_sum(const float (&items)[8]) {
  float value = __fadd_rn(items[0], items[4]);
  #pragma unroll
  for (int item = 1; item < 4; ++item) {
    value = __fadd_rn(value, __fadd_rn(items[item], items[item + 4]));
  }
  return warp_sum(value);
}

__device__ __forceinline__ void load_row(const float* row, int lane, float (&items)[8]) {
  const float4 first = *reinterpret_cast<const float4*>(row + lane * 4);
  const float4 second = *reinterpret_cast<const float4*>(row + 128 + lane * 4);
  items[0] = first.x; items[1] = first.y; items[2] = first.z; items[3] = first.w;
  items[4] = second.x; items[5] = second.y; items[6] = second.z; items[7] = second.w;
}

// clamp_min(norm, eps) propagates NaN like ATen's clamp kernel.
__device__ __forceinline__ float clamp_norm(float norm) {
  return isnan(norm) ? norm : fmaxf(norm, kNormEpsilon);
}

// IEEE division by a row-constant divisor. This is the hardware div.rn.f32
// fast path with its divisor-only reciprocal refinement hoisted. Within the
// range below every operand is normal and the quotient cannot over- or
// underflow, where the sequence is correctly rounded; the caller falls back to
// __fdiv_rn for any other operand, so results equal __fdiv_rn bit for bit.
struct RowDivisor {
  float divisor;
  float reciprocal;
  bool fast;
};

__device__ __forceinline__ bool in_fast_range(float value) {
  const float magnitude = fabsf(value);
  return magnitude >= 0x1p-60f && magnitude < 0x1p60f;
}

__device__ __forceinline__ RowDivisor make_divisor(float divisor) {
  float approximate;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(approximate) : "f"(divisor));
  return {divisor, fmaf(approximate, fmaf(-divisor, approximate, 1.0f), approximate),
      in_fast_range(divisor)};
}

// A signed-zero numerator yields the exactly signed product 0 * reciprocal.
__device__ __forceinline__ float fast_divide(float numerator, const RowDivisor& d) {
  const float quotient = __fmul_rn(numerator, d.reciprocal);
  const float corrected = fmaf(d.reciprocal, fmaf(-d.divisor, quotient, numerator), quotient);
  return numerator == 0.0f ? quotient : corrected;
}

// Divides items[8] by one divisor, warp-uniformly falling back to IEEE division.
__device__ __forceinline__ void divide_row(
    const float (&numerators)[8], const RowDivisor& d, float (&quotients)[8]) {
  bool fast = d.fast;
  #pragma unroll
  for (int item = 0; item < 8; ++item) {
    fast &= in_fast_range(numerators[item]) || numerators[item] == 0.0f;
    quotients[item] = fast_divide(numerators[item], d);
  }
  if (!__all_sync(kWarpMask, fast)) {
    #pragma unroll
    for (int item = 0; item < 8; ++item) {
      quotients[item] = __fdiv_rn(numerators[item], d.divisor);
    }
  }
}

// Grid: 64 clusters (token partitions) x 8 CTAs. CTA rank r of partition p
// owns ATen row y = r / 2 and accumulator slots 2(r % 2) and 2(r % 2) + 1.
// Warp w handles slot 2(r % 2) + w / 4, step i = w % 4: token
// p*4 + y + (slot + 4*i)*256. Distributed shared memory combines the
// partition; the last partition's cluster performs the global combine.
__global__ void __cluster_dims__(kClusterCtas, 1, 1) __launch_bounds__(kForwardThreads)
seqwise_forward_kernel(
    const float* __restrict__ scores,
    const bool* __restrict__ routing_map,
    float* __restrict__ raw_sum,
    float* __restrict__ frequencies_output,
    float* __restrict__ partition_sums,           // [64][256]
    unsigned char* __restrict__ partition_counts,  // [64][256]
    unsigned* __restrict__ arrivals) {            // persistent
  const auto cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int partition = blockIdx.x / kClusterCtas;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = rank >> 1;
  __shared__ __align__(16) float normalized_rows[kWarpsPerCta][kExperts];
  __shared__ unsigned selected_masks[kWarpsPerCta][32];
  __shared__ float slot_sums[2][kExperts];
  __shared__ unsigned slot_counts[2][kExperts / 4];
  __shared__ float row_values[kRows][kSliceExperts];
  __shared__ int row_counts[kRows][kSliceExperts];
  __shared__ float products[kExperts];
  __shared__ float counts[kExperts];
  __shared__ float count_denominator;
  __shared__ float partial[64];
  __shared__ int is_last;

  {
    const int slot = 2 * (rank & 1) + (warp >> 2);
    const int token = partition * kRows + row + (slot + (warp & 3) * kSlots) * 256;
    float items[8];
    load_row(scores + token * kExperts, lane, items);
    float magnitudes[8];
    #pragma unroll
    for (int item = 0; item < 8; ++item) magnitudes[item] = fabsf(items[item]);
    const RowDivisor divisor = make_divisor(clamp_norm(row_sum(magnitudes)));
    float normalized[8];
    divide_row(items, divisor, normalized);
    *reinterpret_cast<float4*>(&normalized_rows[warp][lane * 4]) =
        make_float4(normalized[0], normalized[1], normalized[2], normalized[3]);
    *reinterpret_cast<float4*>(&normalized_rows[warp][128 + lane * 4]) =
        make_float4(normalized[4], normalized[5], normalized[6], normalized[7]);
    unsigned selected = 0;
    #pragma unroll
    for (int item = 0; item < 8; ++item) {
      selected |= unsigned(routing_map[token * kExperts + expert_of(lane, item)]) << item;
    }
    selected_masks[warp][lane] = selected;
  }
  __syncthreads();
  // Accumulator slot of ATen row y: ((((0 + n0) + n1) + n2) + n3).
  #pragma unroll
  for (int local = 0; local < 2; ++local) {
    float value = 0.0f;
    #pragma unroll
    for (int step = 0; step < kSteps; ++step) {
      value = __fadd_rn(value, normalized_rows[local * kSteps + step][threadIdx.x]);
    }
    slot_sums[local][threadIdx.x] = value;
  }
  if (threadIdx.x < 128) {
    // Word w packs experts 4w..4w+3 (lane w % 32, items 4(w / 32)..+3).
    const int local = threadIdx.x >> 6;
    const int word_index = threadIdx.x & 63;
    const int owner = word_index & 31;
    const int shift = (word_index >> 5) * 4;
    unsigned word = 0;
    #pragma unroll
    for (int step = 0; step < kSteps; ++step) {
      const unsigned bits = selected_masks[local * kSteps + step][owner] >> shift;
      word += (bits & 1u) | ((bits >> 1 & 1u) << 8) | ((bits >> 2 & 1u) << 16) |
          ((bits >> 3 & 1u) << 24);
    }
    slot_counts[local][word_index] = word;
  }
  cluster.sync();

  // Partition level for this CTA's 32-expert slice: per ATen row y, slots
  // 0..3 in order, then the y-tree (0+2, 1+3).
  const int slice_lane = threadIdx.x & (kSliceExperts - 1);
  const int expert = rank * kSliceExperts + slice_lane;
  if (threadIdx.x < kRows * kSliceExperts) {
    const int y = threadIdx.x / kSliceExperts;
    float value = 0.0f;
    int count = 0;
    #pragma unroll
    for (int slot = 0; slot < kSlots; ++slot) {
      const int source = y * 2 + (slot >> 1);
      const float* remote_sums = cluster.map_shared_rank(&slot_sums[slot & 1][0], source);
      const unsigned* remote_counts = cluster.map_shared_rank(&slot_counts[slot & 1][0], source);
      const float term = remote_sums[expert];
      value = slot == 0 ? term : __fadd_rn(value, term);
      count += (remote_counts[expert >> 2] >> ((expert & 3) * 8)) & 0xff;
    }
    row_values[y][slice_lane] = value;
    row_counts[y][slice_lane] = count;
  }
  __syncthreads();
  if (threadIdx.x < kSliceExperts) {
    partition_sums[partition * kExperts + expert] = __fadd_rn(
        __fadd_rn(row_values[0][slice_lane], row_values[2][slice_lane]),
        __fadd_rn(row_values[1][slice_lane], row_values[3][slice_lane]));
    partition_counts[partition * kExperts + expert] = static_cast<unsigned char>(
        row_counts[0][slice_lane] + row_counts[1][slice_lane] +
        row_counts[2][slice_lane] + row_counts[3][slice_lane]);
  }
  __syncthreads();
  if (threadIdx.x == 0) __threadfence();
  cluster.sync();  // Also retires every remote read of this CTA's slots.
  if (rank == 0 && threadIdx.x == 0) {
    // Modulo arrival counting needs no reset; 2^32 is a multiple of 64.
    const int last = (atomicAdd(arrivals, 1u) % kPartitions) == kPartitions - 1;
    if (last) __threadfence();
    for (int target = 0; target < kClusterCtas; ++target) {
      *cluster.map_shared_rank(&is_last, target) = last;
    }
  }
  cluster.sync();
  if (!is_last) return;
  __threadfence();

  // Global level for this slice: ATen row y combines partitions y, y+4, ...
  // from zero, then the y-tree; counts are exact integer sums.
  if (threadIdx.x < kRows * kSliceExperts) {
    const int y = threadIdx.x / kSliceExperts;
    float values[16];
    unsigned char bytes[16];
    #pragma unroll
    for (int index = 0; index < 16; ++index) {
      const int source = (y + index * kRows) * kExperts + expert;
      values[index] = __ldcg(partition_sums + source);
      bytes[index] = __ldcg(partition_counts + source);
    }
    float value = 0.0f;
    int count = 0;
    #pragma unroll
    for (int index = 0; index < 16; ++index) {
      value = __fadd_rn(value, values[index]);
      count += bytes[index];
    }
    row_values[y][slice_lane] = value;
    row_counts[y][slice_lane] = count;
  }
  __syncthreads();
  if (threadIdx.x < kSliceExperts) {
    const float probability_sum = __fadd_rn(
        __fadd_rn(row_values[0][slice_lane], row_values[2][slice_lane]),
        __fadd_rn(row_values[1][slice_lane], row_values[3][slice_lane]));
    const float count_value = static_cast<float>(row_counts[0][slice_lane] +
        row_counts[1][slice_lane] + row_counts[2][slice_lane] + row_counts[3][slice_lane]);
    *cluster.map_shared_rank(&counts[expert], 0) = count_value;
    *cluster.map_shared_rank(&products[expert], 0) = probability_sum;
  }
  cluster.sync();
  if (rank != 0) return;
  if (threadIdx.x < 32) {
    float count_items[8];
    #pragma unroll
    for (int item = 0; item < 8; ++item) count_items[item] = counts[expert_of(lane, item)];
    const float total = row_sum(count_items);
    if (lane == 0) count_denominator = clamp_norm(total);
  }
  __syncthreads();
  if (threadIdx.x < kExperts) {
    const float frequency = __fmul_rn(
        __fdiv_rn(counts[threadIdx.x], count_denominator), float(kExperts));
    frequencies_output[threadIdx.x] = frequency;
    products[threadIdx.x] = __fmul_rn(frequency, products[threadIdx.x]);
  }
  __syncthreads();
  if (threadIdx.x < 64) {
    float value = products[threadIdx.x * 4];
    #pragma unroll
    for (int item = 1; item < 4; ++item) {
      value = __fadd_rn(value, products[threadIdx.x * 4 + item]);
    }
    partial[threadIdx.x] = value;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    const float value = warp_sum(__fadd_rn(partial[lane], partial[lane + 32]));
    if (lane == 0) *raw_sum = value;
  }
}

// One warp per token. The native graph computes, for g = d(raw_sum):
//   c_e    = g * frequency_e                         (Mul backward)
//   direct = c_e / denom                             (div self backward)
//   q_e    = -c_e * ((x_e / denom) / denom)          (div other backward)
//   norm   = where(norm > eps, sum256(q), 0)         (expand, clamp_min)
//   grad_e = direct + sgn(x_e) * norm                (norm backward, add)
__global__ void __launch_bounds__(kBackwardThreads) seqwise_backward_kernel(
    const float* __restrict__ grad_raw_sum,
    const float* __restrict__ scores,
    const float* __restrict__ frequencies,
    float* __restrict__ grad_scores) {
  __shared__ float coefficients[kExperts];
  const int lane = threadIdx.x & 31;
  const int token = blockIdx.x * (kBackwardThreads / 32) + (threadIdx.x >> 5);
  float items[8];
  load_row(scores + token * kExperts, lane, items);
  const float upstream = *grad_raw_sum;
  for (int expert = threadIdx.x; expert < kExperts; expert += kBackwardThreads) {
    coefficients[expert] = __fmul_rn(upstream, frequencies[expert]);
  }
  float magnitudes[8];
  #pragma unroll
  for (int item = 0; item < 8; ++item) magnitudes[item] = fabsf(items[item]);
  const float norm = row_sum(magnitudes);
  const RowDivisor divisor = make_divisor(clamp_norm(norm));
  float normalized[8];
  divide_row(items, divisor, normalized);
  float quotients[8];
  divide_row(normalized, divisor, quotients);
  __syncthreads();
  float coefficient[8];
  #pragma unroll
  for (int item = 0; item < 8; ++item) coefficient[item] = coefficients[expert_of(lane, item)];
  float direct[8];
  divide_row(coefficient, divisor, direct);
  float denominator_terms[8];
  #pragma unroll
  for (int item = 0; item < 8; ++item) {
    denominator_terms[item] = __fmul_rn(-coefficient[item], quotients[item]);
  }
  float norm_gradient = row_sum(denominator_terms);
  norm_gradient = norm > kNormEpsilon ? norm_gradient : 0.0f;
  float output[8];
  #pragma unroll
  for (int item = 0; item < 8; ++item) {
    const float sign = static_cast<float>((0.0f < items[item]) - (items[item] < 0.0f));
    output[item] = __fadd_rn(direct[item], __fmul_rn(sign, norm_gradient));
  }
  float* row = grad_scores + token * kExperts;
  *reinterpret_cast<float4*>(row + lane * 4) =
      make_float4(output[0], output[1], output[2], output[3]);
  *reinterpret_cast<float4*>(row + 128 + lane * 4) =
      make_float4(output[4], output[5], output[6], output[7]);
}

at::Tensor empty_output(at::IntArrayRef size, const at::TensorOptions& options) {
  // Every element is written before use; skip deterministic fill poisoning.
  return at::Tensor(at::detail::empty_cuda(size, options));
}

void check_scores(const at::Tensor& scores) {
  TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == at::kFloat &&
      scores.is_contiguous() && !scores.is_neg() && !scores.is_conj() && scores.sizes() == at::IntArrayRef({kTokens, kExperts}) &&
      (reinterpret_cast<uintptr_t>(scores.data_ptr()) & 15u) == 0,
      "seqwise load-balance loss requires 16-byte aligned contiguous CUDA FP32 scores[4096,256]");
}
} // namespace

std::tuple<at::Tensor, at::Tensor> seqwise_load_balance_forward_cuda(
    const at::Tensor& scores, const at::Tensor& routing_map, const at::Tensor& arrivals) {
  check_scores(scores);
  TORCH_CHECK(routing_map.device() == scores.device() && routing_map.scalar_type() == at::kBool &&
      routing_map.is_contiguous() && routing_map.sizes() == scores.sizes(),
      "load-balance loss requires a contiguous boolean routing map matching scores");
  TORCH_CHECK(arrivals.device() == scores.device() && arrivals.scalar_type() == at::kInt &&
      arrivals.is_contiguous() && arrivals.numel() == 1,
      "seqwise load-balance loss requires its persistent arrival counter");
  const c10::cuda::CUDAGuard guard(scores.device());
  const auto options = scores.options();
  auto raw_sum = empty_output({}, options);
  auto frequencies = empty_output({kExperts}, options);
  // Partition sums then byte counts; every consumed entry is overwritten.
  constexpr int64_t kPartitionWords = kPartitions * kExperts;
  auto workspace = empty_output({kPartitionWords + kPartitionWords / 4}, options);
  float* base = workspace.data_ptr<float>();
  seqwise_forward_kernel<<<kPartitions * kClusterCtas, kForwardThreads, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      scores.data_ptr<float>(), routing_map.data_ptr<bool>(), raw_sum.data_ptr<float>(),
      frequencies.data_ptr<float>(),
      base, reinterpret_cast<unsigned char*>(base + kPartitionWords),
      reinterpret_cast<unsigned*>(arrivals.data_ptr<int>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {raw_sum, frequencies};
}

at::Tensor seqwise_load_balance_backward_cuda(
    const at::Tensor& grad_raw_sum, const at::Tensor& scores, const at::Tensor& frequencies) {
  check_scores(scores);
  TORCH_CHECK(grad_raw_sum.device() == scores.device() &&
      grad_raw_sum.scalar_type() == at::kFloat && grad_raw_sum.numel() == 1 &&
      !grad_raw_sum.is_neg(),
      "seqwise load-balance backward requires an FP32 scalar gradient");
  TORCH_CHECK(frequencies.device() == scores.device() && frequencies.scalar_type() == at::kFloat &&
      frequencies.is_contiguous() && frequencies.numel() == kExperts,
      "seqwise load-balance backward requires FP32 frequencies[256]");
  const c10::cuda::CUDAGuard guard(scores.device());
  auto grad_scores = empty_output(scores.sizes(), scores.options());
  seqwise_backward_kernel<<<kTokens / (kBackwardThreads / 32), kBackwardThreads, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      grad_raw_sum.data_ptr<float>(), scores.data_ptr<float>(), frequencies.data_ptr<float>(),
      grad_scores.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_scores;
}
