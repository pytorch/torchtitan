// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <optional>
#include <vector>

namespace {
namespace cg = cooperative_groups;
constexpr unsigned kWarpMask = 0xffffffffu;
constexpr int kExperts = 256;
constexpr int kRows = 4096;
constexpr int kPartitions = 64;
constexpr int kThreads = 512;

__device__ __forceinline__ float sigmoid_native(float value) {
  return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-value)));
}

__device__ __forceinline__ float4 add_vectors(float4 first, float4 second) {
  return make_float4(__fadd_rn(first.x, second.x), __fadd_rn(first.y, second.y),
      __fadd_rn(first.z, second.z), __fadd_rn(first.w, second.w));
}

__device__ __forceinline__ unsigned pack_counts(int first, int second, int third, int fourth) {
  return unsigned(first) | (unsigned(second) << 8) |
      (unsigned(third) << 16) | (unsigned(fourth) << 24);
}

__device__ __forceinline__ float warp_sum(float value) {
  #pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    value = __fadd_rn(value, __shfl_down_sync(kWarpMask, value, offset));
  }
  return __shfl_sync(kWarpMask, value, 0);
}

__device__ __forceinline__ unsigned ordering_key(float value) {
  unsigned bits = __float_as_uint(value);
  // ATen's radix-selection conversion places every NaN above finite values.
  if ((bits & 0x7fffffffu) > 0x7f800000u) return 0xffffffffu;
  return bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

__device__ __forceinline__ unsigned select_eight(
    const float (&scores)[8], int (&selected)[8], int lane,
    int& choice_expert, unsigned& choice_key) {
  unsigned keys[8];
  #pragma unroll
  for (int item = 0; item < 8; ++item) {
    keys[item] = ordering_key(scores[item]);
    selected[item] = 0;
  }
  unsigned threshold = 0;
  #pragma unroll 1
  for (int choice = 0; choice < 8; ++choice) {
    unsigned best_key = 0;
    #pragma unroll
    for (int item = 0; item < 8; ++item) best_key = max(best_key, keys[item]);
    best_key = __reduce_max_sync(kWarpMask, best_key);
    threshold = best_key;
    int best_expert = kExperts;
    #pragma unroll
    for (int item = 0; item < 8; ++item) {
      int expert = lane * 4 + (item & 3) + (item >= 4 ? 128 : 0);
      if (keys[item] == best_key) best_expert = min(best_expert, expert);
    }
    best_expert = __reduce_min_sync(kWarpMask, best_expert);
    if (lane == choice) {
      choice_expert = best_expert;
      choice_key = best_key;
    }
    #pragma unroll
    for (int item = 0; item < 8; ++item) {
      int expert = lane * 4 + (item & 3) + (item >= 4 ? 128 : 0);
      if (expert == best_expert) {
        selected[item] = 1;
        keys[item] = 0;
      }
    }
  }
  return threshold;
}

__device__ __forceinline__ float key_value(unsigned key) {
  return __uint_as_float(key ^ ((key & 0x80000000u) ? 0x80000000u : 0xffffffffu));
}

__device__ __forceinline__ float group_top_two(
    const float (&scores)[8], int half, int lane) {
  const unsigned group_mask = 0xffu << (lane & ~7);
  unsigned keys[4];
  #pragma unroll
  for (int item = 0; item < 4; ++item) keys[item] = ordering_key(scores[half * 4 + item]);
  unsigned top_keys[2];
  #pragma unroll
  for (int choice = 0; choice < 2; ++choice) {
    unsigned best_key = max(max(keys[0], keys[1]), max(keys[2], keys[3]));
    best_key = __reduce_max_sync(group_mask, best_key);
    int best_expert = kExperts;
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
      if (keys[item] == best_key) best_expert = min(best_expert, lane * 4 + item);
    }
    best_expert = __reduce_min_sync(group_mask, best_expert);
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
      if (lane * 4 + item == best_expert) keys[item] = 0;
    }
    top_keys[choice] = best_key;
  }
  // Native sum2 starts from +0, which matters when both selected values are -0.
  return __fadd_rn(__fadd_rn(0.0f, key_value(top_keys[0])), key_value(top_keys[1]));
}

__device__ __forceinline__ unsigned select_groups(
    const float (&scores)[8], int lane) {
  const float first = group_top_two(scores, 0, lane);
  const float second = group_top_two(scores, 1, lane);
  const float first_group = __shfl_sync(kWarpMask, first, (lane & 3) * 8);
  const float second_group = __shfl_sync(kWarpMask, second, (lane & 3) * 8);
  unsigned key = lane < 8 ? ordering_key(lane < 4 ? first_group : second_group) : 0;
  unsigned selected_groups = 0;
  #pragma unroll
  for (int choice = 0; choice < 4; ++choice) {
    const unsigned best_key = __reduce_max_sync(kWarpMask, key);
    const int best_group = __reduce_min_sync(kWarpMask, key == best_key ? lane : 32);
    selected_groups |= 1u << best_group;
    if (lane == best_group) key = 0;
  }
  return selected_groups;
}

__device__ __forceinline__ int native_order(
    int expert, unsigned choice_key, unsigned threshold, int lane) {
  // Native unsorted topk emits indices above the threshold first in ascending
  // index order, followed by threshold ties. Sort only the eight selected IDs.
  int order_key = expert + (choice_key == threshold ? kExperts : 0);
  #pragma unroll
  for (int group = 2; group <= 8; group <<= 1) {
    #pragma unroll
    for (int distance = group >> 1; distance; distance >>= 1) {
      const int peer = __shfl_xor_sync(0xffu, order_key, distance, 8);
      const bool take_minimum = ((lane & group) == 0) == ((lane & distance) == 0);
      order_key = take_minimum ? min(order_key, peer) : max(order_key, peer);
    }
  }
  return order_key & (kExperts - 1);
}

// Four accumulator slots exactly reproduce native Reduce.cuh's vt0=4. The
// corresponding native outer reduction uses block=(32,4), grid=(2,64), vec4.
// Sixteen warps compute a block's token rows concurrently. Four warps then fold
// the probabilities from shared memory in the original four-iteration order.
// A cooperative grid supplies synchronization without a scratch memset launch.
__global__ __launch_bounds__(kThreads, 2) void learned_router_forward_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ expert_bias,
    float* __restrict__ weights,
    int64_t* __restrict__ expert_ids,
    float* __restrict__ scores_output,
    float* __restrict__ row_norms,
    float* __restrict__ raw_loss,
    int64_t* __restrict__ dispatch_counts,
    bool* __restrict__ routing_map,
    float* __restrict__ selected_scores_output,
    float* __restrict__ route_denominators,
    float* __restrict__ norm_denominators,
    float* __restrict__ auxiliary_frequencies,
    float* __restrict__ partial_probability,
    int* __restrict__ partial_counts,
    float* __restrict__ partition_probability,
    int* __restrict__ partition_counts) {
  const auto grid = cg::this_grid();
  const int lane = threadIdx.x & 31;
  const int token_lane = (threadIdx.x >> 5) & 3;
  const int warp_index = threadIdx.x >> 5;
  const int partition = blockIdx.x % kPartitions;
  const int accumulator_slot = blockIdx.x / kPartitions;
  __shared__ float shared_values[16][kExperts];
  __shared__ int shared_counts[4][kExperts];
  float biases[8];
  #pragma unroll
  for (int item = 0; item < 8; ++item) {
    const int expert = lane * 4 + (item & 3) + (item >= 4 ? 128 : 0);
    biases[item] = expert_bias ? expert_bias[expert] : 0.0f;
  }

  {
    const int iteration = threadIdx.x >> 7;
    const int token = partition * 4 + token_lane +
        (accumulator_slot + iteration * 4) * 256;
    float4 logits_first;
    float4 logits_second;
    if ((reinterpret_cast<uintptr_t>(logits) & 15u) == 0) {
      logits_first = reinterpret_cast<const float4*>(logits)[token * 64 + lane];
      logits_second = reinterpret_cast<const float4*>(logits)[token * 64 + lane + 32];
    } else {
      // A contiguous tensor can still have an unaligned storage offset.
      const int offset = token * kExperts + lane * 4;
      logits_first = make_float4(logits[offset], logits[offset + 1], logits[offset + 2], logits[offset + 3]);
      logits_second = make_float4(logits[offset + 128], logits[offset + 129], logits[offset + 130], logits[offset + 131]);
    }
    float scores[8] = {
        sigmoid_native(logits_first.x), sigmoid_native(logits_first.y),
        sigmoid_native(logits_first.z), sigmoid_native(logits_first.w),
        sigmoid_native(logits_second.x), sigmoid_native(logits_second.y),
        sigmoid_native(logits_second.z), sigmoid_native(logits_second.w)};
    const float4 scores_first = make_float4(scores[0], scores[1], scores[2], scores[3]);
    const float4 scores_second = make_float4(scores[4], scores[5], scores[6], scores[7]);
    reinterpret_cast<float4*>(scores_output)[token * 64 + lane] = scores_first;
    reinterpret_cast<float4*>(scores_output)[token * 64 + lane + 32] = scores_second;
    reinterpret_cast<float4*>(shared_values[warp_index])[lane] = scores_first;
    reinterpret_cast<float4*>(shared_values[warp_index])[lane + 32] = scores_second;
    // Native vectorized norm256: four independent accumulators, each receiving
    // two values separated by 128, then sequential combine and shuffle-down.
    float row_norm = __fadd_rn(scores[0], scores[4]);
    #pragma unroll
    for (int item = 1; item < 4; ++item) {
      row_norm = __fadd_rn(row_norm, __fadd_rn(scores[item], scores[item + 4]));
    }
    row_norm = warp_sum(row_norm);
    if (lane == 0) row_norms[token] = row_norm;
    const float divisor = isnan(row_norm) ? row_norm : fmaxf(row_norm, 1.0e-12f);
    if (lane == 0) norm_denominators[token] = divisor;
    float choice_scores[8];
    #pragma unroll
    for (int item = 0; item < 8; ++item) {
      choice_scores[item] = expert_bias ? __fadd_rn(scores[item], biases[item]) : scores[item];
    }
    const unsigned selected_groups = select_groups(choice_scores, lane);
    #pragma unroll
    for (int item = 0; item < 8; ++item) {
      const int group = lane / 8 + (item >= 4 ? 4 : 0);
      if (!(selected_groups & (1u << group))) choice_scores[item] = -INFINITY;
    }
    int dispatch_selected[8];
    int choice_expert = 0;
    unsigned choice_key = 0;
    const unsigned threshold = select_eight(choice_scores, dispatch_selected, lane, choice_expert, choice_key);
    __syncwarp();
    if (lane < 8) {
      const int expert = native_order(choice_expert, choice_key, threshold, lane);
      const float selected = shared_values[warp_index][expert];
      float denominator = selected;
      #pragma unroll
      for (int offset = 4; offset; offset >>= 1) {
        denominator = __fadd_rn(
            denominator, __shfl_down_sync(0xffu, denominator, offset, 8));
      }
      denominator = __fadd_rn(__shfl_sync(0xffu, denominator, 0, 8), 1.0e-20f);
      weights[token * 8 + lane] = __fmul_rn(__fdiv_rn(selected, denominator), 2.5f);
      expert_ids[token * 8 + lane] = expert;
      selected_scores_output[token * 8 + lane] = selected;
      if (lane == 0) route_denominators[token] = denominator;
    }
    #pragma unroll
    for (int item = 0; item < 8; ++item) {
      const int expert = lane * 4 + (item & 3) + (item >= 4 ? 128 : 0);
      routing_map[token * kExperts + expert] = dispatch_selected[item];
    }
    // Packed integer counts use one byte per expert; each partition later sums
    // at most 64 tokens, so additions cannot carry between expert counts.
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int item = half * 4;
      const int expert_vector = lane + half * 32;
      reinterpret_cast<float4*>(shared_values[warp_index])[expert_vector] = make_float4(
          __fdiv_rn(scores[item], divisor), __fdiv_rn(scores[item + 1], divisor),
          __fdiv_rn(scores[item + 2], divisor), __fdiv_rn(scores[item + 3], divisor));
      reinterpret_cast<unsigned*>(shared_counts)[warp_index * 64 + expert_vector] =
          pack_counts(dispatch_selected[item], dispatch_selected[item + 1],
              dispatch_selected[item + 2], dispatch_selected[item + 3]);
    }
  }
  __syncthreads();
  if (threadIdx.x < 128) {
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int expert_vector = lane + half * 32;
      float4 probability = make_float4(0.f, 0.f, 0.f, 0.f);
      unsigned count_word = 0;
      #pragma unroll
      for (int iteration = 0; iteration < 4; ++iteration) {
        const int row_slot = token_lane + iteration * 4;
        probability = add_vectors(probability,
            reinterpret_cast<float4*>(shared_values[row_slot])[expert_vector]);
        count_word += reinterpret_cast<unsigned*>(shared_counts)[row_slot * 64 + expert_vector];
      }
      const int offset = ((partition * 4 + accumulator_slot) * 4 + token_lane) * 64 + expert_vector;
      reinterpret_cast<float4*>(partial_probability)[offset] = probability;
      partial_counts[offset] = count_word;
    }
  }
  grid.sync();

  if (blockIdx.x < kPartitions) {
    if (threadIdx.x < 128) {
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int expert_vector = lane + half * 32;
      int offset = (blockIdx.x * 16 + token_lane) * 64 + expert_vector;
      float4 value = reinterpret_cast<const float4*>(partial_probability)[offset];
      unsigned count_word = partial_counts[offset];
      #pragma unroll
      for (int slot = 1; slot < 4; ++slot) {
        value = add_vectors(value, reinterpret_cast<const float4*>(partial_probability)[offset + slot * 256]);
        count_word += partial_counts[offset + slot * 256];
      }
      reinterpret_cast<float4*>(shared_values[token_lane])[expert_vector] = value;
      shared_counts[token_lane][expert_vector] = count_word;
    }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      #pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int expert_vector = lane + half * 32;
        const float4 first = reinterpret_cast<float4*>(shared_values[0])[expert_vector];
        const float4 second = reinterpret_cast<float4*>(shared_values[1])[expert_vector];
        const float4 third = reinterpret_cast<float4*>(shared_values[2])[expert_vector];
        const float4 fourth = reinterpret_cast<float4*>(shared_values[3])[expert_vector];
        reinterpret_cast<float4*>(partition_probability)[blockIdx.x * 64 + expert_vector] =
            add_vectors(add_vectors(first, third), add_vectors(second, fourth));
        partition_counts[blockIdx.x * 64 + expert_vector] = shared_counts[0][expert_vector] +
            shared_counts[1][expert_vector] + shared_counts[2][expert_vector] + shared_counts[3][expert_vector];
      }
    }
  }
  grid.sync();

  if (blockIdx.x == 0) {
    if (threadIdx.x < 128) {
    #pragma unroll 1
    for (int half = 0; half < 2; ++half) {
      const int expert_vector = lane + half * 32;
      float4 values[16];
      unsigned count_words[16];
      #pragma unroll
      for (int iteration = 0; iteration < 16; ++iteration) {
        const int partition_index = token_lane + iteration * 4;
        values[iteration] = reinterpret_cast<const float4*>(partition_probability)[partition_index * 64 + expert_vector];
        count_words[iteration] = partition_counts[partition_index * 64 + expert_vector];
      }
      float4 value = make_float4(0.f, 0.f, 0.f, 0.f);
      int4 count = make_int4(0, 0, 0, 0);
      #pragma unroll
      for (int iteration = 0; iteration < 16; ++iteration) {
        value = add_vectors(value, values[iteration]);
        const unsigned count_word = count_words[iteration];
        count.x += count_word & 0xff;
        count.y += (count_word >> 8) & 0xff;
        count.z += (count_word >> 16) & 0xff;
        count.w += count_word >> 24;
      }
      reinterpret_cast<float4*>(shared_values[token_lane])[expert_vector] = value;
      reinterpret_cast<int4*>(shared_counts[token_lane])[expert_vector] = count;
    }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      #pragma unroll
      for (int item = 0; item < 8; ++item) {
        const int expert = lane * 4 + (item & 3) + (item >= 4 ? 128 : 0);
        const float value = __fadd_rn(
            __fadd_rn(shared_values[0][expert], shared_values[2][expert]),
            __fadd_rn(shared_values[1][expert], shared_values[3][expert]));
        const int count = shared_counts[0][expert] + shared_counts[1][expert] +
            shared_counts[2][expert] + shared_counts[3][expert];
        const float frequency = __fmul_rn(static_cast<float>(count), 0.0078125f);
        auxiliary_frequencies[expert] = frequency;
        shared_values[0][expert] = __fmul_rn(frequency, value);
        dispatch_counts[expert] = count;
      }
    }
    __syncthreads();
    // Native contiguous sum256 has 64 threads, vec4 and a 64->32->... tree.
    if (threadIdx.x < 64) {
      float value = shared_values[0][threadIdx.x * 4];
      #pragma unroll
      for (int item = 1; item < 4; ++item) {
        value = __fadd_rn(value, shared_values[0][threadIdx.x * 4 + item]);
      }
      shared_values[1][threadIdx.x] = value;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      float value = __fadd_rn(shared_values[1][lane], shared_values[1][lane + 32]);
      value = warp_sum(value);
      if (threadIdx.x == 0) {
        *raw_loss = value;
      }
    }
  }
}

at::Tensor empty_output(at::IntArrayRef size, const at::TensorOptions& options) {
  // All entries are written by the kernel before any read. Bypass deterministic
  // allocation poisoning without changing the user's global debug settings.
  return at::Tensor(at::detail::empty_cuda(size, options));
}
} // namespace

std::vector<at::Tensor> learned_router_forward_cuda(
    const at::Tensor& logits, const std::optional<at::Tensor>& expert_bias) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat &&
      logits.is_contiguous() && logits.sizes() == at::IntArrayRef({kRows, kExperts}) &&
      !logits.is_neg() && !logits.is_conj(),
      "learned router requires contiguous CUDA FP32 logits[4096,256]");
  TORCH_CHECK(!expert_bias.has_value() ||
      (expert_bias->device() == logits.device() && expert_bias->scalar_type() == at::kFloat &&
       expert_bias->is_contiguous() && expert_bias->sizes() == at::IntArrayRef({kExperts}) &&
       !expert_bias->is_neg() && !expert_bias->is_conj()),
      "learned router requires optional FP32 bias[256]");
  const c10::cuda::CUDAGuard guard(logits.device());
  // Each invocation owns its scratch. Independent streams cannot race through
  // a cached workspace, and all consumed entries are overwritten before use.
  const auto options = logits.options();
  auto partial_probability = empty_output({64, 4, 4, 256}, options);
  auto partial_counts = empty_output({64, 4, 4, 64}, options.dtype(at::kInt));
  auto partition_probability = empty_output({64, 256}, options);
  auto partition_counts = empty_output({64, 64}, options.dtype(at::kInt));
  auto weights = empty_output({kRows, 8}, options);
  auto ids = empty_output({kRows, 8}, options.dtype(at::kLong));
  auto routing_map = empty_output({kRows, kExperts}, options.dtype(at::kBool));
  auto raw = empty_output({}, options);
  auto dispatch = empty_output({kExperts}, options.dtype(at::kLong));
  auto scores = empty_output(logits.sizes(), options);
  auto norms = empty_output({kRows, 1}, options);
  auto selected = empty_output({kRows, 8}, options);
  auto route_denominator = empty_output({kRows, 1}, options);
  auto norm_denominator = empty_output({kRows, 1}, options);
  auto frequencies = empty_output({kExperts}, options);
  auto* logits_ptr = logits.data_ptr<float>();
  auto* bias_ptr = expert_bias.has_value() ? expert_bias->data_ptr<float>() : nullptr;
  auto* weights_ptr = weights.data_ptr<float>();
  auto* ids_ptr = ids.data_ptr<int64_t>();
  auto* scores_ptr = scores.data_ptr<float>();
  auto* norms_ptr = norms.data_ptr<float>();
  auto* raw_ptr = raw.data_ptr<float>();
  auto* dispatch_ptr = dispatch.data_ptr<int64_t>();
  auto* map_ptr = routing_map.data_ptr<bool>();
  auto* selected_ptr = selected.data_ptr<float>();
  auto* route_denominator_ptr = route_denominator.data_ptr<float>();
  auto* norm_denominator_ptr = norm_denominator.data_ptr<float>();
  auto* frequencies_ptr = frequencies.data_ptr<float>();
  auto* partial_probability_ptr = partial_probability.data_ptr<float>();
  auto* partial_counts_ptr = partial_counts.data_ptr<int>();
  auto* partition_probability_ptr = partition_probability.data_ptr<float>();
  auto* partition_counts_ptr = partition_counts.data_ptr<int>();
  void* arguments[] = {
      &logits_ptr, &bias_ptr, &weights_ptr, &ids_ptr, &scores_ptr, &norms_ptr,
      &raw_ptr, &dispatch_ptr, &map_ptr, &selected_ptr, &route_denominator_ptr,
      &norm_denominator_ptr, &frequencies_ptr, &partial_probability_ptr,
      &partial_counts_ptr, &partition_probability_ptr, &partition_counts_ptr};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(learned_router_forward_kernel), dim3(256), dim3(kThreads),
      arguments, 0, at::cuda::getCurrentCUDAStream()));
  return {weights, ids, routing_map, raw, dispatch, scores, norms, selected,
      route_denominator, norm_denominator, frequencies};
}
