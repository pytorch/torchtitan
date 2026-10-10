// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <optional>

namespace {
constexpr unsigned kWarpMask = 0xffffffffu;
constexpr int kRows = 4096;
constexpr int kExperts = 256;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

struct BackwardInputs {
  const float* scores;
  const float* row_norm;
  const int64_t* expert_ids;
  const float* selected_scores;
  const float* route_denominator;
  const float* norm_denominator;
  const float* auxiliary_frequencies;
  const float* grad_route_weights;
  const float* grad_raw_sum;
  int64_t route_row_stride;
  int64_t route_column_stride;
  bool negate_route_gradient;
  bool negate_auxiliary_gradient;
  bool deterministic;
};

__device__ __forceinline__ float4 load_four(
    const float* pointer, int64_t offset, int64_t stride = 1) {
  if (stride == 1 && ((reinterpret_cast<uintptr_t>(pointer + offset) & 15u) == 0)) {
    return *reinterpret_cast<const float4*>(pointer + offset);
  }
  return make_float4(pointer[offset], pointer[offset + stride],
      pointer[offset + 2 * stride], pointer[offset + 3 * stride]);
}

__device__ __forceinline__ float warp_sum(float value) {
  #pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    value = __fadd_rn(value, __shfl_down_sync(kWarpMask, value, offset));
  }
  return __shfl_sync(kWarpMask, value, 0);
}

__device__ __forceinline__ float sum_256(const float (&values)[8]) {
  // Native contiguous row reduction: four accumulator slots, with each slot
  // receiving its two values 128 experts apart before the warp reduction.
  float result = __fadd_rn(__fadd_rn(0.0f, values[0]), values[4]);
  #pragma unroll
  for (int item = 1; item < 4; ++item) {
    result = __fadd_rn(result, __fadd_rn(__fadd_rn(0.0f, values[item]), values[item + 4]));
  }
  return warp_sum(result);
}

__device__ __forceinline__ float refined_native_reciprocal(float denominator) {
  float approximation;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(approximation) : "f"(denominator));
  const float error = __fmaf_rn(-denominator, approximation, 1.0f);
  return __fmaf_rn(approximation, error, approximation);
}

__device__ __forceinline__ float regular_native_divide(
    float numerator, float denominator, float reciprocal) {
  const float quotient = __fmaf_rn(numerator, reciprocal, 0.0f);
  const float remainder = __fmaf_rn(-denominator, quotient, numerator);
  return __fmaf_rn(reciprocal, remainder, quotient);
}

template <bool HasRoute, bool HasAuxiliary>
__global__ void learned_router_backward_kernel(BackwardInputs input, float* output) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * kWarps + warp;
  __shared__ float route_gradients[kWarps][kExperts];
  __shared__ float probability_sum_gradients[kExperts];

  if constexpr (HasRoute) {
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
      reinterpret_cast<float4*>(route_gradients[warp])[lane + half * 32] =
          make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  if constexpr (HasAuxiliary) {
    float upstream = *input.grad_raw_sum;
    if (input.negate_auxiliary_gradient) upstream = -upstream;
    #pragma unroll
    for (int expert = threadIdx.x; expert < kExperts; expert += kThreads) {
      probability_sum_gradients[expert] = __fmul_rn(upstream, input.auxiliary_frequencies[expert]);
    }
    __syncthreads();
  } else if constexpr (HasRoute) {
    __syncwarp();
  }
  if constexpr (HasRoute) {
    if (lane < 8) {
      float upstream = input.grad_route_weights[
          row * input.route_row_stride + lane * input.route_column_stride];
      if (input.negate_route_gradient) upstream = -upstream;
      const float scaled = __fmul_rn(upstream, 2.5f);
      const float selected = input.selected_scores[row * 8 + lane];
      const float denominator = input.route_denominator[row];
      const float contribution = __fmul_rn(-scaled,
          __fdiv_rn(__fdiv_rn(selected, denominator), denominator));
      float denominator_gradient = __fadd_rn(0.0f, contribution);
      if (input.route_row_stride > 0 && input.route_column_stride > input.route_row_stride) {
        // TensorIterator preserves this input's transposed memory order. The
        // corresponding native outer sum8 folds four accumulator slots in order.
        denominator_gradient = __fadd_rn(denominator_gradient,
            __shfl_down_sync(0xffu, contribution, 4, 8));
        float folded = __shfl_sync(0xffu, denominator_gradient, 0, 8);
        #pragma unroll
        for (int item = 1; item < 4; ++item) {
          folded = __fadd_rn(folded, __shfl_sync(0xffu, denominator_gradient, item, 8));
        }
        denominator_gradient = folded;
      } else {
        #pragma unroll
        for (int offset = 4; offset; offset >>= 1) {
          denominator_gradient = __fadd_rn(denominator_gradient,
              __shfl_down_sync(0xffu, denominator_gradient, offset, 8));
        }
        denominator_gradient = __shfl_sync(0xffu, denominator_gradient, 0, 8);
      }
      const float selected_gradient = __fadd_rn(__fdiv_rn(scaled, denominator), denominator_gradient);
      // Native unique-ID scatter_add still adds to a +0 destination. Its
      // nondeterministic FP32 atomic path additionally flushes subnormals.
      float scattered = __fadd_rn(0.0f, selected_gradient);
      if (!input.deterministic && (__float_as_uint(scattered) & 0x7fffffffu) < 0x00800000u) {
        scattered = 0.0f;
      }
      const int expert = static_cast<int>(input.expert_ids[row * 8 + lane]);
      CUDA_KERNEL_ASSERT(static_cast<unsigned>(expert) < kExperts);
      route_gradients[warp][expert] = scattered;
    }
    __syncwarp();
  }

  const int first_expert = lane * 4;
  const float4 first = load_four(input.scores, row * kExperts + first_expert);
  const float4 second = load_four(input.scores, row * kExperts + first_expert + 128);
  const float scores[8] = {first.x, first.y, first.z, first.w, second.x, second.y, second.z, second.w};
  float score_gradients[8] = {};
  if constexpr (HasAuxiliary) {
    const float denominator = input.norm_denominator[row];
    const float4 first_gradient = reinterpret_cast<float4*>(probability_sum_gradients)[lane];
    const float4 second_gradient = reinterpret_cast<float4*>(probability_sum_gradients)[lane + 32];
    const float gradients[8] = {first_gradient.x, first_gradient.y, first_gradient.z, first_gradient.w,
        second_gradient.x, second_gradient.y, second_gradient.z, second_gradient.w};
    float contributions[8];
    unsigned minimum_score = 0xffffffffu;
    unsigned maximum_score = 0;
    unsigned minimum_gradient = 0xffffffffu;
    unsigned maximum_gradient = 0;
    #pragma unroll
    for (int item = 0; item < 8; ++item) {
      const unsigned score_bits = __float_as_uint(scores[item]) & 0x7fffffffu;
      const unsigned gradient_bits = __float_as_uint(gradients[item]) & 0x7fffffffu;
      minimum_score = min(minimum_score, score_bits);
      maximum_score = max(maximum_score, score_bits);
      minimum_gradient = min(minimum_gradient, gradient_bits);
      maximum_gradient = max(maximum_gradient, gradient_bits);
    }
    const unsigned denominator_bits = __float_as_uint(denominator);
    // Conservative row-wide domain: denominator in [2**-12,2**12], absolute
    // scores in [2**-24,2**12], absolute upstream probability gradients in
    // [2**-40,2**12]. Both chained quotients and every correction intermediate
    // stay normal; zero, subnormal, nonfinite and other rows use native div.
    const bool regular_row = __all_sync(kWarpMask,
        denominator_bits >= 0x39800000u && denominator_bits <= 0x45800000u &&
        minimum_score >= 0x33800000u && maximum_score <= 0x45800000u &&
        minimum_gradient >= 0x2b800000u && maximum_gradient <= 0x45800000u);
    if (regular_row) {
      const float reciprocal = refined_native_reciprocal(denominator);
      #pragma unroll
      for (int item = 0; item < 8; ++item) {
        contributions[item] = __fmul_rn(-gradients[item], regular_native_divide(
            regular_native_divide(scores[item], denominator, reciprocal), denominator, reciprocal));
        const float direct = regular_native_divide(gradients[item], denominator, reciprocal);
        score_gradients[item] = direct;
      }
    } else {
      #pragma unroll
      for (int item = 0; item < 8; ++item) {
        contributions[item] = __fmul_rn(-gradients[item],
            __fdiv_rn(__fdiv_rn(scores[item], denominator), denominator));
        const float direct = __fdiv_rn(gradients[item], denominator);
        score_gradients[item] = direct;
      }
    }
    float norm_gradient = sum_256(contributions);
    if (!(input.row_norm[row] > 1.0e-12f)) norm_gradient = 0.0f;
    #pragma unroll
    for (int item = 0; item < 8; ++item) {
      const float sign = static_cast<float>((scores[item] > 0.0f) - (scores[item] < 0.0f));
      score_gradients[item] = __fadd_rn(score_gradients[item], __fmul_rn(sign, norm_gradient));
    }
  }
  if constexpr (HasRoute) {
    const float4 first_gradient = reinterpret_cast<float4*>(route_gradients[warp])[lane];
    const float4 second_gradient = reinterpret_cast<float4*>(route_gradients[warp])[lane + 32];
    const float gradients[8] = {first_gradient.x, first_gradient.y, first_gradient.z, first_gradient.w,
        second_gradient.x, second_gradient.y, second_gradient.z, second_gradient.w};
    #pragma unroll
    for (int item = 0; item < 8; ++item) {
      if constexpr (HasAuxiliary) score_gradients[item] = __fadd_rn(score_gradients[item], gradients[item]);
      else score_gradients[item] = gradients[item];
    }
  }
  #pragma unroll
  for (int item = 0; item < 8; ++item) {
    score_gradients[item] = __fmul_rn(__fmul_rn(score_gradients[item],
        __fsub_rn(1.0f, scores[item])), scores[item]);
  }
  reinterpret_cast<float4*>(output)[row * 64 + lane] =
      make_float4(score_gradients[0], score_gradients[1], score_gradients[2], score_gradients[3]);
  reinterpret_cast<float4*>(output)[row * 64 + lane + 32] =
      make_float4(score_gradients[4], score_gradients[5], score_gradients[6], score_gradients[7]);
}

void validate_saved(const at::Tensor& value, const at::Tensor& scores,
    at::IntArrayRef shape, at::ScalarType type) {
  TORCH_CHECK(value.device() == scores.device() && value.scalar_type() == type &&
      value.sizes() == shape && value.is_contiguous() && !value.is_neg() && !value.is_conj(),
      "invalid learned router backward saved tensor");
}

void validate_gradient(const std::optional<at::Tensor>& value, const at::Tensor& scores,
    at::IntArrayRef shape) {
  TORCH_CHECK(!value.has_value() || (value->device() == scores.device() &&
      value->scalar_type() == at::kFloat && value->sizes() == shape && !value->is_conj()),
      "invalid learned router backward output gradient");
}
} // namespace

at::Tensor learned_router_backward_cuda(
    const at::Tensor& scores, const at::Tensor& row_norm,
    const at::Tensor& expert_ids, const at::Tensor& selected_scores,
    const at::Tensor& route_denominator, const at::Tensor& norm_denominator,
    const at::Tensor& auxiliary_frequencies,
    const std::optional<at::Tensor>& grad_route_weights,
    const std::optional<at::Tensor>& grad_raw_sum) {
  TORCH_CHECK(scores.is_cuda(), "learned router backward requires CUDA tensors");
  validate_saved(scores, scores, {kRows, kExperts}, at::kFloat);
  validate_saved(row_norm, scores, {kRows, 1}, at::kFloat);
  validate_saved(expert_ids, scores, {kRows, 8}, at::kLong);
  validate_saved(selected_scores, scores, {kRows, 8}, at::kFloat);
  validate_saved(route_denominator, scores, {kRows, 1}, at::kFloat);
  validate_saved(norm_denominator, scores, {kRows, 1}, at::kFloat);
  validate_saved(auxiliary_frequencies, scores, {kExperts}, at::kFloat);
  validate_gradient(grad_route_weights, scores, {kRows, 8});
  validate_gradient(grad_raw_sum, scores, {});
  const int branches = int(grad_route_weights.has_value()) + 2 * int(grad_raw_sum.has_value());
  TORCH_CHECK(branches != 0, "learned router backward requires an output gradient");
  const c10::cuda::CUDAGuard guard(scores.device());
  auto output = at::Tensor(at::detail::empty_cuda(scores.sizes(), scores.options()));
  BackwardInputs input{
      scores.data_ptr<float>(), row_norm.data_ptr<float>(), expert_ids.data_ptr<int64_t>(),
      selected_scores.data_ptr<float>(), route_denominator.data_ptr<float>(), norm_denominator.data_ptr<float>(),
      auxiliary_frequencies.data_ptr<float>(),
      grad_route_weights.has_value() ? grad_route_weights->data_ptr<float>() : nullptr,
      grad_raw_sum.has_value() ? grad_raw_sum->data_ptr<float>() : nullptr,
      grad_route_weights.has_value() ? grad_route_weights->stride(0) : 0,
      grad_route_weights.has_value() ? grad_route_weights->stride(1) : 0,
      grad_route_weights.has_value() && grad_route_weights->is_neg(),
      grad_raw_sum.has_value() && grad_raw_sum->is_neg(),
      at::globalContext().deterministicAlgorithms()};
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(kRows / kWarps);
  const dim3 block(kThreads);
  switch (branches) {
    case 1: learned_router_backward_kernel<true, false><<<grid, block, 0, stream>>>(input, output.data_ptr<float>()); break;
    case 2: learned_router_backward_kernel<false, true><<<grid, block, 0, stream>>>(input, output.data_ptr<float>()); break;
    case 3: learned_router_backward_kernel<true, true><<<grid, block, 0, stream>>>(input, output.data_ptr<float>()); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
