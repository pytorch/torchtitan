#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kThreadsPerBlock = 256;

template <bool Conjugate>
__global__ void rope_kernel(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    const float2* freqs,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t input_stride0,
    int64_t input_stride1,
    int64_t input_stride2,
    int64_t input_stride3,
    int64_t output_stride0,
    int64_t output_stride1,
    int64_t output_stride2,
    int64_t output_stride3,
    int64_t freqs_stride0,
    int64_t freqs_stride1,
    int64_t freqs_stride2,
    int64_t freqs_stride3) {
  const int64_t rotary_dim = head_dim / 2;
  const int64_t total =
      batch_size * seq_len * num_heads * rotary_dim;

  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }

  int64_t linear = index;
  const int64_t rotary_index = linear % rotary_dim;
  linear /= rotary_dim;
  const int64_t head_index = linear % num_heads;
  linear /= num_heads;
  const int64_t seq_index = linear % seq_len;
  const int64_t batch_index = linear / seq_len;

  const int64_t input_base =
      batch_index * input_stride0 +
      seq_index * input_stride1 +
      head_index * input_stride2 +
      rotary_index * 2 * input_stride3;
  const int64_t output_base =
      batch_index * output_stride0 +
      seq_index * output_stride1 +
      head_index * output_stride2 +
      rotary_index * 2 * output_stride3;
  const int64_t freqs_index =
      seq_index * freqs_stride1 + rotary_index * freqs_stride3;

  const float2 freq = freqs[freqs_index];
  const float cos = freq.x;
  const float sin = Conjugate ? -freq.y : freq.y;

  const float real = __bfloat162float(input[input_base]);
  const float imag = __bfloat162float(input[input_base + input_stride3]);

  const float out_real = real * cos - imag * sin;
  const float out_imag = real * sin + imag * cos;

  output[output_base] = __float2bfloat16_rn(out_real);
  output[output_base + output_stride3] = __float2bfloat16_rn(out_imag);
}

template <bool Conjugate>
torch::Tensor launch_rope(
    const torch::Tensor& input,
    const torch::Tensor& freqs) {
  auto output = torch::empty_like(input);

  const int64_t batch_size = input.size(0);
  const int64_t seq_len = input.size(1);
  const int64_t num_heads = input.size(2);
  const int64_t head_dim = input.size(3);
  const int64_t total = batch_size * seq_len * num_heads * (head_dim / 2);

  if (total == 0) {
    return output;
  }

  const dim3 block(kThreadsPerBlock);
  const dim3 grid((total + kThreadsPerBlock - 1) / kThreadsPerBlock);
  auto stream = c10::cuda::getCurrentCUDAStream(input.device().index());

  rope_kernel<Conjugate><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      reinterpret_cast<const float2*>(freqs.data_ptr<c10::complex<float>>()),
      batch_size,
      seq_len,
      num_heads,
      head_dim,
      input.stride(0),
      input.stride(1),
      input.stride(2),
      input.stride(3),
      output.stride(0),
      output.stride(1),
      output.stride(2),
      output.stride(3),
      freqs.stride(0),
      freqs.stride(1),
      freqs.stride(2),
      freqs.stride(3));
  AT_CUDA_CHECK(cudaGetLastError());

  return output;
}

} // namespace

std::vector<torch::Tensor> rope_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& freqs) {
  c10::cuda::CUDAGuard device_guard(q.device());
  return {
      launch_rope<false>(q, freqs),
      launch_rope<false>(k, freqs),
  };
}

std::vector<torch::Tensor> rope_backward_cuda(
    const torch::Tensor& grad_k,
    const torch::Tensor& grad_q,
    const torch::Tensor& freqs) {
  c10::cuda::CUDAGuard device_guard(grad_q.device());
  return {
      launch_rope<true>(grad_k, freqs),
      launch_rope<true>(grad_q, freqs),
  };
}
