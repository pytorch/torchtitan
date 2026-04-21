#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> rope_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& freqs);

std::vector<torch::Tensor> rope_backward_cuda(
    const torch::Tensor& grad_k,
    const torch::Tensor& grad_q,
    const torch::Tensor& freqs);

namespace {

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

void check_rope_input(const torch::Tensor& tensor, const char* name) {
  check_cuda_tensor(tensor, name);
  TORCH_CHECK(tensor.dtype() == torch::kBFloat16, name, " must be bf16");
  TORCH_CHECK(tensor.dim() == 4, name, " must be rank-4");
  TORCH_CHECK(tensor.size(-1) % 2 == 0, name, " head_dim must be even");
  TORCH_CHECK(tensor.stride(-1) == 1, name, " must have contiguous head_dim");
  TORCH_CHECK(
      tensor.is_non_overlapping_and_dense(),
      name,
      " must be dense and non-overlapping");
}

void check_freqs(const torch::Tensor& freqs) {
  check_cuda_tensor(freqs, "freqs");
  TORCH_CHECK(freqs.dtype() == torch::kComplexFloat, "freqs must be complex64");
  TORCH_CHECK(freqs.dim() == 4, "freqs must be rank-4");
  TORCH_CHECK(freqs.size(0) == 1, "freqs batch dimension must be 1");
  TORCH_CHECK(freqs.size(2) == 1, "freqs head broadcast dimension must be 1");
  TORCH_CHECK(freqs.stride(-1) == 1, "freqs must be contiguous along rotary dim");
}

void check_compatible(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& freqs) {
  TORCH_CHECK(q.device() == k.device(), "q/k must share a device");
  TORCH_CHECK(q.device() == freqs.device(), "q/k/freqs must share a device");
  TORCH_CHECK(q.size(0) == k.size(0), "q/k batch size mismatch");
  TORCH_CHECK(q.size(1) == k.size(1), "q/k sequence length mismatch");
  TORCH_CHECK(q.size(3) == k.size(3), "q/k head_dim mismatch");
  TORCH_CHECK(freqs.size(1) == q.size(1), "freqs sequence length mismatch");
  TORCH_CHECK(
      freqs.size(3) * 2 == q.size(3),
      "freqs rotary dimension must match q/k head_dim");
}

} // namespace

std::vector<torch::Tensor> rope_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& freqs) {
  check_rope_input(q, "q");
  check_rope_input(k, "k");
  check_freqs(freqs);
  check_compatible(q, k, freqs);
  return rope_forward_cuda(q, k, freqs);
}

std::vector<torch::Tensor> rope_backward(
    const torch::Tensor& grad_k,
    const torch::Tensor& grad_q,
    const torch::Tensor& freqs) {
  check_rope_input(grad_k, "grad_k");
  check_rope_input(grad_q, "grad_q");
  check_freqs(freqs);
  check_compatible(grad_q, grad_k, freqs);
  return rope_backward_cuda(grad_k, grad_q, freqs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "rope_forward",
      &rope_forward,
      "Fuse RoPE forward for q/k bf16 tensors");
  m.def(
      "rope_backward",
      &rope_backward,
      "Fuse RoPE backward for q/k bf16 tensors");
}
