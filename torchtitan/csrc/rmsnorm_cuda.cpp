#include <torch/extension.h>
#include <vector>
#include <cassert>
#include "utils.h"


namespace {
void compute_n1_n2(const at::Tensor& input,
                    at::IntArrayRef normalized_shape,
                    int& n1, int& n2) {
    int idiff = input.ndimension() - normalized_shape.size();

    n2 = 1;
    for (int i = 0; i < normalized_shape.size(); ++i) {
        TORCH_CHECK(input.size(i + idiff) == normalized_shape[i],
            "Input shape and normalized_shape do not match.");
        n2 *= normalized_shape[i];
    }

    n1 = input.size(0);
    for (int i = 1; i < idiff; ++i) {
        n1 *= input.size(i);
    }
}

} // namespace

void check_args(
    at::IntArrayRef normalized_shape,
    at::Tensor gamma
    )
{
    TORCH_CHECK(!gamma.defined() || gamma.sizes().equals(normalized_shape));
}

void check_args(
    at::Tensor input,
    at::IntArrayRef normalized_shape,
    int& n1,
    int& n2
    )
{
    int64_t normalized_ndim = normalized_shape.size();

    if (normalized_ndim < 1) {
      std::stringstream ss;
      ss << "Expected normalized_shape to be at least 1-dimensional, i.e., "
         << "containing at least one element, but got normalized_shape="
         << normalized_shape;
      throw std::runtime_error(ss.str());
    }

    auto input_shape = input.sizes();
    auto input_ndim = input.dim();

    if (input_ndim < normalized_ndim ||
        !input_shape.slice(input_ndim - normalized_ndim).equals(normalized_shape)) {
      std::stringstream ss;
      ss << "Given normalized_shape=" << normalized_shape
         << ", expected input with shape [*";
      for (auto size : normalized_shape) {
        ss << ", " << size;
      }
      ss << "], but got input of size" << input_shape;
      throw std::runtime_error(ss.str());
    }

    compute_n1_n2(input,normalized_shape,n1,n2);
}

std::vector<at::Tensor> rms_norm_affine(
    at::Tensor input,
    at::IntArrayRef normalized_shape,
    at::Tensor gamma,
    double epsilon) {
  //CHECK_INPUT(input);
  //CHECK_INPUT(gamma);
  int n1,n2;
  //check_args(input,normalized_shape,gamma,n1,n2);
  check_args(input,normalized_shape,n1,n2);
  check_args(normalized_shape,gamma);
  at::Tensor output = at::empty_like(input);
  const auto stats_dtype = (input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16) ? at::ScalarType::Float : input.scalar_type();
  at::Tensor invvar = at::empty({n1}, input.options().dtype(stats_dtype));
  cuda_rms_norm(&output,&invvar,&input,n1,n2,
      normalized_shape,&gamma,epsilon);
  return {output, invvar};
}
