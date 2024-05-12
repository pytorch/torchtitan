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
