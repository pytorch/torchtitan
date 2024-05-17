
import sys

import pytest
import torch
import torch.nn as nn
from torch import Tensor

#sys.path.append("..")
from fused_rms_norm import FusedRMSNorm
#from nv_apex import FusedRMSNorm as nv_apex_FusedRMSNorm

#from .testing_utils import assert_expected, gpu_test, set_rng_seed


n_dim = 768

layer_weight_size = (5, n_dim,n_dim)
test_eps = 1e-8
atol_precision = 1e-2
rtol_precision = 1e-2

sample_x = torch.randn(
    layer_weight_size, dtype=torch.float32, device="cuda", requires_grad=True
)

#expected_rms_func = TorchRMSNorm(layer_weight_size, eps=test_eps).to("cuda")
fused_rms_norm = FusedRMSNorm(layer_weight_size, eps=test_eps).to("cuda")
fused_out = fused_rms_norm(sample_x)
print(f"Success! {fused_out[0:5]=}")

# Backward pass
grad_output = torch.randn_like(fused_out)

        #expected_rms.backward(grad_output)
        #dy_expected = sample_x.grad

        #sample_x.grad = None

fused_out.backward(grad_output)
dy_fused = sample_x.grad

print(f"{dy_fused[0:5]=}")
        #print(f"{dy_expected[0:5]=}")
