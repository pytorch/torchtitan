import torch
from apex.normalization import FusedLayerNorm
from import FusedRMSNorm


from torch.testing._internal import common_utils
from torch.testing._internal.common_device_type import instantiate_device_type_tests

from itertools import product

def _prep_inputs(batch_size, normalized_shape, dtype):
    shape = (batch_size, *normalized_shape)
    fused = torch.randn(shape).cuda().requires_grad_(True)
    with torch.no_grad():
        native = fused.clone().to(dtype).requires_grad_(True)
    return native, fused
