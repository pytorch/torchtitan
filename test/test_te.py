import torch
import torch.nn as nn
import torchtitan.te_utils as te_utils
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

fp8_format = Format.HYBRID
fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
maybe_te_float8_ctx = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)

def test():
    # for now, single GPU smoke test of TE fp8

    x = torch.randn(32, 32, device='cuda')

    m = nn.Sequential(nn.Linear(32, 32)).cuda()
    te_utils.swap_linear_to_te_linear(m)
    print(m)

    with maybe_te_float8_ctx:
        y = m(x)
    y.sum().backward()

    print('done')

if __name__ == '__main__':
    test()
