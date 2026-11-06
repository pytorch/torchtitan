from torch import device
import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1):
        v2 = torch.ops.aten.new_zeros.default(in0, [9, 2880], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'))
        v3 = torch.ops.aten.index_add.default(v2, 0, in1, in0);  v2 = in1 = in0 = None
        v4 = torch.ops.aten.slice.Tensor(v3, 0, 0, 8);  v3 = None
        return v4


def run(*args):
    return Reference()(*args)
