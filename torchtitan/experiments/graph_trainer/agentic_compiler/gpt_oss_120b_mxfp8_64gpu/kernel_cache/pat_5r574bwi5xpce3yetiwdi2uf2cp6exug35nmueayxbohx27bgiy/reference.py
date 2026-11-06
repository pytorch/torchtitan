from torch import device
import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2):
        v3 = torch.ops.aten.new_zeros.default(in0, [16384, 2880], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'))
        v4 = torch.ops.aten.index_put.default(v3, [in1], in0, True);  v3 = in1 = in0 = None
        v5 = torch.ops.aten.add.Tensor(v4, in2);  v4 = in2 = None
        return v5


def run(*args):
    return Reference()(*args)
