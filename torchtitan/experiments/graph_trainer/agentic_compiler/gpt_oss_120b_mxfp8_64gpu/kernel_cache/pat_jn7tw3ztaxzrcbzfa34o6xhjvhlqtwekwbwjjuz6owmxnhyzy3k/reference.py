from torch import device
import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2):
        v3 = torch.ops.aten.new_zeros.default(in0, [in1, 2880], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  in1 = None
        v4 = torch.ops.aten.index_put.default(v3, [in2], in0, True);  v3 = in2 = in0 = None
        return v4


def run(*args):
    return Reference()(*args)
