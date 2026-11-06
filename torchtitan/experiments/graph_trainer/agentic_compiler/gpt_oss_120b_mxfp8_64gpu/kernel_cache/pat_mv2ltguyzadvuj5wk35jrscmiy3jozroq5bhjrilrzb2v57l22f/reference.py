from torch import device
import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2):
        v3 = torch.ops.aten._to_copy.default(in0, dtype = torch.float32);  in0 = None
        v4 = torch.ops.aten.mul.Tensor(v3, in1);  in1 = None
        v5 = torch.ops.aten.slice.Tensor(v3, 2, 0, 32)
        v6 = torch.ops.aten.slice.Tensor(v3, 2, 32, 9223372036854775807);  v3 = None
        v7 = torch.ops.aten.neg.default(v6);  v6 = None
        v8 = torch.ops.aten.cat.default([v7, v5], -1);  v7 = v5 = None
        v9 = torch.ops.aten.mul.Tensor(v8, in2);  v8 = in2 = None
        v10 = torch.ops.aten.add.Tensor(v4, v9);  v4 = v9 = None
        v11 = torch.ops.aten._to_copy.default(v10, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v10 = None
        return v11


def run(*args):
    return Reference()(*args)
