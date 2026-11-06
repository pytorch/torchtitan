from torch import device
import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2):
        v3 = torch.ops.aten._to_copy.default(in0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'));  in0 = None
        v4 = torch.ops.aten.mul.Tensor(v3, in1);  in1 = None
        v5 = torch.ops.aten.slice.Tensor(v4, 2, 0, 32)
        v6 = torch.ops.aten.slice.Tensor(v4, 2, 32, 64);  v4 = None
        v7 = torch.ops.aten.neg.default(v5);  v5 = None
        v8 = torch.ops.aten.slice_backward.default(v7, [16384, 8, 64], 2, 32, 9223372036854775807, 1);  v7 = None
        v9 = torch.ops.aten.slice_backward.default(v6, [16384, 8, 64], 2, 0, 32, 1);  v6 = None
        v10 = torch.ops.aten.add.Tensor(v8, v9);  v8 = v9 = None
        v11 = torch.ops.aten.mul.Tensor(v3, in2);  v3 = in2 = None
        v12 = torch.ops.aten.add.Tensor(v10, v11);  v10 = v11 = None
        v13 = torch.ops.aten._to_copy.default(v12, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v12 = None
        return v13


def run(*args):
    return Reference()(*args)
