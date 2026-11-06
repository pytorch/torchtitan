from torch import device
import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3):
        v4 = torch.ops.aten.reshape.default(in0, [16384, 64, 64]);  in0 = None
        v5 = torch.ops.aten.mul.Tensor(v4, in1);  in1 = None
        v6 = torch.ops.aten.mul.Tensor(v4, in2);  v4 = in2 = None
        v7 = torch.ops.aten.sum.dim_IntList(v5, [2], True);  v5 = None
        v8 = torch.ops.aten._to_copy.default(v7, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'));  v7 = None
        v9 = torch.ops.aten.squeeze.dim(v8, -1);  v8 = None
        v10 = torch.ops.aten.alias.default(in3);  in3 = None
        v11 = torch.ops.aten.sigmoid_backward.default(v9, v10);  v9 = v10 = None
        v12 = torch.ops.aten.neg.default(v11);  v11 = None
        v13 = torch.ops.aten.sum.dim_IntList(v12, [0], True);  v12 = None
        v14 = torch.ops.aten._to_copy.default(v13, dtype = torch.bfloat16);  v13 = None
        v15 = torch.ops.aten.reshape.default(v14, [64]);  v14 = None
        return (v6, v15)


def run(*args):
    return Reference()(*args)
