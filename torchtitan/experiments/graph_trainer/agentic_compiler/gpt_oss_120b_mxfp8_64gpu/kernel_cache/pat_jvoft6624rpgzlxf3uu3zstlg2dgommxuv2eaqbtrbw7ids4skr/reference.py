from torch import device
import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3, in4):
        v5 = torch.ops.aten.gather.default(in0, 0, in1);  in0 = in1 = None
        v6 = torch.ops.aten._to_copy.default(v5, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'));  v5 = None
        v7 = torch.ops.aten.mul.Tensor(v6, in2);  in2 = None
        v8 = torch.ops.aten.mul.Tensor(v6, in3);  v6 = in3 = None
        v9 = torch.ops.aten._to_copy.default(v8, dtype = torch.bfloat16);  v8 = None
        v10 = torch.ops.aten.sum.dim_IntList(v7, [1], True);  v7 = None
        v11 = torch.ops.aten.reshape.default(v10, [65536]);  v10 = None
        v12 = torch.ops.aten.new_zeros.default(v11, [65536], dtype = torch.float32, layout = torch.strided, device = device(type='cuda'))
        v13 = torch.ops.aten.index_put.default(v12, [in4], v11, True);  v12 = in4 = v11 = None
        v14 = torch.ops.aten.reshape.default(v13, [16384, 4]);  v13 = None
        return (v9, v14)


def run(*args):
    return Reference()(*args)
