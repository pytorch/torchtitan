from torch import device
import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3, in4, in5, in6, in7):
        v8 = torch.ops.aten.mul.Scalar(in0, 1);  in0 = None
        v9 = torch.ops.aten.mul.Tensor(in1, v8);  in1 = v8 = None
        v10 = torch.ops.aten.mul.Tensor(in2, in3);  in3 = None
        v11 = torch.ops.aten.mul.Tensor(in2, in4);  in2 = in4 = None
        v12 = torch.ops.aten.alias.default(in5);  in5 = None
        v13 = torch.ops.aten.sigmoid_backward.default(v10, v12);  v10 = v12 = None
        v14 = torch.ops.aten.mul.Tensor(v13, 1.702);  v13 = None
        v15 = torch.ops.aten.add.Tensor(v11, v14);  v11 = v14 = None
        v16 = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'))
        v17 = torch.ops.aten.gt.Scalar(in6, -7.0)
        v18 = torch.ops.aten.lt.Scalar(in6, 7.0);  in6 = None
        v19 = torch.ops.aten.logical_and_.default(v17, v18);  v17 = v18 = None
        v20 = torch.ops.aten.where.self(v19, v9, v16);  v19 = v9 = v16 = None
        v21 = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'))
        v22 = torch.ops.aten.lt.Scalar(in7, 7.0);  in7 = None
        v23 = torch.ops.aten.where.self(v22, v15, v21);  v22 = v15 = v21 = None
        v24 = torch.ops.aten.stack.default([v23, v20], -2);  v23 = v20 = None
        return v24


def run(*args):
    return Reference()(*args)
