import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3, in4, in5):
        v6 = torch.ops.aten.mul.Tensor(in0, 1.0);  in0 = None
        v7 = torch.ops.aten.neg.default(v6)
        v8 = torch.ops.aten.div.Tensor(in1, in2);  in1 = None
        v9 = torch.ops.aten.div.Tensor(v8, in2);  v8 = None
        v10 = torch.ops.aten.mul.Tensor(v7, v9);  v7 = v9 = None
        v11 = torch.ops.aten.div.Tensor(v6, in2);  v6 = in2 = None
        v12 = torch.ops.aten.sum.dim_IntList(v10, [1], True);  v10 = None
        v13 = torch.ops.aten.expand.default(v12, [16384, 4]);  v12 = None
        v14 = torch.ops.aten.add.Tensor(v11, v13);  v11 = v13 = None
        v15 = torch.ops.aten.new_zeros.default(v14, [16384, 128])
        v16 = torch.ops.aten.scatter_add.default(v15, -1, in3, v14);  v15 = in3 = v14 = None
        v17 = torch.ops.aten.add.Tensor(in4, v16);  in4 = v16 = None
        v18 = torch.ops.aten.alias.default(in5);  in5 = None
        v19 = torch.ops.aten._softmax_backward_data.default(v17, v18, -1, torch.float32);  v17 = v18 = None
        return v19


def run(*args):
    return Reference()(*args)
