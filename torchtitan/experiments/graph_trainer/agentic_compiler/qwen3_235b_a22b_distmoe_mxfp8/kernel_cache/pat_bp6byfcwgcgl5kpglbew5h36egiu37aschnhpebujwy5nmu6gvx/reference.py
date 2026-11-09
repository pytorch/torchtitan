import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2):
        v3 = torch.ops.aten.zeros_like.default(in0, dtype = torch.bool, pin_memory = False);  in0 = None
        v4 = torch.ops.aten.scatter_.value(v3, -1, in1, True);  v3 = in1 = None
        v5 = torch.ops.aten.unsqueeze.default(in2, -1);  in2 = None
        v6 = torch.ops.aten.bitwise_not.default(v5);  v5 = None
        v7 = torch.ops.aten.bitwise_and.Tensor(v4, v6);  v4 = v6 = None
        v8 = torch.ops.aten.sum.dim_IntList(v7, [0])
        v9 = torch.ops.aten._to_copy.default(v7, dtype = torch.float32);  v7 = None
        v10 = torch.ops.aten.sum.dim_IntList(v9, [0]);  v9 = None
        v11 = torch.ops.aten.view.default(v10, [-1]);  v10 = None
        v12 = torch.ops.aten.cat.default([v11]);  v11 = None
        return (v8, v12)


def run(*args):
    return Reference()(*args)
