import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1):
        v2 = torch.ops.aten._softmax.default(in0, -1, False);  in0 = None
        v3 = torch.ops.aten.alias.default(v2)
        v4 = torch.ops.aten.gather.default(v2, -1, in1);  in1 = None
        v5 = torch.ops.aten.sum.dim_IntList(v4, [-1], True)
        v6 = torch.ops.aten.add.Tensor(v5, 1e-20);  v5 = None
        v7 = torch.ops.aten.div.Tensor(v4, v6)
        v8 = torch.ops.aten.mul.Tensor(v7, 1.0);  v7 = None
        return (v2, v3, v4, v6, v8)


def run(*args):
    return Reference()(*args)
