import torch


class Reference(torch.nn.Module):
    def forward(self, in0):
        v1 = torch.ops.aten.unbind.int(in0, -2);  in0 = None
        v2 = v1[0]
        v3 = v1[1];  v1 = None
        v4 = torch.ops.aten.clamp.default(v2, None, 7.0)
        v5 = torch.ops.aten.clamp.default(v3, -7.0, 7.0)
        v6 = torch.ops.aten.mul.Tensor(v4, 1.702)
        v7 = torch.ops.aten.sigmoid.default(v6);  v6 = None
        v8 = torch.ops.aten.alias.default(v7)
        v9 = torch.ops.aten.mul.Tensor(v4, v7)
        v10 = torch.ops.aten.addcmul.default(v9, v9, v5)
        return (v2, v3, v4, v5, v7, v8, v9, v10)


def run(*args):
    return Reference()(*args)
