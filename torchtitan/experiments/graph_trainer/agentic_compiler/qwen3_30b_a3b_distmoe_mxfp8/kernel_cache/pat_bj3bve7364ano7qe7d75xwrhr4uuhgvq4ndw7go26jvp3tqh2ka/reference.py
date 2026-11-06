import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1):
        v2 = torch.ops.aten.linalg_vector_norm.default(in0, 1, [-1], True)
        v3 = torch.ops.aten.clamp_min.default(v2, 1e-12)
        v4 = torch.ops.aten.expand.default(v3, [32768, 128]);  v3 = None
        v5 = torch.ops.aten.div.Tensor(in0, v4);  in0 = None
        v6 = torch.ops.aten.mul.Tensor(v5, in1);  v5 = in1 = None
        v7 = torch.ops.aten.sum.dim_IntList(v6, [0]);  v6 = None
        return (v2, v4, v7)


def run(*args):
    return Reference()(*args)
