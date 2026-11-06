import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2):
        v3 = torch.ops.aten.add.Tensor(in0, in1);  in0 = in1 = None
        v4 = torch.ops.aten._fused_rms_norm.default(v3, [2048], in2, 1e-06);  in2 = None
        v5 = v4[0]
        v6 = v4[1];  v4 = None
        v7 = torch.ops.aten.alias.default(v6);  v6 = None
        return (v3, v5, v7)


def run(*args):
    return Reference()(*args)
