import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1):
        v2 = torch.ops.aten.mul.Scalar(in0, 1);  in0 = None
        v3 = torch.ops.aten.mul.Tensor(in1, v2);  in1 = v2 = None
        return v3


def run(*args):
    return Reference()(*args)
