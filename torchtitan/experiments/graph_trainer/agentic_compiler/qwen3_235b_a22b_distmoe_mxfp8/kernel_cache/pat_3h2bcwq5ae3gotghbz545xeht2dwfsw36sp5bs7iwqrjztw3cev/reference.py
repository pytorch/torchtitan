import torch
import torch._inductor.fx_passes.bucketing  # bucketing::_pre_bucket_reduce_scatter


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3, in4, in5, in6):
        v7 = torch.ops.bucketing._pre_bucket_reduce_scatter.default([in0, in1, in2, in3, in4, in5, in6], 256, [2, 3, 4]);  in0 = in1 = in2 = in3 = in4 = in5 = in6 = None
        return v7


def run(*args):
    return Reference()(*args)
