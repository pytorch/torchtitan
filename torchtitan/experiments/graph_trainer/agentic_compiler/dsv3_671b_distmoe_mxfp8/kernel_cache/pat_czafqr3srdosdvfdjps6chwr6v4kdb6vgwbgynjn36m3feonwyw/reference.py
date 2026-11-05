import torch
import torch._inductor.fx_passes.bucketing  # bucketing::_pre_bucket_reduce_scatter


class Reference(torch.nn.Module):
    def forward(self, in0, in1):
        v2 = torch.ops.aten.reshape.default(in0, [4, 2, 2048, 7168]);  in0 = None
        v3 = torch.ops.bucketing._pre_bucket_reduce_scatter.default([in1, v2], 4);  in1 = v2 = None
        return v3


def run(*args):
    return Reference()(*args)
