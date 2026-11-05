import torch
import torch._inductor.fx_passes.bucketing  # bucketing::_pre_bucket_reduce_scatter


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14):
        v15 = torch.ops.aten._scaled_mm_v2.default(in0, in1, [in2], [3], [1], [in3], [3], [1], None, torch.bfloat16);  in0 = in1 = in2 = in3 = None
        v16 = torch.ops.bucketing._pre_bucket_reduce_scatter.default([in4, in5, in6, in7, in8, in9, v15, in10, in11, in12, in13, in14], 256, [3]);  in4 = in5 = in6 = in7 = in8 = in9 = v15 = in10 = in11 = in12 = in13 = in14 = None
        return v16


def run(*args):
    return Reference()(*args)
