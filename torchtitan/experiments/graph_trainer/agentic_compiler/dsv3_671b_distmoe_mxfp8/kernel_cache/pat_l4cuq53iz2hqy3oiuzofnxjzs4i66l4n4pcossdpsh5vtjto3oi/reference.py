import torch
import torch._inductor.fx_passes.bucketing  # bucketing::_pre_bucket_all_gather


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11):
        v12 = torch.ops.aten._to_copy.default(in0, dtype = torch.bfloat16);  in0 = None
        v13 = torch.ops.aten.constant_pad_nd.default(v12, [0, 0, 0, 0], 0.0);  v12 = None
        v14 = torch.ops.bucketing._pre_bucket_all_gather.default([in1, in2, in3, v13, in4, in5, in6, in7, in8, in9, in10, in11], 256, torch.bfloat16, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, -1, 3]);  in1 = in2 = in3 = v13 = in4 = in5 = in6 = in7 = in8 = in9 = in10 = in11 = None
        v15 = torch.ops.aten.slice.Tensor(v14, 0, 0, 915520)
        return (v14, v15)


def run(*args):
    return Reference()(*args)
