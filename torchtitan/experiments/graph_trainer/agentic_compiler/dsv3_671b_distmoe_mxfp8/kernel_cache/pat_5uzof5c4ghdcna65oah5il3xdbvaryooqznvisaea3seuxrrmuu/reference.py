import torch
import dist_moe._blockscaled  # dist_moe::prepare_mxfp8_weight


class Reference(torch.nn.Module):
    def forward(self, in0):
        v1 = torch.ops.aten.view.default(in0, [4, 7168, 2048]);  in0 = None
        v2 = torch.ops.dist_moe.prepare_mxfp8_weight.default(v1);  v1 = None
        v3 = v2[0]
        v4 = v2[1]
        v5 = v2[2];  v2 = None
        return (v3, v4, v5)


def run(*args):
    return Reference()(*args)
