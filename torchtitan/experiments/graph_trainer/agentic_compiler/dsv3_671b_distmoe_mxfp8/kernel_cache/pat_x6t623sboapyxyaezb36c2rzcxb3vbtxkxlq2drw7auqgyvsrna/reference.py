import torch
import dist_moe._blockscaled  # dist_moe::prepare_mxfp8_weight


class Reference(torch.nn.Module):
    def forward(self, in0):
        v1 = torch.ops.aten.view.default(in0, [4, 2, 2048, 7168]);  in0 = None
        v2 = torch.ops.aten.reshape.default(v1, [4, 4096, 7168]);  v1 = None
        v3 = torch.ops.dist_moe.prepare_mxfp8_weight.default(v2);  v2 = None
        v4 = v3[0]
        v5 = v3[1]
        v6 = v3[2];  v3 = None
        return (v4, v5, v6)


def run(*args):
    return Reference()(*args)
