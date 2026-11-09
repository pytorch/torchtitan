import torch
import dist_moe._blockscaled  # dist_moe::prepare_mxfp8_weight


class Reference(torch.nn.Module):
    def forward(self, in0):
        v1 = torch.ops.aten.view.default(in0, [32, 2, 192, 4096]);  in0 = None
        v2 = torch.ops.aten.split.Tensor(v1, 4);  v1 = None
        v3 = v2[0]
        v4 = v2[1]
        v5 = v2[2]
        v6 = v2[3]
        v7 = v2[4]
        v8 = v2[5]
        v9 = v2[6]
        v10 = v2[7];  v2 = None
        v11 = torch.ops.aten.cat.default([v3, v4, v5, v6, v7, v8, v9, v10], 2);  v3 = v4 = v5 = v6 = v7 = v8 = v9 = v10 = None
        v12 = torch.ops.aten.reshape.default(v11, [4, 3072, 4096]);  v11 = None
        v13 = torch.ops.dist_moe.prepare_mxfp8_weight.default(v12);  v12 = None
        v14 = v13[0]
        v15 = v13[1]
        v16 = v13[2];  v13 = None
        return (v14, v15, v16)


def run(*args):
    return Reference()(*args)
