import torch
import torch._inductor.fx_passes.bucketing  # bucketing::_pre_bucket_reduce_scatter


class Reference(torch.nn.Module):
    def forward(self, in0, in1):
        v2 = torch.ops.aten.reshape.default(in0, [4, 2, 1536, 4096]);  in0 = None
        v3 = torch.ops.aten.split.Tensor(in1, 512, 1);  in1 = None
        v4 = v3[0]
        v5 = v3[1]
        v6 = v3[2]
        v7 = v3[3]
        v8 = v3[4]
        v9 = v3[5]
        v10 = v3[6]
        v11 = v3[7];  v3 = None
        v12 = torch.ops.aten.cat.default([v4, v5, v6, v7, v8, v9, v10, v11]);  v4 = v5 = v6 = v7 = v8 = v9 = v10 = v11 = None
        v13 = torch.ops.aten.split.Tensor(v2, 192, 2);  v2 = None
        v14 = v13[0]
        v15 = v13[1]
        v16 = v13[2]
        v17 = v13[3]
        v18 = v13[4]
        v19 = v13[5]
        v20 = v13[6]
        v21 = v13[7];  v13 = None
        v22 = torch.ops.aten.cat.default([v14, v15, v16, v17, v18, v19, v20, v21]);  v14 = v15 = v16 = v17 = v18 = v19 = v20 = v21 = None
        v23 = torch.ops.bucketing._pre_bucket_reduce_scatter.default([v12, v22], 8);  v12 = v22 = None
        return v23


def run(*args):
    return Reference()(*args)
