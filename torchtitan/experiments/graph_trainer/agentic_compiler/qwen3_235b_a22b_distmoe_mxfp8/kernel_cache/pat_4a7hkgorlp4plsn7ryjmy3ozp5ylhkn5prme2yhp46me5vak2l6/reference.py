import torch
import torch._inductor.fx_passes.bucketing  # bucketing::_pre_bucket_all_gather


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3, in4, in5, in6):
        v7 = torch.ops.aten._to_copy.default(in0, dtype = torch.bfloat16);  in0 = None
        v8 = torch.ops.aten.constant_pad_nd.default(v7, [0, 0], 0.0);  v7 = None
        v9 = torch.ops.aten._to_copy.default(in1, dtype = torch.bfloat16);  in1 = None
        v10 = torch.ops.aten.constant_pad_nd.default(v9, [0, 0], 0.0);  v9 = None
        v11 = torch.ops.aten._to_copy.default(in2, dtype = torch.bfloat16);  in2 = None
        v12 = torch.ops.aten.constant_pad_nd.default(v11, [0, 0, 0, 0], 0.0);  v11 = None
        v13 = torch.ops.bucketing._pre_bucket_all_gather.default([in3, in4, v8, v10, v12, in5, in6], 256, torch.bfloat16, [1, 1, 1, 1, 1, 1, 1], 0, [0, 1, 5, 6, -1, 2, 3, 4]);  in3 = in4 = v8 = v10 = v12 = in5 = in6 = None
        v14 = torch.ops.aten.slice.Tensor(v13, 0, 0, 282658)
        return (v13, v14)


def run(*args):
    return Reference()(*args)
