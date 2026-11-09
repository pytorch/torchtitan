import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3, in4, in5, in6, in7):
        v8 = torch.ops.aten._scaled_mm_v2.default(in0, in1, [in2], [3], [1], [in3], [3], [1], None, torch.bfloat16);  in0 = in1 = in2 = in3 = None
        v9 = torch.ops.aten.alias.default(in4);  in4 = None
        v10 = torch.ops.aten._fused_rms_norm_backward.default(v8, in5, [4096], v9, in6, [True, True]);  v8 = in5 = v9 = in6 = None
        v11 = v10[0]
        v12 = v10[1];  v10 = None
        v13 = torch.ops.aten.add.Tensor(in7, v11);  in7 = v11 = None
        return (v12, v13)


def run(*args):
    return Reference()(*args)
