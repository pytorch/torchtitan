import torch
import torchao.prototype.mx_formats.kernels  # torchao::mxfp8_quantize, torchao::triton_mx_block_rearrange


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3, in4):
        v5 = torch.ops.aten.alias.default(in0);  in0 = None
        v6 = torch.ops.aten._fused_rms_norm_backward.default(in1, in2, [7168], v5, in3, [True, True]);  in1 = in2 = v5 = in3 = None
        v7 = v6[0]
        v8 = v6[1];  v6 = None
        v9 = torch.ops.aten.add.Tensor(in4, v7);  in4 = v7 = None
        v10 = torch.ops.torchao.mxfp8_quantize.default(v9, True, True, 32, 32, 'e4m3', 'rceil')
        v11 = v10[0]
        v12 = v10[1]
        v13 = v10[2]
        v14 = v10[3];  v10 = None
        v15 = torch.ops.torchao.triton_mx_block_rearrange.default(v13);  v13 = None
        v16 = torch.ops.torchao.triton_mx_block_rearrange.default(v14);  v14 = None
        v17 = torch.ops.aten.t.default(v12);  v12 = None
        return (v8, v9, v11, v15, v16, v17)


def run(*args):
    return Reference()(*args)
