import torch
import torchao.prototype.mx_formats.kernels  # torchao::mxfp8_quantize, torchao::triton_mx_block_rearrange


class Reference(torch.nn.Module):
    def forward(self, in0, in1):
        v2 = torch.ops.aten._fused_rms_norm.default(in0, [7168], in1, 1e-06);  in0 = in1 = None
        v3 = v2[0]
        v4 = torch.ops.torchao.mxfp8_quantize.default(v3, True, False, 32, 1, 'e4m3', 'rceil')
        v5 = v4[0]
        v6 = v4[2];  v4 = None
        v7 = torch.ops.torchao.triton_mx_block_rearrange.default(v6);  v6 = None
        v8 = v2[1];  v2 = None
        v9 = torch.ops.torchao.mxfp8_quantize.default(v3, True, False, 32, 1, 'e4m3', 'rceil')
        v10 = v9[0]
        v11 = v9[2];  v9 = None
        v12 = torch.ops.torchao.triton_mx_block_rearrange.default(v11);  v11 = None
        return (v3, v5, v7, v8, v10, v12)


def run(*args):
    return Reference()(*args)
