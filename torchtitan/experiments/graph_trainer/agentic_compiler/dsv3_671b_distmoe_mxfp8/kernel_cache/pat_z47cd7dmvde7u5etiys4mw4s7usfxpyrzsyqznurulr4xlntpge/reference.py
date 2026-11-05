import torch
import torchao.prototype.mx_formats.kernels  # torchao::mxfp8_quantize, torchao::triton_mx_block_rearrange


class Reference(torch.nn.Module):
    def forward(self, in0, in1):
        v2 = torch.ops.aten.view.default(in0, [7168]);  in0 = None
        v3 = torch.ops.aten._fused_rms_norm.default(in1, [7168], v2, 1e-06);  in1 = v2 = None
        v4 = v3[0];  v3 = None
        v5 = torch.ops.torchao.mxfp8_quantize.default(v4, True, False, 32, 1, 'e4m3', 'rceil')
        v6 = v5[0]
        v7 = v5[2];  v5 = None
        v8 = torch.ops.torchao.triton_mx_block_rearrange.default(v7);  v7 = None
        v9 = torch.ops.torchao.mxfp8_quantize.default(v4, True, False, 32, 1, 'e4m3', 'rceil');  v4 = None
        v10 = v9[0]
        v11 = v9[2];  v9 = None
        v12 = torch.ops.torchao.triton_mx_block_rearrange.default(v11);  v11 = None
        return (v6, v8, v10, v12)


def run(*args):
    return Reference()(*args)
