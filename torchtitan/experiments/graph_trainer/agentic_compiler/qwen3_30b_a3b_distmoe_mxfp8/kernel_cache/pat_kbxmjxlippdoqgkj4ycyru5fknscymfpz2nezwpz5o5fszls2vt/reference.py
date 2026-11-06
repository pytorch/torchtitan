import torch
import torchao.prototype.mx_formats.kernels  # torchao::mxfp8_quantize, torchao::triton_mx_block_rearrange


class Reference(torch.nn.Module):
    def forward(self, in0, in1):
        v2 = torch.ops.aten._fused_rms_norm.default(in0, [2048], in1, 1e-06);  in0 = in1 = None
        v3 = v2[0]
        v4 = v2[1];  v2 = None
        v5 = torch.ops.aten.alias.default(v4);  v4 = None
        v6 = torch.ops.torchao.mxfp8_quantize.default(v3, True, False, 32, 1, 'e4m3', 'rceil')
        v7 = v6[0]
        v8 = v6[2];  v6 = None
        v9 = torch.ops.torchao.triton_mx_block_rearrange.default(v8);  v8 = None
        return (v3, v5, v7, v9)


def run(*args):
    return Reference()(*args)
