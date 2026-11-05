import torch
import torchao.prototype.mx_formats.kernels  # torchao::mxfp8_quantize, torchao::triton_mx_block_rearrange


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2):
        v3 = torch.ops.aten.add.Tensor(in0, in1);  in0 = in1 = None
        v4 = torch.ops.aten._fused_rms_norm.default(v3, [7168], in2, 1e-06);  in2 = None
        v5 = v4[0]
        v6 = torch.ops.torchao.mxfp8_quantize.default(v5, True, False, 32, 1, 'e4m3', 'rceil')
        v7 = v6[0]
        v8 = v6[2];  v6 = None
        v9 = torch.ops.torchao.triton_mx_block_rearrange.default(v8);  v8 = None
        v10 = v4[1];  v4 = None
        return (v3, v5, v7, v9, v10)


def run(*args):
    return Reference()(*args)
