import torch
import torchao.prototype.mx_formats.kernels  # torchao::mxfp8_quantize, torchao::triton_mx_block_rearrange


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2):
        v3 = torch.ops.aten.view.default(in0, [7168]);  in0 = None
        v4 = torch.ops.aten.add.Tensor(in1, in2);  in1 = in2 = None
        v5 = torch.ops.aten._fused_rms_norm.default(v4, [7168], v3, 1e-06);  v3 = None
        v6 = v5[0];  v5 = None
        v7 = torch.ops.torchao.mxfp8_quantize.default(v6, True, False, 32, 1, 'e4m3', 'rceil')
        v8 = v7[0]
        v9 = v7[2];  v7 = None
        v10 = torch.ops.torchao.triton_mx_block_rearrange.default(v9);  v9 = None
        return (v4, v6, v8, v10)


def run(*args):
    return Reference()(*args)
