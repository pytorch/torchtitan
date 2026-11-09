import torch
import torchao.prototype.mx_formats.kernels  # torchao::mxfp8_quantize, torchao::triton_mx_block_rearrange


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2):
        v3 = torch.ops.torchao.mxfp8_quantize.default(in0, True, False, 32, 1, 'e4m3', 'rceil');  in0 = None
        v4 = v3[0]
        v5 = v3[2];  v3 = None
        v6 = torch.ops.torchao.triton_mx_block_rearrange.default(v5);  v5 = None
        v7 = torch.ops.aten._scaled_mm_v2.default(v4, in1, [v6], [3], [1], [in2], [3], [1], None, torch.bfloat16);  v4 = in1 = v6 = in2 = None
        return v7


def run(*args):
    return Reference()(*args)
