import torch
import torchao.prototype.mx_formats.kernels  # torchao::mxfp8_quantize, torchao::triton_mx_block_rearrange


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3, in4, in5):
        v6 = torch.ops.aten.add.Tensor(in0, in1);  in0 = in1 = None
        v7 = torch.ops.aten.alias.default(in2);  in2 = None
        v8 = torch.ops.aten._fused_rms_norm_backward.default(v6, in3, [2048], v7, in4, [True, True]);  v6 = in3 = v7 = in4 = None
        v9 = v8[0]
        v10 = v8[1];  v8 = None
        v11 = torch.ops.aten.add.Tensor(in5, v9);  in5 = v9 = None
        v12 = torch.ops.torchao.mxfp8_quantize.default(v11, True, True, 32, 32, 'e4m3', 'rceil')
        v13 = v12[0]
        v14 = v12[1]
        v15 = v12[2]
        v16 = v12[3];  v12 = None
        v17 = torch.ops.torchao.triton_mx_block_rearrange.default(v15);  v15 = None
        v18 = torch.ops.torchao.triton_mx_block_rearrange.default(v16);  v16 = None
        v19 = torch.ops.aten.t.default(v14);  v14 = None
        return (v10, v11, v13, v17, v18, v19)


def run(*args):
    return Reference()(*args)
