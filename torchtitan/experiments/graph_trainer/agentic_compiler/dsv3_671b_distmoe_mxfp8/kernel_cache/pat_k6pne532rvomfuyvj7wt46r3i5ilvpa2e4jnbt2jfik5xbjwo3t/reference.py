import torch
import torchao.prototype.mx_formats.kernels  # torchao::mxfp8_quantize, torchao::triton_mx_block_rearrange
import torchtitan_recipes.overrides.fused_mla  # torchtitan::fused_mla_kv_backward


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3):
        v4 = torch.ops.aten.unsqueeze.default(in0, 0);  in0 = None
        v5 = torch.ops.aten.unsqueeze.default(in1, 0);  in1 = None
        v6 = torch.ops.torchtitan.fused_mla_kv_backward.default(v5, v4, in2, in3, 128, 64);  v5 = v4 = in2 = in3 = None
        v7 = v6[0]
        v8 = v6[1];  v6 = None
        v9 = torch.ops.aten.squeeze.dim(v8, 0);  v8 = None
        v10 = torch.ops.aten.squeeze.dim(v7, 0);  v7 = None
        v11 = torch.ops.aten.reshape.default(v10, [4096, 32768]);  v10 = None
        v12 = torch.ops.torchao.mxfp8_quantize.default(v11, True, True, 32, 32, 'e4m3', 'rceil');  v11 = None
        v13 = v12[0]
        v14 = v12[1]
        v15 = v12[2]
        v16 = v12[3];  v12 = None
        v17 = torch.ops.torchao.triton_mx_block_rearrange.default(v15);  v15 = None
        v18 = torch.ops.torchao.triton_mx_block_rearrange.default(v16);  v16 = None
        v19 = torch.ops.aten.t.default(v14);  v14 = None
        return (v9, v13, v17, v18, v19)


def run(*args):
    return Reference()(*args)
