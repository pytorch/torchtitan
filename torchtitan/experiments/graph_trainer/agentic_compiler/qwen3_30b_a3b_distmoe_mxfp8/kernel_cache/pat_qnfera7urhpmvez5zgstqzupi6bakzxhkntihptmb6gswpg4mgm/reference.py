from torch import device
import torch
import torchao.prototype.mx_formats.kernels  # torchao::mxfp8_quantize, torchao::triton_mx_block_rearrange


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10):
        v11 = torch.ops.aten._to_copy.default(in0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'));  in0 = None
        v12 = torch.ops.aten._to_copy.default(in1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'));  in1 = None
        v13 = torch.ops.aten.mul.Tensor(v11, in2)
        v14 = torch.ops.aten.slice.Tensor(v13, 2, 0, 64)
        v15 = torch.ops.aten.slice.Tensor(v13, 2, 64, 128);  v13 = None
        v16 = torch.ops.aten.neg.default(v14);  v14 = None
        v17 = torch.ops.aten.slice_backward.default(v16, [32768, 4, 128], 2, 64, 9223372036854775807, 1);  v16 = None
        v18 = torch.ops.aten.slice_backward.default(v15, [32768, 4, 128], 2, 0, 64, 1);  v15 = None
        v19 = torch.ops.aten.add.Tensor(v17, v18);  v17 = v18 = None
        v20 = torch.ops.aten.mul.Tensor(v11, in3);  v11 = None
        v21 = torch.ops.aten.add.Tensor(v19, v20);  v19 = v20 = None
        v22 = torch.ops.aten._to_copy.default(v21, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v21 = None
        v23 = torch.ops.aten.mul.Tensor(v12, in2);  in2 = None
        v24 = torch.ops.aten.slice.Tensor(v23, 2, 0, 64)
        v25 = torch.ops.aten.slice.Tensor(v23, 2, 64, 128);  v23 = None
        v26 = torch.ops.aten.neg.default(v24);  v24 = None
        v27 = torch.ops.aten.slice_backward.default(v26, [32768, 32, 128], 2, 64, 9223372036854775807, 1);  v26 = None
        v28 = torch.ops.aten.slice_backward.default(v25, [32768, 32, 128], 2, 0, 64, 1);  v25 = None
        v29 = torch.ops.aten.add.Tensor(v27, v28);  v27 = v28 = None
        v30 = torch.ops.aten.mul.Tensor(v12, in3);  v12 = in3 = None
        v31 = torch.ops.aten.add.Tensor(v29, v30);  v29 = v30 = None
        v32 = torch.ops.aten._to_copy.default(v31, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v31 = None
        v33 = torch.ops.aten.alias.default(in4);  in4 = None
        v34 = torch.ops.aten._fused_rms_norm_backward.default(v22, in5, [128], v33, in6, [True, True]);  v22 = in5 = v33 = in6 = None
        v35 = v34[0]
        v36 = v34[1];  v34 = None
        v37 = torch.ops.aten.alias.default(in7);  in7 = None
        v38 = torch.ops.aten._fused_rms_norm_backward.default(v32, in8, [128], v37, in9, [True, True]);  v32 = in8 = v37 = in9 = None
        v39 = v38[0]
        v40 = v38[1];  v38 = None
        v41 = torch.ops.aten.reshape.default(in10, [32768, 4, 1, 128]);  in10 = None
        v42 = torch.ops.aten.reshape.default(v35, [32768, 4, 1, 128]);  v35 = None
        v43 = torch.ops.aten.reshape.default(v39, [32768, 4, 8, 128]);  v39 = None
        v44 = torch.ops.aten.cat.default([v43, v42, v41], 2);  v43 = v42 = v41 = None
        v45 = torch.ops.aten.reshape.default(v44, [32768, 5120]);  v44 = None
        v46 = torch.ops.torchao.mxfp8_quantize.default(v45, True, True, 32, 32, 'e4m3', 'rceil');  v45 = None
        v47 = v46[0]
        v48 = v46[1]
        v49 = v46[2]
        v50 = v46[3];  v46 = None
        v51 = torch.ops.torchao.triton_mx_block_rearrange.default(v49);  v49 = None
        v52 = torch.ops.torchao.triton_mx_block_rearrange.default(v50);  v50 = None
        v53 = torch.ops.aten.t.default(v48);  v48 = None
        return (v36, v40, v47, v51, v52, v53)


def run(*args):
    return Reference()(*args)
