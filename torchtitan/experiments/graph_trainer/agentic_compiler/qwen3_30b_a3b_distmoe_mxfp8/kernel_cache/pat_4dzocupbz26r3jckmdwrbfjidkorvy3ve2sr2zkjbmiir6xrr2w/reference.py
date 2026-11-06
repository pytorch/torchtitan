from torch import device
import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3, in4):
        v5 = torch.ops.aten.reshape.default(in0, [32768, -1, 10, 128]);  in0 = None
        v6 = torch.ops.aten.split_with_sizes.default(v5, [8, 1, 1], -2);  v5 = None
        v7 = v6[0]
        v8 = v6[1]
        v9 = v6[2];  v6 = None
        v10 = torch.ops.aten.clone.default(v7, memory_format = torch.contiguous_format);  v7 = None
        v11 = torch.ops.aten.reshape.default(v10, [32768, 32, 128]);  v10 = None
        v12 = torch.ops.aten.reshape.default(v8, [32768, 4, 128]);  v8 = None
        v13 = torch.ops.aten.clone.default(v12, memory_format = torch.contiguous_format);  v12 = None
        v14 = torch.ops.aten.reshape.default(v9, [32768, 4, 128]);  v9 = None
        v15 = torch.ops.aten.clone.default(v14, memory_format = torch.contiguous_format);  v14 = None
        v16 = torch.ops.aten._fused_rms_norm.default(v11, [128], in1, 1e-06);  in1 = None
        v17 = v16[0]
        v18 = v16[1];  v16 = None
        v19 = torch.ops.aten.alias.default(v18);  v18 = None
        v20 = torch.ops.aten._fused_rms_norm.default(v13, [128], in2, 1e-06);  in2 = None
        v21 = v20[0]
        v22 = v20[1];  v20 = None
        v23 = torch.ops.aten.alias.default(v22);  v22 = None
        v24 = torch.ops.aten.index.Tensor(in3, [in4]);  in3 = in4 = None
        v25 = torch.ops.aten.reshape.default(v24, [32768, 1, 256]);  v24 = None
        v26 = torch.ops.aten.slice.Tensor(v25, 2, 0, 128)
        v27 = torch.ops.aten.slice.Tensor(v25, 2, 128, 9223372036854775807);  v25 = None
        v28 = torch.ops.aten._to_copy.default(v17, dtype = torch.float32);  v17 = None
        v29 = torch.ops.aten.mul.Tensor(v28, v26)
        v30 = torch.ops.aten.slice.Tensor(v28, 2, 0, 64)
        v31 = torch.ops.aten.slice.Tensor(v28, 2, 64, 9223372036854775807);  v28 = None
        v32 = torch.ops.aten.neg.default(v31);  v31 = None
        v33 = torch.ops.aten.cat.default([v32, v30], -1);  v32 = v30 = None
        v34 = torch.ops.aten.mul.Tensor(v33, v27);  v33 = None
        v35 = torch.ops.aten.add.Tensor(v29, v34);  v29 = v34 = None
        v36 = torch.ops.aten._to_copy.default(v21, dtype = torch.float32);  v21 = None
        v37 = torch.ops.aten.mul.Tensor(v36, v26)
        v38 = torch.ops.aten.slice.Tensor(v36, 2, 0, 64)
        v39 = torch.ops.aten.slice.Tensor(v36, 2, 64, 9223372036854775807);  v36 = None
        v40 = torch.ops.aten.neg.default(v39);  v39 = None
        v41 = torch.ops.aten.cat.default([v40, v38], -1);  v40 = v38 = None
        v42 = torch.ops.aten.mul.Tensor(v41, v27);  v41 = None
        v43 = torch.ops.aten.add.Tensor(v37, v42);  v37 = v42 = None
        v44 = torch.ops.aten._to_copy.default(v35, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v35 = None
        v45 = torch.ops.aten._to_copy.default(v43, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v43 = None
        return (v11, v13, v15, v19, v23, v26, v27, v44, v45)


def run(*args):
    return Reference()(*args)
