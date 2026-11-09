from torch import device
import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3, in4):
        v5 = torch.ops.aten.view.default(in0, [256]);  in0 = None
        v6 = torch.ops.aten.slice.Tensor(v5, 0, 0, 128);  v5 = None
        v7 = torch.ops.aten.view.default(in1, [256]);  in1 = None
        v8 = torch.ops.aten.slice.Tensor(v7, 0, 0, 128);  v7 = None
        v9 = torch.ops.aten.reshape.default(in2, [8192, -1, 18, 128]);  in2 = None
        v10 = torch.ops.aten.split_with_sizes.default(v9, [16, 1, 1], -2);  v9 = None
        v11 = v10[0]
        v12 = torch.ops.aten.clone.default(v11, memory_format = torch.contiguous_format);  v11 = None
        v13 = torch.ops.aten.reshape.default(v12, [8192, 64, 128]);  v12 = None
        v14 = torch.ops.aten._fused_rms_norm.default(v13, [128], v6, 1e-06)
        v15 = v14[0]
        v16 = torch.ops.aten._to_copy.default(v15, dtype = torch.float32);  v15 = None
        v17 = torch.ops.aten.mul.Tensor(v16, in3)
        v18 = torch.ops.aten.slice.Tensor(v16, 2, 0, 64)
        v19 = torch.ops.aten.slice.Tensor(v16, 2, 64, 9223372036854775807);  v16 = None
        v20 = torch.ops.aten.neg.default(v19);  v19 = None
        v21 = torch.ops.aten.cat.default([v20, v18], -1);  v20 = v18 = None
        v22 = torch.ops.aten.mul.Tensor(v21, in4);  v21 = None
        v23 = torch.ops.aten.add.Tensor(v17, v22);  v17 = v22 = None
        v24 = torch.ops.aten._to_copy.default(v23, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v23 = None
        v25 = v14[1];  v14 = None
        v26 = torch.ops.aten.alias.default(v25);  v25 = None
        v27 = v10[1]
        v28 = torch.ops.aten.reshape.default(v27, [8192, 4, 128]);  v27 = None
        v29 = torch.ops.aten.clone.default(v28, memory_format = torch.contiguous_format);  v28 = None
        v30 = torch.ops.aten._fused_rms_norm.default(v29, [128], v8, 1e-06)
        v31 = v30[0]
        v32 = torch.ops.aten._to_copy.default(v31, dtype = torch.float32);  v31 = None
        v33 = torch.ops.aten.mul.Tensor(v32, in3);  in3 = None
        v34 = torch.ops.aten.slice.Tensor(v32, 2, 0, 64)
        v35 = torch.ops.aten.slice.Tensor(v32, 2, 64, 9223372036854775807);  v32 = None
        v36 = torch.ops.aten.neg.default(v35);  v35 = None
        v37 = torch.ops.aten.cat.default([v36, v34], -1);  v36 = v34 = None
        v38 = torch.ops.aten.mul.Tensor(v37, in4);  v37 = in4 = None
        v39 = torch.ops.aten.add.Tensor(v33, v38);  v33 = v38 = None
        v40 = torch.ops.aten._to_copy.default(v39, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda'));  v39 = None
        v41 = v30[1];  v30 = None
        v42 = torch.ops.aten.alias.default(v41);  v41 = None
        v43 = v10[2];  v10 = None
        v44 = torch.ops.aten.reshape.default(v43, [8192, 4, 128]);  v43 = None
        v45 = torch.ops.aten.clone.default(v44, memory_format = torch.contiguous_format);  v44 = None
        return (v6, v8, v13, v24, v26, v29, v40, v42, v45)


def run(*args):
    return Reference()(*args)
