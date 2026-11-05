import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3, in4, in5):
        v6 = torch.ops.aten.sigmoid.default(in0);  in0 = None
        v7 = torch.ops.aten.gather.default(v6, -1, in1);  in1 = None
        v8 = torch.ops.aten.sum.dim_IntList(v7, [-1], True)
        v9 = torch.ops.aten.add.Tensor(v8, 1e-20);  v8 = None
        v10 = torch.ops.aten.div.Tensor(v7, v9)
        v11 = torch.ops.aten.mul.Tensor(v10, 2.5);  v10 = None
        v12 = torch.ops.aten.masked_fill.Scalar(v11, in2, 0.0);  v11 = in2 = None
        v13 = torch.ops.aten._to_copy.default(in3, dtype = torch.float32);  in3 = None
        v14 = torch.ops.aten.sum.dim_IntList(v13, [0]);  v13 = None
        v15 = torch.ops.aten.linalg_vector_norm.default(v6, 1, [-1], True)
        v16 = torch.ops.aten.clamp_min.default(v15, 1e-12)
        v17 = torch.ops.aten.expand.default(v16, [4096, 256]);  v16 = None
        v18 = torch.ops.aten.div.Tensor(v6, v17)
        v19 = torch.ops.aten.unsqueeze.default(in4, -1);  in4 = None
        v20 = torch.ops.aten.bitwise_not.default(v19);  v19 = None
        v21 = torch.ops.aten.mul.Tensor(v18, v20);  v18 = None
        v22 = torch.ops.aten.sum.dim_IntList(v21, [0]);  v21 = None
        v23 = torch.ops.aten.linalg_vector_norm.default(v14, 1, [0], True)
        v24 = torch.ops.aten.clamp_min.default(v23, 1e-12);  v23 = None
        v25 = torch.ops.aten.expand.default(v24, [256]);  v24 = None
        v26 = torch.ops.aten.div.Tensor(v14, v25);  v14 = v25 = None
        v27 = torch.ops.aten.mul.Tensor(v26, 256);  v26 = None
        v28 = torch.ops.aten.mul.Tensor(v27, v22);  v22 = None
        v29 = torch.ops.aten.sum.default(v28);  v28 = None
        v30 = torch.ops.aten.clamp_min.default(in5, 1);  in5 = None
        v31 = torch.ops.aten.reciprocal.default(v30);  v30 = None
        v32 = torch.ops.aten.mul.Tensor(v31, 1.0);  v31 = None
        v33 = torch.ops.aten.mul.Tensor(v32, 0.001);  v32 = None
        v34 = torch.ops.aten.mul.Tensor(v29, v33);  v29 = None
        v35 = torch.ops.aten.alias.default(v6)
        return (v6, v7, v9, v12, v15, v17, v20, v27, v33, v34, v35)


def run(*args):
    return Reference()(*args)
