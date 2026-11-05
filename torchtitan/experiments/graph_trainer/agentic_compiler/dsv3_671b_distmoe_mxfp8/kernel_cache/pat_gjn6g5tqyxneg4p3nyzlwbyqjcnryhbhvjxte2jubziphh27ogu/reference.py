from math import inf
import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2, in3):
        v4 = torch.ops.aten.sigmoid.default(in0);  in0 = None
        v5 = torch.ops.aten.add.Tensor(v4, in1);  in1 = None
        v6 = torch.ops.aten.reshape.default(v5, [4096, 8, 32]);  v5 = None
        v7 = torch.ops.aten.topk.default(v6, 2)
        v8 = v7[0];  v7 = None
        v9 = torch.ops.aten.sum.dim_IntList(v8, [-1]);  v8 = None
        v10 = torch.ops.aten.topk.default(v9, 4, -1, True, False)
        v11 = v10[1];  v10 = None
        v12 = torch.ops.aten.ones_like.default(v9, dtype = torch.bool, pin_memory = False);  v9 = None
        v13 = torch.ops.aten.scatter_.value(v12, -1, v11, False);  v12 = v11 = None
        v14 = torch.ops.aten.unsqueeze.default(v13, -1);  v13 = None
        v15 = torch.ops.aten.masked_fill.Scalar(v6, v14, -inf);  v6 = v14 = None
        v16 = torch.ops.aten.reshape.default(v15, [4096, 256]);  v15 = None
        v17 = torch.ops.aten.topk.default(v16, 8, -1, True, False);  v16 = None
        v18 = v17[1];  v17 = None
        v19 = torch.ops.aten.gather.default(v4, -1, v18)
        v20 = torch.ops.aten.sum.dim_IntList(v19, [-1], True)
        v21 = torch.ops.aten.add.Tensor(v20, 1e-20);  v20 = None
        v22 = torch.ops.aten.div.Tensor(v19, v21);  v19 = v21 = None
        v23 = torch.ops.aten.mul.Tensor(v22, 2.5);  v22 = None
        v24 = torch.ops.aten.zeros_like.default(v4, dtype = torch.bool, pin_memory = False)
        v25 = torch.ops.aten.scatter_.value(v24, -1, v18, True);  v24 = None
        v26 = torch.ops.aten.unsqueeze.default(in2, -1)
        v27 = torch.ops.aten.bitwise_not.default(v26);  v26 = None
        v28 = torch.ops.aten.bitwise_and.Tensor(v25, v27);  v25 = v27 = None
        v29 = torch.ops.aten.sum.dim_IntList(v28, [0])
        v30 = torch.ops.aten._to_copy.default(v28, dtype = torch.float32)
        v31 = torch.ops.aten.sum.dim_IntList(v30, [0]);  v30 = None
        v32 = torch.ops.aten.linalg_vector_norm.default(v4, 1, [-1], True)
        v33 = torch.ops.aten.clamp_min.default(v32, 1e-12);  v32 = None
        v34 = torch.ops.aten.expand.default(v33, [4096, 256]);  v33 = None
        v35 = torch.ops.aten.div.Tensor(v4, v34);  v4 = v34 = None
        v36 = torch.ops.aten.unsqueeze.default(in2, -1)
        v37 = torch.ops.aten.bitwise_not.default(v36);  v36 = None
        v38 = torch.ops.aten.mul.Tensor(v35, v37);  v35 = v37 = None
        v39 = torch.ops.aten.sum.dim_IntList(v38, [0]);  v38 = None
        v40 = torch.ops.aten.linalg_vector_norm.default(v31, 1, [0], True)
        v41 = torch.ops.aten.clamp_min.default(v40, 1e-12);  v40 = None
        v42 = torch.ops.aten.expand.default(v41, [256]);  v41 = None
        v43 = torch.ops.aten.div.Tensor(v31, v42);  v31 = v42 = None
        v44 = torch.ops.aten.mul.Tensor(v43, 256);  v43 = None
        v45 = torch.ops.aten.mul.Tensor(v44, v39);  v44 = v39 = None
        v46 = torch.ops.aten.sum.default(v45);  v45 = None
        v47 = torch.ops.aten.clamp_min.default(in3, 1);  in3 = None
        v48 = torch.ops.aten.reciprocal.default(v47);  v47 = None
        v49 = torch.ops.aten.mul.Tensor(v48, 1.0);  v48 = None
        v50 = torch.ops.aten.mul.Tensor(v46, v49);  v46 = v49 = None
        v51 = torch.ops.aten.unsqueeze.default(in2, -1);  in2 = None
        v52 = torch.ops.aten.masked_fill.Scalar(v18, v51, -1)
        v53 = torch.ops.aten.masked_fill.Scalar(v23, v51, 0.0);  v23 = v51 = None
        return (v18, v28, v29, v50, v52, v53)


def run(*args):
    return Reference()(*args)
